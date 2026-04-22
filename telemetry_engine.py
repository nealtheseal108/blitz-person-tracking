"""
Vending-machine conversion-funnel telemetry engine.

Pipeline per frame:
    grab frame ─► YOLOv8 person detection ─► centroid tracker ─►
    funnel state machine ─► annotate + write event log

Run:
    python telemetry_engine.py --config config.yaml
    python telemetry_engine.py --source path/to/clip.mp4
    python telemetry_engine.py --source rtsp://user:pass@10.0.0.42/stream1

Outputs land in `runs/<timestamp>/`:
    annotated.mp4   – frames with bboxes, IDs, trails, zones, HUD
    events.jsonl    – one JSON event per line (visitor_seen,
                      engagement_start, transaction, track_lost)
    funnel.csv      – one row per unique track with dwell + outcomes
    summary.json    – aggregate counts and conversion rates
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import threading
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from detector import Detection, PersonDetector
from funnel import FunnelStage, FunnelTracker
from gaze import GazeConfig, GazeDetector
from privacy import PrivacyConfig, redact
from tracker import CentroidTracker
from zones import ZoneManager
from dashboard_server import Dashboard


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def open_source(source) -> cv2.VideoCapture:
    """Source can be int (webcam index), str path, or RTSP/HTTP URL."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    return cap


# Use distinct, deterministic colors per track so the demo video looks legible
_TRACK_COLORS = [
    (255, 99, 71), (60, 179, 113), (30, 144, 255), (255, 215, 0),
    (218, 112, 214), (255, 140, 0), (0, 206, 209), (220, 20, 60),
    (154, 205, 50), (138, 43, 226),
]


def color_for(track_id: int) -> tuple:
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


def transaction_target_point(zone_mgr: ZoneManager, frame_shape: tuple[int, int]) -> tuple[int, int]:
    """Best-effort machine target point used for approach-motion detection."""
    z = zone_mgr.zones.get(ZoneManager.TRANSACTION)
    if hasattr(z, "polygon"):
        poly = getattr(z, "polygon")
        cx = int(np.mean(poly[:, 0]))
        cy = int(np.mean(poly[:, 1]))
        return cx, cy
    # Fallback for non-polygon zones
    h, w = frame_shape[:2]
    return int(0.85 * w), int(0.75 * h)


def compute_approaching_ids(
    tracks: dict,
    machine_pt: tuple[int, int],
    min_move_px: float = 8.0,
    min_cosine: float = 0.55,
) -> set[int]:
    """Track ids actively moving toward machine target."""
    out: set[int] = set()
    mx, my = machine_pt
    for tid, t in tracks.items():
        if len(t.history) < 4:
            continue
        x0, y0 = t.history[-4]
        x1, y1 = t.history[-1]
        mvx, mvy = (x1 - x0), (y1 - y0)
        move_mag = math.hypot(mvx, mvy)
        if move_mag < min_move_px:
            continue
        tx, ty = (mx - x1), (my - y1)
        target_mag = math.hypot(tx, ty)
        if target_mag < 1e-6:
            continue
        cosine = (mvx * tx + mvy * ty) / (move_mag * target_mag)
        if cosine >= min_cosine:
            out.add(tid)
    return out


def clothing_histogram(frame: np.ndarray, det: Detection) -> np.ndarray | None:
    """Torso-focused color histogram used as a lightweight clothing signature."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 12 or y2 - y1 < 24:
        return None
    bh = y2 - y1
    ty1 = y1 + int(0.25 * bh)
    ty2 = y1 + int(0.85 * bh)
    roi = frame[ty1:ty2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)


def is_human_shape(
    det: Detection,
    frame_shape: tuple[int, int],
    min_aspect_ratio: float = 0.75,
    max_aspect_ratio: float = 4.8,
    min_area_frac: float = 0.003,
    max_area_frac: float = 0.55,
) -> bool:
    """Reject detections that don't match a standing human-like bbox shape."""
    h, w = frame_shape[:2]
    bw = max(1, det.x2 - det.x1)
    bh = max(1, det.y2 - det.y1)
    aspect = bh / float(bw)
    area_frac = (bw * bh) / float(max(1, h * w))
    if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
        return False
    if area_frac < min_area_frac or area_frac > max_area_frac:
        return False
    return True


def detect_upper_bodies(frame: np.ndarray) -> list[Detection]:
    """Secondary detector for upper-half humans."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_upperbody.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return []
    bodies = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=3,
        minSize=(36, 36),
    )
    out: list[Detection] = []
    h, w = frame.shape[:2]
    for (x, y, bw, bh) in bodies:
        x1 = max(0, int(x - 0.10 * bw))
        x2 = min(w, int(x + 1.10 * bw))
        y1 = max(0, int(y - 0.06 * bh))
        y2 = min(h, int(y + 2.10 * bh))
        out.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.55))
    return out


def iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.x1, a.y1, a.x2, a.y2
    bx1, by1, bx2, by2 = b.x1, b.y1, b.x2, b.y2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def merge_person_detections(primary: list[Detection], secondary: list[Detection], iou_thr: float = 0.45) -> list[Detection]:
    merged = list(primary)
    for sd in secondary:
        if all(iou(sd, pd) < iou_thr for pd in merged):
            merged.append(sd)
    return merged


def annotate_frame(
    frame: np.ndarray,
    tracks: dict,
    funnel: FunnelTracker,
    zone_mgr: ZoneManager,
    fps: float,
    status_map: Optional[dict[int, str]] = None,
) -> np.ndarray:
    out = zone_mgr.draw(frame)

    for tid, t in tracks.items():
        col = color_for(tid)
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)

        # Draw foot-point trail
        if len(t.history) >= 2:
            pts = np.array(t.history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=False, color=col, thickness=2)
        cv2.circle(out, t.foot_point, 4, col, -1)

        # ID + funnel stage label
        stage = funnel.states.get(tid)
        stage_str = stage.stage.value.upper() if stage else "?"
        status = (status_map or {}).get(tid, stage_str)
        label = f"#{tid} {status}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # HUD
    summary = funnel.summary()
    hud_lines = [
        f"FPS: {fps:5.1f}",
        f"Passed By: {summary['passed_by']}",
        f"Approached: {summary['approached']}  ({summary['approach_rate']*100:5.1f}%)",
        f"Purchased: {summary['purchased']} ({summary['purchase_rate']*100:5.1f}%)",
        f"Overall:   {summary['overall_conversion']*100:5.1f}%",
    ]
    pad = 8
    line_h = 22
    box_w = 320
    box_h = pad * 2 + line_h * len(hud_lines)
    cv2.rectangle(out, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    cv2.rectangle(out, (10, 10), (10 + box_w, 10 + box_h), (255, 255, 255), 1)
    for i, line in enumerate(hud_lines):
        cv2.putText(
            out, line, (10 + pad, 10 + pad + line_h * (i + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vending machine telemetry engine.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--source", default=None,
                        help="Override config source (int, path, or RTSP URL).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (handy for tests).")
    parser.add_argument("--show", action="store_true", help="Show live preview window.")
    parser.add_argument("--dashboard", action="store_true", help="Serve dashboard with live metrics/video.")
    parser.add_argument("--dashboard-host", default="0.0.0.0", help="Dashboard bind host.")
    parser.add_argument("--dashboard-port", type=int, default=8000, help="Dashboard port.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.source is not None:
        # CLI override; coerce numeric strings to int for webcam indices
        try:
            cfg["source"] = int(args.source)
        except ValueError:
            cfg["source"] = args.source
    if args.show:
        cfg.setdefault("output", {})["show_window"] = True

    # ── Open source first so we know the frame size for normalized zones ──
    cap = open_source(cfg["source"])
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read first frame from source.")
    frame_h, frame_w = first_frame.shape[:2]

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if src_fps <= 1 or src_fps > 240:
        src_fps = 30.0  # webcams sometimes report nonsense

    # ── Build subsystems ──────────────────────────────────────────────────
    zone_mgr = ZoneManager.from_config(cfg, frame_shape=(frame_h, frame_w))

    det_cfg = cfg.get("detector", {})
    shape_cfg = det_cfg.get("shape_filter", {})
    detector = PersonDetector(
        weights=det_cfg.get("weights", "yolov8n.pt"),
        confidence=det_cfg.get("confidence", 0.4),
        iou=det_cfg.get("iou", 0.5),
        device=det_cfg.get("device"),
        imgsz=det_cfg.get("imgsz", 640),
    )

    trk_cfg = cfg.get("tracker", {})
    tracker = CentroidTracker(
        max_disappeared=trk_cfg.get("max_disappeared", 30),
        max_distance=trk_cfg.get("max_distance", 120),
        appearance_weight=trk_cfg.get("appearance_weight", 0.45),
    )
    machine_pt = transaction_target_point(zone_mgr, (frame_h, frame_w))
    approach_cfg = cfg.get("approach", {})

    gaze_cfg_raw = cfg.get("gaze", {})
    gaze_detector: Optional[GazeDetector] = None
    if gaze_cfg_raw.get("enabled", True):
        gaze_detector = GazeDetector(
            GazeConfig(
                enabled=True,
                head_region_frac=gaze_cfg_raw.get("head_region_frac", 0.35),
                min_face_size_frac=gaze_cfg_raw.get("min_face_size_frac", 0.08),
                scale_factor=gaze_cfg_raw.get("scale_factor", 1.10),
                min_neighbors=gaze_cfg_raw.get("min_neighbors", 2),
                min_turn_offset_frac=gaze_cfg_raw.get("min_turn_offset_frac", 0.04),
            )
        )

    # ── Set up output directory + writers ─────────────────────────────────
    out_root = Path(cfg.get("output_dir", "runs"))
    run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    out_cfg = cfg.get("output", {})
    write_video = out_cfg.get("write_annotated_video", True)
    write_events = out_cfg.get("write_event_log", True)
    write_summary = out_cfg.get("write_summary", True)
    show_window = out_cfg.get("show_window", False)

    video_writer: Optional[cv2.VideoWriter] = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(run_dir / "annotated.mp4"),
            fourcc,
            src_fps,
            (frame_w, frame_h),
        )

    event_file = open(run_dir / "events.jsonl", "w") if write_events else None
    dashboard: Optional[Dashboard] = None

    def event_sink(ev: dict) -> None:
        if event_file is not None:
            event_file.write(json.dumps(ev) + "\n")
            event_file.flush()
        if dashboard is not None:
            dashboard.publish_event(ev)

    funnel_cfg = cfg.get("funnel", {})
    funnel = FunnelTracker(
        zone_manager=zone_mgr,
        engagement_dwell_sec=funnel_cfg.get("engagement_dwell_sec", 2.0),
        transaction_dwell_sec=funnel_cfg.get("transaction_dwell_sec", 6.0),
        visitor_min_confidence=funnel_cfg.get("visitor_min_confidence", 0.45),
        visitor_min_motion_px=funnel_cfg.get("visitor_min_motion_px", 10.0),
        visitor_min_seen_frames=funnel_cfg.get("visitor_min_seen_frames", 6),
        visitor_min_foot_traffic_frames=funnel_cfg.get("visitor_min_foot_traffic_frames", 4),
        visitor_min_displacement_px=funnel_cfg.get("visitor_min_displacement_px", 28.0),
        visitor_min_path_px=funnel_cfg.get("visitor_min_path_px", 42.0),
        event_sink=event_sink,
    )

    priv_cfg_raw = cfg.get("privacy", {})
    mode_raw = priv_cfg_raw.get("mode", "silhouette")
    # YAML 1.1 treats bare "off" as boolean false, so normalize that case.
    if isinstance(mode_raw, bool):
        mode = "off" if mode_raw is False else "silhouette"
    else:
        mode = str(mode_raw)
    privacy_cfg = PrivacyConfig(
        mode=mode,
        blur_kernel=priv_cfg_raw.get("blur_kernel", 51),
        pixelate_blocks=priv_cfg_raw.get("pixelate_blocks", 12),
    )

    if args.dashboard:
        dashboard = Dashboard(
            funnel=funnel,
            machine_api_token=os.getenv("MACHINE_API_TOKEN"),
        )
        dashboard.set_click_handler(lambda source: funnel.note_click(source))
        dashboard.start(host=args.dashboard_host, port=args.dashboard_port)
        print(f"[telemetry] dashboard live at http://{args.dashboard_host}:{args.dashboard_port}/")

    # ── Graceful shutdown on Ctrl-C ───────────────────────────────────────
    stop = {"flag": False}

    def _handler(signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    # ── Main loop ─────────────────────────────────────────────────────────
    print(f"[telemetry] writing run to {run_dir}")
    print(f"[telemetry] source={cfg['source']!r}  resolution={frame_w}x{frame_h}  fps_src={src_fps:.1f}")

    frame = first_frame
    frame_count = 0
    t0 = time.time()
    fps_smoothed = 0.0

    try:
        while True:
            tick = time.time()

            detections = detector.detect(frame)
            if det_cfg.get("enable_upper_body_assist", True):
                detections = merge_person_detections(
                    detections,
                    detect_upper_bodies(frame),
                    iou_thr=det_cfg.get("upper_body_merge_iou", 0.45),
                )
            detections = [
                d for d in detections
                if is_human_shape(
                    d,
                    frame_shape=frame.shape[:2],
                    min_aspect_ratio=shape_cfg.get("min_aspect_ratio", 0.75),
                    max_aspect_ratio=shape_cfg.get("max_aspect_ratio", 4.8),
                    min_area_frac=shape_cfg.get("min_area_frac", 0.003),
                    max_area_frac=shape_cfg.get("max_area_frac", 0.55),
                )
            ]
            for d in detections:
                d.appearance_hist = clothing_histogram(frame, d)
            # Privacy-protect footage before any writer/dashboard sees it.
            safe_frame = redact(frame, detections, privacy_cfg)
            tracks = tracker.update(detections, frame_shape=frame.shape[:2])
            approaching_ids = compute_approaching_ids(
                tracks,
                machine_pt,
                min_move_px=approach_cfg.get("min_move_px", 8.0),
                min_cosine=approach_cfg.get("min_cosine", 0.55),
            )
            funnel.update(tracks)

            looking_ids: set[int] = set()
            facing_camera_ids: set[int] = set()
            if gaze_detector is not None:
                for tid, t in tracks.items():
                    det = Detection(
                        x1=t.bbox[0], y1=t.bbox[1], x2=t.bbox[2], y2=t.bbox[3],
                        confidence=t.confidence,
                    )
                    facing_camera = gaze_detector.is_looking(frame, det)
                    if facing_camera:
                        facing_camera_ids.add(tid)
                    # Product rule:
                    # LOOK = head toward camera, while NOT in active approach motion.
                    looking = facing_camera and (tid not in approaching_ids)
                    funnel.note_glance(tid, looking)
                    if looking:
                        looking_ids.add(tid)

            for tid in approaching_ids:
                # Approach rule:
                # BODY moving toward camera/machine target AND facing camera.
                if tid in facing_camera_ids:
                    funnel.note_approach_motion(tid)

            status_map: dict[int, str] = {}
            for tid in tracks.keys():
                if tid in approaching_ids:
                    status_map[tid] = "APPROACHING"
                elif tid in looking_ids:
                    status_map[tid] = "LOOKING"
                else:
                    status_map[tid] = "PASSING_BY"

            annotated = annotate_frame(
                safe_frame, tracks, funnel, zone_mgr, fps_smoothed, status_map=status_map
            )

            if video_writer is not None:
                video_writer.write(annotated)
            if dashboard is not None:
                dashboard.publish_frame(annotated)

            if show_window:
                cv2.imshow("telemetry", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            tock = time.time()
            inst_fps = 1.0 / max(tock - tick, 1e-6)
            # Exponential smoothing so the HUD doesn't jitter
            fps_smoothed = inst_fps if fps_smoothed == 0 else 0.9 * fps_smoothed + 0.1 * inst_fps

            if args.max_frames is not None and frame_count >= args.max_frames:
                break
            if stop["flag"]:
                break

            ok, frame = cap.read()
            if not ok:
                break  # end of file or stream dropped

    finally:
        # ── Teardown ──────────────────────────────────────────────────────
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if show_window:
            cv2.destroyAllWindows()
        if event_file is not None:
            event_file.close()
        if dashboard is not None:
            dashboard.stop()

        # Per-track CSV
        rows = funnel.per_track_rows()
        with open(run_dir / "funnel.csv", "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            else:
                f.write("track_id\n")  # empty header so the file isn't blank

        if write_summary:
            summary = funnel.summary()
            summary["frames_processed"] = frame_count
            summary["wallclock_sec"] = round(time.time() - t0, 2)
            summary["avg_fps"] = round(frame_count / max(time.time() - t0, 1e-6), 2)
            summary["source"] = cfg["source"]
            summary["resolution"] = [frame_w, frame_h]
            with open(run_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        # Print to console so you get something useful even without opening files
        print("\n=== Run complete ===")
        print(f"frames:      {frame_count}")
        print(f"wallclock:   {time.time() - t0:.2f}s")
        print(f"avg fps:     {frame_count / max(time.time() - t0, 1e-6):.2f}")
        print(f"\nfunnel:")
        for k, v in funnel.summary().items():
            print(f"  {k:32s} {v}")
        print(f"\noutputs in: {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
