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

from detector import PersonDetector
from funnel import FunnelStage, FunnelTracker
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


def annotate_frame(
    frame: np.ndarray,
    tracks: dict,
    funnel: FunnelTracker,
    zone_mgr: ZoneManager,
    fps: float,
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
        label = f"#{tid} {stage_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # HUD
    summary = funnel.summary()
    hud_lines = [
        f"FPS: {fps:5.1f}",
        f"Visitors:  {summary['visitors']}",
        f"Engaged:   {summary['engaged']}  ({summary['engagement_rate']*100:5.1f}%)",
        f"Converted: {summary['converted']} ({summary['purchase_rate_given_engaged']*100:5.1f}%)",
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
        event_sink=event_sink,
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
            tracks = tracker.update(detections)
            funnel.update(tracks)

            annotated = annotate_frame(frame, tracks, funnel, zone_mgr, fps_smoothed)

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
