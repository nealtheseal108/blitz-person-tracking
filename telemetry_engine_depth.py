"""
Blitz Machine Telemetry Engine.

Supports two modes:
  Webcam    --source 0             (laptop / IP cam, distance estimated from bbox height)
  Depth     --backend realsense    (Intel RealSense D4xx or Luxonis OAK-D)

Funnel (new order):
  Passed By  ->  Approached  ->  Looked At  ->  Clicked  ->  Purchased

Webcam mode uses THREE separate detection layers:
  1. Passed By   — YOLO body detection (yolov8n.pt) tracks full-body presence at
                   distance. High confidence threshold keeps noise low. Any body
                   detected for ≥4 frames = one unique visitor.
  2. Approached  — Body bbox height grows over a sliding window = person closing
                   distance. No face required. This is the PREREQUISITE for Looked At.
  3. Looked At   — MediaPipe FaceLandmarker detects faces on the full frame at any
                   distance. Face matched to a body track. Only counted if that body
                   track has already been marked as approaching.

Privacy:
  In depth mode, RGB is overwritten with a 1x1 black pixel after gaze detection
  runs. The annotated output video is rendered from a colorized depth map -- it
  literally cannot show a face. In webcam mode, silhouette redaction is applied
  before any frame is written.

Usage:
  # Webcam (no hardware needed):
  python telemetry_engine_depth.py --source 0 --show

  # RealSense:
  python telemetry_engine_depth.py --backend realsense --show

  # OAK-D:
  python telemetry_engine_depth.py --backend oakd --show
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

from dashboard_server import Dashboard
from depth_source import colorize_depth, open_depth_source, sample_depth
from depth_detector import DepthPersonDetector
from detector import Detection, PersonDetector
from face_detector import FaceDetector
from funnel import FunnelTracker
from gaze import GazeConfig, GazeDetector
from privacy import PrivacyConfig, redact_inplace
from tracker import CentroidTracker, Track
from zones import ZoneManager


# ── Constants ───────────────────────────────────────────────────────────────

# Approach detection (webcam/face mode): face must be visible AND distance must
# drop by this many metres over a DIST_WINDOW_FRAMES sliding window.
APPROACH_DIST_DROP_M  = 0.08     # must close 8 cm+ over the window
DIST_WINDOW_FRAMES    = 12       # ~1 second at 12 FPS

# Approach detection (depth mode legacy — vertical motion heuristic)
APPROACH_DY_THRESHOLD = 5        # pixels of downward centroid shift per frame

# Stage label + color (BGR) map
STAGE_STYLE: dict[str, Tuple[str, Tuple[int, int, int]]] = {
    "approaching": ("APPROACHING", (0,  165, 255)),
    "looking":     ("LOOKING",     (80, 200, 255)),
    "engaged":     ("AT MACHINE",  (0,  200, 255)),
    "converted":   ("PURCHASED",   (0,  100, 255)),
    "visitor":     ("PASSING",     (80, 210,  80)),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def is_distance_decreasing(dist_history: list, drop_m: float = APPROACH_DIST_DROP_M) -> bool:
    """Return True if distance has dropped by at least drop_m over the history window.

    Compares the average of the oldest third vs the newest third of the window
    so single noisy frames don't trigger a false positive.
    """
    valid = [d for d in dist_history if d == d and d > 0]  # drop NaN / zeros
    if len(valid) < 6:
        return False
    third = max(1, len(valid) // 3)
    older_avg = sum(valid[:third]) / third
    newer_avg = sum(valid[-third:]) / third
    return (older_avg - newer_avg) >= drop_m


def estimate_distances_webcam(
    tracks: Dict[int, Track],
    frame_height: int,
    scale: float = 2.0,
) -> Dict[int, float]:
    """Estimate distance from camera using bbox height.

    distance_m ≈ scale * frame_height / bbox_height_px

    With scale=2.0 and a 480p frame:
      bbox_height 240px  ->  4.0m
      bbox_height 120px  ->  8.0m
      bbox_height 320px  ->  3.0m
    Tune scale with --webcam-scale if your camera is at a different height.
    """
    result: Dict[int, float] = {}
    for tid, t in tracks.items():
        bh = t.bbox[3] - t.bbox[1]
        result[tid] = scale * frame_height / bh if bh > 0 else float("nan")
    return result


def compute_distances_depth(
    tracks: Dict[int, Track],
    depth_m: np.ndarray,
    sample_window: int = 7,
) -> Dict[int, float]:
    """Sample the depth map at each track's foot-point."""
    result: Dict[int, float] = {}
    for tid, t in tracks.items():
        x, y = t.foot_point
        result[tid] = sample_depth(depth_m, x, y, k=sample_window)
    return result


# ── Shared helpers ───────────────────────────────────────────────────────────

def det_from_track(track: Track) -> Detection:
    """Synthesise a minimal Detection from a Track's last bbox (for gaze/depth)."""
    x1, y1, x2, y2 = track.bbox
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=1.0)


def motion_dy(track: Track) -> float:
    """Vertical centroid delta (pixels) over the last two frames.

    Positive = centroid moved downward (person walking away from a top-mounted
    camera in depth mode).  Used as a lightweight approach heuristic.
    """
    if len(track.centroids) < 2:
        return 0.0
    return float(track.centroids[-1][1] - track.centroids[-2][1])


# ── Depth-mode body-box annotation ───────────────────────────────────────────

def draw_annotations(
    frame: np.ndarray,
    tracks: Dict[int, Track],
    funnel: FunnelTracker,
    distances: Dict[int, float],
    face_boxes: Optional[Dict[int, list]] = None,
    approaching_ids: Optional[set] = None,
) -> np.ndarray:
    """Draw per-person boxes for depth mode (YOLO body tracks)."""
    out = frame.copy()
    approaching_ids = approaching_ids or set()

    for tid, track in tracks.items():
        state = funnel.states.get(tid)
        if tid in approaching_ids:
            style_key = "approaching"
        elif state and getattr(state, "_looking_now", False):
            style_key = "looking"
        else:
            style_key = "visitor"

        label_text, color = STAGE_STYLE.get(style_key, ("PASSING", (80, 210, 80)))

        dist = distances.get(tid)
        dist_str = f" {dist:.1f}m" if dist is not None and np.isfinite(dist) else ""
        full_label = f"#{tid} {label_text}{dist_str}"

        x1, y1, x2, y2 = track.bbox
        thickness = 3 if style_key in ("looking", "approaching") else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        lx, ly = x1, max(y1 - 4, th + 6)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 8, ly + 2), color, -1)
        cv2.putText(out, full_label, (lx + 4, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    return out


# ── Face-mode helpers ─────────────────────────────────────────────────────────

_PROFILE_STATE = {"orientation": "profile", "yaw_deg": 0.0, "is_looking": False}
_MATCH_RADIUS_PX = 180   # max centroid distance to accept a face→track match


def _update_face_states(
    tracks: Dict[int, Track],
    faces: list,
    face_states: Dict[int, dict],
) -> None:
    """Match each active track to its nearest face detection and cache the state.

    When no face is found close enough to a track (MediaPipe loses a profile
    or side-on face), the track's state is *reset* to a neutral PASSING
    profile rather than keeping the last known frontal state.  This ensures
    the box color switches from LOOKING→PASSING as soon as the face turns
    away, instead of freezing on yellow.
    """
    for tid, track in tracks.items():
        tx = (track.bbox[0] + track.bbox[2]) // 2
        ty = (track.bbox[1] + track.bbox[3]) // 2

        if not faces:
            face_states[tid] = _PROFILE_STATE
            continue

        best = min(
            faces,
            key=lambda f: (f.centroid[0] - tx) ** 2 + (f.centroid[1] - ty) ** 2,
        )
        dist_sq = (best.centroid[0] - tx) ** 2 + (best.centroid[1] - ty) ** 2
        if dist_sq <= _MATCH_RADIUS_PX ** 2:
            face_states[tid] = {
                "orientation": best.orientation,
                "yaw_deg":     best.yaw_deg,
                "is_looking":  best.is_looking,
            }
        else:
            # Track is alive but no close face match → treat as passing
            face_states[tid] = _PROFILE_STATE


# ── Face-mode annotation ─────────────────────────────────────────────────────

# Orientation → (label, BGR color)
FACE_STYLE = {
    "approaching": ("APPROACHING", (0,   165, 255)),   # orange
    "looking":     ("LOOKING",     (0,   230, 255)),   # cyan
    "profile":     ("PASSING",     (80,  210,  80)),   # green
    "visitor":     ("PASSING",     (80,  210,  80)),
    "engaged":     ("AT MACHINE",  (0,   200, 255)),
    "converted":   ("PURCHASED",   (0,   100, 255)),
}

def draw_face_annotations(
    frame: np.ndarray,
    tracks: Dict[int, Track],
    funnel: FunnelTracker,
    distances: Dict[int, float],
    face_states: Dict[int, dict],      # tid -> {orientation, yaw_deg, is_looking}
    approaching_ids: Optional[set] = None,
) -> np.ndarray:
    """Draw one tight box per tracked face, colored by orientation/state."""
    out = frame.copy()
    approaching_ids = approaching_ids or set()

    for tid, track in tracks.items():
        state = funnel.states.get(tid)
        fs = face_states.get(tid, {})
        orientation = fs.get("orientation", "profile")
        is_looking = fs.get("is_looking", False)
        yaw = fs.get("yaw_deg", 0.0)

        if tid in approaching_ids:
            style_key = "approaching"
        elif is_looking or (state and state._looking_now):
            style_key = "looking"
        else:
            style_key = orientation   # "profile" → PASSING

        label_text, color = FACE_STYLE.get(style_key, ("PASSING", (80, 210, 80)))

        dist = distances.get(tid)
        dist_str = f" {dist:.1f}m" if dist is not None and np.isfinite(dist) else ""
        yaw_str = f" {yaw:+.0f}°" if fs else ""
        full_label = f"#{tid} {label_text}{dist_str}{yaw_str}"

        x1, y1, x2, y2 = track.bbox
        # Thicker border for LOOKING/APPROACHING
        thickness = 3 if style_key in ("looking", "approaching") else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # Label pill above the box
        (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        lx = x1
        ly = max(y1 - 4, th + 6)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 8, ly + 2), color, -1)
        cv2.putText(out, full_label, (lx + 4, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def draw_hud(frame: np.ndarray, summary: dict, fps: float) -> np.ndarray:
    """Bottom-left HUD with the five funnel metrics."""
    out = frame.copy()
    h = out.shape[0]
    lines = [
        f"FPS: {fps:.1f}",
        f"Passed By:  {summary.get('passed_by', 0)}",
        f"Looked At:  {summary.get('looked_at', 0)}",
        f"Approached: {summary.get('approached', 0)}",
        f"Clicked:    {summary.get('clicked', 0)}",
        f"Purchased:  {summary.get('purchased', 0)}",
    ]
    y = h - 10 - 20 * len(lines)
    for line in lines:
        # Drop shadow
        cv2.putText(out, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 240, 255), 1, cv2.LINE_AA)
        y += 20
    return out


# ── Three-layer webcam helpers ───────────────────────────────────────────────


def _match_face_to_body_tracks(
    faces: list,
    tracks: Dict[int, Track],
    slack_px: int = 100,
) -> Dict[int, object]:
    """Assign each face detection to its best body track.

    Pass 1 — bbox containment:
      Face centroid must fall inside the full body bbox ± slack_px.
      No head-region restriction: YOLO sometimes clips the head from the
      top of the box, so limiting to the top fraction misses those cases.

    Pass 2 — nearest-centroid fallback:
      Any face still unmatched after pass 1 is assigned to the body track
      whose centroid is closest, as long as the distance is < half the
      frame height.  This catches the "face detected but slightly outside
      the bbox" case that happens at distance / unusual camera angles.

    When multiple faces compete for the same track, the one closest to the
    track's horizontal centre wins.  Unmatched tracks get None.
    """
    if not faces or not tracks:
        return {tid: None for tid in tracks}

    result: dict = {tid: None for tid in tracks}
    unmatched_faces = []

    # ── Pass 1: bbox containment ─────────────────────────────────────────────
    for face in faces:
        fx, fy = face.centroid
        best_tid = None
        best_dx = float("inf")

        for tid, track in tracks.items():
            x1, y1, x2, y2 = track.bbox
            in_x = (x1 - slack_px) <= fx <= (x2 + slack_px)
            in_y = (y1 - slack_px) <= fy <= (y2 + slack_px)
            if not (in_x and in_y):
                continue
            tx = (x1 + x2) // 2
            dx = abs(fx - tx)
            if dx < best_dx:
                best_dx = dx
                best_tid = tid

        if best_tid is not None:
            cur = result[best_tid]
            if cur is None:
                result[best_tid] = face
            else:
                track = tracks[best_tid]
                tx = (track.bbox[0] + track.bbox[2]) // 2
                if abs(fx - tx) < abs(cur.centroid[0] - tx):
                    result[best_tid] = face
        else:
            unmatched_faces.append(face)

    # ── Pass 2: nearest-centroid fallback ────────────────────────────────────
    # Estimate a reasonable search radius from any track's bbox height.
    if unmatched_faces and tracks:
        sample_h = max(t.bbox[3] - t.bbox[1] for t in tracks.values())
        max_dist_sq = (max(sample_h, 100) * 2) ** 2   # 2× body height

        for face in unmatched_faces:
            fx, fy = face.centroid
            best_tid = None
            best_d_sq = max_dist_sq

            for tid, track in tracks.items():
                tx = (track.bbox[0] + track.bbox[2]) // 2
                ty = (track.bbox[1] + track.bbox[3]) // 2
                d_sq = (fx - tx) ** 2 + (fy - ty) ** 2
                if d_sq < best_d_sq:
                    best_d_sq = d_sq
                    best_tid = tid

            if best_tid is not None and result[best_tid] is None:
                result[best_tid] = face

    return result


def _draw_three_layer_annotations(
    frame: np.ndarray,
    tracks: Dict[int, Track],
    funnel: FunnelTracker,
    distances: Dict[int, float],
    looking_tids: set,                  # track IDs with a Haar frontal face
    approaching_ids: set,
    face_boxes: Dict[int, tuple],       # tid -> (x, y, w, h) Haar bbox
) -> np.ndarray:
    """Body bbox colored by funnel stage, thin Haar face overlay when detected."""
    out = frame.copy()

    for tid, track in tracks.items():
        state      = funnel.states.get(tid)
        is_looking = tid in looking_tids

        if is_looking:
            style_key = "looking"
        elif tid in approaching_ids:
            style_key = "approaching"
        else:
            style_key = "visitor"

        label_text, color = STAGE_STYLE.get(style_key, ("PASSING", (80, 210, 80)))

        dist = distances.get(tid)
        dist_str = f" {dist:.1f}m" if dist is not None and np.isfinite(dist) else ""

        # Counted-stage flags: P=passed_by  A=approached  L=looked_at
        p_flag = "P" if (state and state.counted_visitor)  else "-"
        a_flag = "A" if tid in funnel.approached_ids        else "-"
        l_flag = "L" if (state and state._looker_counted)  else "-"
        flags  = f" {p_flag}{a_flag}{l_flag}"

        full_label = f"#{tid} {label_text}{dist_str}{flags}"

        x1, y1, x2, y2 = track.bbox
        thickness = 3 if style_key in ("looking", "approaching") else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        lx = x1
        ly = max(y1 - 4, th + 6)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 8, ly + 2), color, -1)
        cv2.putText(out, full_label, (lx + 4, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

        # No separate face box — one box per person, color reflects state.

    return out


# ── Webcam mode ──────────────────────────────────────────────────────────────

def run_webcam_mode(
    args: argparse.Namespace,
    cfg: dict,
    funnel: FunnelTracker,
    dash: Dashboard,
) -> None:
    """Three-layer webcam pipeline.

    Layer 1 — Passed By (YOLO body):
      YOLOv8n detects full/partial body silhouettes at 5 m+. High confidence
      threshold (≥0.45) keeps false positives low. CentroidTracker assigns
      stable IDs. Any track visible ≥4 frames = one unique Passed By.

    Layer 2 — Approached (body bbox growth):
      Distance estimated from body bbox height (distance_m ≈ webcam_scale × H / h).
      If distance has dropped ≥8 cm over a 12-frame window → Approached.
      No face required. Approach is the prerequisite for Looked At.

    Layer 3 — Looked At (MediaPipe FaceLandmarker):
      MediaPipe runs on the full frame at every tick. Face detections are matched
      to body tracks by checking whether the face centroid falls in the top 45% of
      the body bbox. Frontal face on an approaching track → Looked At counted.

    Funnel order: Passed By → Approached → Looked At → Clicked → Purchased
    """
    det_cfg = cfg.get("detector", {})

    # ── Layer 1+2: YOLO body detector ────────────────────────────────────────
    # Floor confidence at 0.45 — the old 0.35 default caused too much noise.
    body_confidence = max(float(det_cfg.get("confidence", 0.50)), 0.50)
    body_detector = PersonDetector(
        weights=det_cfg.get("weights", "yolov8n.pt"),
        confidence=body_confidence,
        iou=det_cfg.get("iou", 0.5),
        imgsz=det_cfg.get("imgsz", 160),
        device=det_cfg.get("device"),
    )

    # ── Layer 3: OpenCV Haar frontal face detector ────────────────────────────
    # Simpler and more sensitive than MediaPipe at distance.
    # Ships with OpenCV — no download needed.
    _haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(_haar_path)
    if face_cascade.empty():
        raise SystemExit(f"Haar cascade not found at {_haar_path}")

    tracker = CentroidTracker(
        max_disappeared=cfg.get("tracker", {}).get("max_disappeared", 1),
        max_distance=cfg.get("tracker", {}).get("max_distance", 200),
    )

    src: int | str = args.source
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {src!r}")

    prev_ts       = time.time()
    fps_est       = 30.0
    _dist_history: Dict[int, list] = {}   # tid -> sliding window of distances (m)
    _frame_idx    = 0

    print(f"[engine] three-layer webcam  source={src!r}  "
          f"body_conf={body_confidence:.2f}  "
          f"dashboard=http://{args.host}:{args.port}/")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[engine] end of stream")
                break

            h, _w = frame.shape[:2]
            frame_area = h * _w
            _frame_idx += 1

            # ── 1. Layer 1: detect bodies (YOLO every frame at imgsz=160) ────
            _raw = body_detector.detect(frame)
            body_dets = [
                d for d in _raw
                if (d.x2 - d.x1) >= 40
                and (d.y2 - d.y1) / max(d.x2 - d.x1, 1) <= 5.0
                and (d.x2 - d.x1) * (d.y2 - d.y1) <= 0.80 * frame_area
            ]

            # ── 2. Track body centroids ───────────────────────────────────────
            tracks = tracker.update(body_dets, frame_shape=frame.shape)

            # ── 3. Distance from body bbox height ────────────────────────────
            #   distance_m ≈ webcam_scale × frame_height / body_bbox_height_px
            distances: Dict[int, float] = {}
            for tid, t in tracks.items():
                body_h = t.bbox[3] - t.bbox[1]
                distances[tid] = (
                    args.webcam_scale * h / body_h if body_h > 0 else float("nan")
                )

            # ── 4. Sliding distance history ───────────────────────────────────
            for tid, d in distances.items():
                if tid not in _dist_history:
                    _dist_history[tid] = []
                if d == d and d > 0:                    # drop NaN/zero
                    _dist_history[tid].append(d)
                    if len(_dist_history[tid]) > DIST_WINDOW_FRAMES:
                        _dist_history[tid].pop(0)
            for tid in list(_dist_history):
                if tid not in tracks:
                    del _dist_history[tid]

            # ── 5. Funnel update (MUST precede note_* calls) ──────────────────
            funnel.update(tracks, distances_m=distances)

            # ── 6. Layer 2: approach detection ────────────────────────────────
            approaching_ids: set = set()
            for tid in tracks:
                if is_distance_decreasing(_dist_history.get(tid, [])):
                    approaching_ids.add(tid)
                    funnel.note_approach_motion(tid)

            # ── 7. Layer 3: Haar frontal face detection ───────────────────────
            # Downscale to half res for speed; scale coords back up after.
            HAAR_SCALE = 0.5
            small = cv2.resize(frame, (int(_w * HAAR_SCALE), int(h * HAAR_SCALE)))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(gray, gray)
            haar_faces_small = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(15, 15),
            )
            # Scale detected rects back to full-res coordinates
            if len(haar_faces_small) > 0:
                haar_faces = (haar_faces_small / HAAR_SCALE).astype(int)
            else:
                haar_faces = haar_faces_small
            # haar_faces is a list of (x, y, w, h) or empty tuple

            # Match each detected face to the nearest body track
            looking_tids: set = set()
            face_boxes: Dict[int, tuple] = {}   # tid -> (x,y,w,h) for drawing
            for face_rect in (haar_faces if len(haar_faces) > 0 else []):
                fx, fy, fw, fh = face_rect
                face_cx = fx + fw // 2
                face_cy = fy + fh // 2
                best_tid = None
                best_dist = float("inf")
                for tid, track in tracks.items():
                    x1, y1, x2, y2 = track.bbox
                    slack = 80
                    if not ((x1 - slack) <= face_cx <= (x2 + slack) and
                            (y1 - slack) <= face_cy <= (y2 + slack)):
                        continue
                    tx = (x1 + x2) // 2
                    d  = abs(face_cx - tx)
                    if d < best_dist:
                        best_dist = d
                        best_tid = tid
                if best_tid is not None:
                    looking_tids.add(best_tid)
                    face_boxes[best_tid] = face_rect

            for tid in tracks:
                funnel.note_glance(tid, tid in looking_tids)

            # ── 8. Sync counters + flags directly to box color ────────────────
            # Box color is the source of truth.
            # Yellow → L flag + Looked At increments immediately.
            # Orange → A flag + Approached increments immediately.
            _now = time.time()
            for tid in tracks:
                state = funnel.states.get(tid)
                if state is None:
                    continue
                is_yellow = tid in looking_tids
                is_orange = (not is_yellow) and (tid in approaching_ids)

                # Green must exist before yellow can count
                if is_yellow and state.counted_visitor and not state._looker_counted:
                    fp = state.last_foot_point or state.first_foot_point
                    if fp is None or not funnel._is_duplicate_looker(_now, fp):
                        funnel.unique_lookers += 1
                        if fp:
                            funnel._recent_looker_points.append((_now, fp))
                    state._looker_counted = True
                    state._looking_now = True

                # Yellow must exist before orange can count
                if (is_yellow or is_orange) and state._looker_counted and tid not in funnel.approached_ids:
                    funnel.approached_ids.add(tid)

            annotated = _draw_three_layer_annotations(
                frame, tracks, funnel, distances,
                looking_tids, approaching_ids, face_boxes,
            )
            annotated = draw_hud(annotated, funnel.summary(), fps_est)

            # ── 9. Publish ────────────────────────────────────────────────────
            dash.publish_frame(annotated)

            if args.show:
                cv2.imshow("Blitz Telemetry", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 / max(1e-3, now - prev_ts)
            prev_ts = now

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


# ── Depth mode ───────────────────────────────────────────────────────────────

def run_depth_mode(
    args: argparse.Namespace,
    cfg: dict,
    funnel: FunnelTracker,
    dash: Dashboard,
) -> None:
    """Main loop for Intel RealSense or Luxonis OAK-D depth camera."""
    depth_cfg = cfg.get("depth", {})
    gaze_cfg_data = cfg.get("gaze", {})

    depth_detector = DepthPersonDetector(
        weights=cfg.get("detector", {}).get("weights", "yolov8n.pt"),
        confidence=cfg.get("detector", {}).get("confidence", 0.4),
        sample_window=depth_cfg.get("sample_window_px", 7),
        max_distance_m=depth_cfg.get("max_distance_m", 8.0),
        min_distance_m=depth_cfg.get("min_distance_m", 0.2),
        device=cfg.get("detector", {}).get("device"),
    )
    gaze_cfg = GazeConfig(
        enabled=gaze_cfg_data.get("enabled", True),
        head_region_frac=gaze_cfg_data.get("head_region_frac", 0.35),
    )
    gaze = GazeDetector(cfg=gaze_cfg)
    tracker = CentroidTracker(
        max_disappeared=cfg.get("tracker", {}).get("max_disappeared", 30),
        max_distance=cfg.get("tracker", {}).get("max_distance", 120),
    )
    viz_max_m: float = depth_cfg.get("viz_max_distance_m", 6.0)
    sample_window: int = depth_cfg.get("sample_window_px", 7)

    backend = args.backend or depth_cfg.get("backend", "auto")
    src = open_depth_source(
        backend,
        width=depth_cfg.get("width", 848),
        height=depth_cfg.get("height", 480),
        fps=depth_cfg.get("fps", 30),
    )

    prev_ts = time.time()
    fps_est = 30.0

    print(f"[engine] depth backend={backend!r}  dashboard=http://{args.host}:{args.port}/")

    try:
        for depth_frame in src.frames():
            h, w = depth_frame.depth_meters.shape[:2]

            # 1. Detect (RGB still alive: release_rgb=False)
            depth_dets = depth_detector.detect(depth_frame, release_rgb=False)

            # 2. Track (DepthDetection is a Detection subclass; fully compatible)
            tracks = tracker.update(depth_dets, frame_shape=(h, w))

            # 3. Gaze -- must run BEFORE release_rgb
            for tid, track in tracks.items():
                synth = det_from_track(track)
                attn, looked = gaze.attention_and_look(depth_frame.rgb, synth)
                funnel.note_glance(tid, looked)
                funnel.note_attention(tid, attn)

                if motion_dy(track) > APPROACH_DY_THRESHOLD:
                    funnel.note_approach_motion(tid)

            # 4. PRIVACY: destroy RGB -- nothing downstream can access it
            depth_frame.release_rgb()

            # 5. Distances: sample depth map at each foot-point
            distances = compute_distances_depth(
                tracks, depth_frame.depth_meters, sample_window=sample_window
            )

            # 6. Funnel update
            funnel.update(tracks, distances_m=distances)

            # 7. Render from DEPTH (never RGB) + annotations
            viz = colorize_depth(depth_frame.depth_meters, max_m=viz_max_m)
            annotated = draw_annotations(viz, tracks, funnel, distances)
            annotated = draw_hud(annotated, funnel.summary(), fps_est)

            # 8. Publish
            dash.publish_frame(annotated)

            if args.show:
                cv2.imshow("Blitz Depth Telemetry", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 / max(1e-3, now - prev_ts)
            prev_ts = now

    finally:
        src.close()
        if args.show:
            cv2.destroyAllWindows()


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blitz vending machine telemetry engine"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--source", default="0",
        help="Webcam source: int index, video path, or RTSP URL (webcam mode only)",
    )
    parser.add_argument(
        "--backend", default=None,
        help="Depth backend: realsense | oakd | auto (triggers depth mode)",
    )
    parser.add_argument("--show", action="store_true", help="Show live OpenCV window")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard HTTP port")
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard bind host")
    parser.add_argument(
        "--webcam-scale", type=float, default=2.0,
        help="[depth/body mode] Distance scale: distance_m = scale * frame_h / body_bbox_h",
    )
    parser.add_argument(
        "--face-scale", type=float, default=0.19,
        help=(
            "[face-only webcam mode] Distance scale: distance_m = scale * frame_h / face_h. "
            "Tune this for your lens + mounting height. "
            "Default 0.19 ≈ waist-height webcam, ~23 cm face, 480p."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Zones are config-driven; frame_shape used only for normalizing polygon coords
    frame_shape = (
        cfg.get("depth", {}).get("height", 480),
        cfg.get("depth", {}).get("width", 848),
    )
    zones = ZoneManager.from_config(cfg, frame_shape=frame_shape)

    # Dashboard + funnel share a reference so the server can always read live data
    dash = Dashboard(funnel=None)
    funnel_cfg = cfg.get("funnel", {})
    funnel = FunnelTracker(
        zone_manager=zones,
        engagement_dwell_sec=funnel_cfg.get("engagement_dwell_sec", 2.0),
        transaction_dwell_sec=funnel_cfg.get("transaction_dwell_sec", 6.0),
        event_sink=dash.event_sink(),
    )
    dash.funnel = funnel
    dash.set_click_handler(lambda source: funnel.note_click(source=source))
    dash.start(host=args.host, port=args.port)

    try:
        if args.backend is not None:
            run_depth_mode(args, cfg, funnel, dash)
        else:
            run_webcam_mode(args, cfg, funnel, dash)
    except KeyboardInterrupt:
        print("\n[engine] stopped by user")
    finally:
        dash.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
