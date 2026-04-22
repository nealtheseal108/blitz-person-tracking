"""
Interactive zone calibration.

    python calibrate.py --source 0 --config config.yaml

Workflow:
    1. Grabs one frame from the source.
    2. For each of the three zones (foot_traffic, engagement, transaction)
       you left-click vertices on the image; press ENTER to close the
       polygon and move to the next zone, or BACKSPACE to undo the last
       point. Press 'r' to reset the current zone.
    3. Polygons are saved to config.yaml as normalized 0-1 coordinates,
       so they survive camera resolution changes.

Tip: set the foot_traffic polygon to cover the whole approach corridor,
engagement to ~3ft in front of the machine, and transaction to the
touchscreen / payment area. They're allowed (in fact, expected) to
overlap and nest.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy

import cv2
import numpy as np
import yaml

ZONES_TO_DRAW = [
    ("foot_traffic", (255, 200, 0)),   # cyan-ish
    ("engagement",   (0, 200, 255)),   # orange
    ("transaction",  (0, 255, 0)),     # green
]


def grab_frame(source) -> np.ndarray:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source!r}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Failed to read a frame from source.")
    return frame


def draw_zone(frame: np.ndarray, name: str, color: tuple) -> np.ndarray:
    """Returns the polygon in pixel coords as an (N,2) int32 array."""
    pts: list[list[int]] = []
    base = frame.copy()

    def render():
        canvas = base.copy()
        instructions = [
            f"Drawing zone: {name.upper()}",
            "L-click: add point   |   ENTER: finish   |   BACKSPACE: undo   |   r: reset",
        ]
        for i, line in enumerate(instructions):
            cv2.putText(canvas, line, (12, 28 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (12, 28 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)

        if pts:
            arr = np.array(pts, dtype=np.int32)
            if len(pts) >= 3:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [arr], color)
                canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
            cv2.polylines(canvas, [arr], isClosed=False, color=color, thickness=2)
            for p in pts:
                cv2.circle(canvas, tuple(p), 5, color, -1)
        return canvas

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])

    win = f"calibrate :: {name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, render())
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):  # ENTER
            if len(pts) < 3:
                print(f"  need at least 3 points; got {len(pts)}")
                continue
            break
        if key == 8:  # BACKSPACE
            if pts:
                pts.pop()
        if key in (ord("r"), ord("R")):
            pts.clear()
        if key == 27:  # ESC
            cv2.destroyWindow(win)
            raise SystemExit("Calibration aborted.")

    cv2.destroyWindow(win)
    return np.array(pts, dtype=np.int32)


def normalize(polygon: np.ndarray, w: int, h: int) -> list:
    return [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in polygon]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="Webcam index, video path, or RTSP URL.")
    parser.add_argument("--config", default="config.yaml",
                        help="Config file to update.")
    args = parser.parse_args()

    # Coerce numeric source to int (webcam index)
    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    print(f"[calibrate] grabbing frame from {src!r} ...")
    frame = grab_frame(src)
    h, w = frame.shape[:2]
    print(f"[calibrate] frame: {w}x{h}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    new_zones = []
    for name, color in ZONES_TO_DRAW:
        print(f"[calibrate] draw zone: {name}")
        poly = draw_zone(frame, name, color)
        new_zones.append({
            "name": name,
            "color": list(color),
            "polygon": normalize(poly, w, h),
        })

    cfg = deepcopy(cfg)
    cfg["zones"] = new_zones

    with open(args.config, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[calibrate] wrote {len(new_zones)} zones to {args.config}")


if __name__ == "__main__":
    sys.exit(main())
