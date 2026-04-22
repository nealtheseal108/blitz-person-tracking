"""
Synthetic visual demo for the dashboard.

Run:
    python demo_synthetic.py --port 8000
Then open:
    http://localhost:8000/
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from dashboard_server import Dashboard
from funnel import FunnelTracker
from tracker import Track
from zones import DistanceBandZone, ZoneManager


def build_funnel() -> FunnelTracker:
    zones = ZoneManager(
        [
            DistanceBandZone("foot_traffic", min_m=0.6, max_m=5.0, color=(255, 200, 0)),
            DistanceBandZone("engagement", min_m=0.3, max_m=1.0, color=(0, 200, 255)),
            DistanceBandZone("transaction", min_m=0.0, max_m=0.4, color=(0, 255, 0)),
        ]
    )
    return FunnelTracker(zones, engagement_dwell_sec=1.4, transaction_dwell_sec=2.8)


def make_track(track_id: int, x: int, y: int) -> Track:
    w, h = 52, 92
    return Track(
        track_id=track_id,
        bbox=(x - w // 2, y - h, x + w // 2, y),
        centroid=(x, y - h // 2),
        foot_point=(x, y),
        confidence=0.99,
        history=[(x, y)],
    )


def render_scene(frame_idx: int, tracks: dict[int, Track], distances: dict[int, float], summary: dict) -> np.ndarray:
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    img[:] = (14, 18, 30)

    # Corridor and machine blocks
    cv2.rectangle(img, (0, 370), (960, 540), (28, 38, 62), -1)
    cv2.rectangle(img, (750, 180), (920, 500), (34, 58, 94), -1)
    cv2.putText(img, "Machine", (782, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 210, 255), 2, cv2.LINE_AA)

    for tid, t in tracks.items():
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 210, 140), 2)
        cv2.circle(img, t.foot_point, 4, (255, 255, 255), -1)
        cv2.putText(
            img,
            f"#{tid} {distances.get(tid, float('nan')):.2f}m",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 240, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(img, f"synthetic frame {frame_idx}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(img, f"passed:{summary['passed_by']} looked:{summary['looked_at']} approached:{summary['approached']}", (18, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"clicked:{summary['clicked']} purchased:{summary['purchased']}", (18, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 220, 255), 1, cv2.LINE_AA)
    return img


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    funnel = build_funnel()
    dash = Dashboard(funnel=funnel)
    dash.set_click_handler(lambda source: funnel.note_click(source))
    dash.start(host=args.host, port=args.port)
    print(f"[demo] dashboard at http://{args.host}:{args.port}/")

    fps = 12.0
    dt = 1.0 / fps
    frame_idx = 0
    start = time.time()

    while True:
        t = time.time() - start

        # Synthetic motion:
        # - track 0 passes by
        # - track 1 approaches and looks twice
        # - track 2 approaches and converts
        tracks: dict[int, Track] = {}
        distances: dict[int, float] = {}

        x0 = int(50 + (t * 80) % 950)
        y0 = 440
        tracks[0] = make_track(0, x0, y0)
        distances[0] = 3.8

        x1 = int(120 + (t * 45) % 620)
        y1 = 445
        tracks[1] = make_track(1, x1, y1)
        distances[1] = max(0.65, 2.4 - 0.02 * frame_idx)

        x2 = int(250 + (t * 38) % 520)
        y2 = 445
        tracks[2] = make_track(2, x2, y2)
        distances[2] = max(0.28, 2.2 - 0.022 * frame_idx)

        funnel.update(tracks, distances_m=distances)

        # synthetic gaze toggles
        funnel.note_glance(1, looking=(frame_idx % 30 in {3, 4, 5, 18, 19, 20}))
        funnel.note_glance(2, looking=(frame_idx % 25 in {2, 3, 4, 5, 12, 13}))

        # synthetic machine click + purchase attribution
        if frame_idx % 70 == 0 and frame_idx > 0:
            funnel.note_click("synthetic_machine")
        if frame_idx % 100 == 0 and frame_idx > 0:
            funnel.confirm_most_recent_engaged(source="synthetic_pos")

        frame = render_scene(frame_idx, tracks, distances, funnel.summary())
        dash.publish_frame(frame)

        frame_idx += 1
        time.sleep(dt)


if __name__ == "__main__":
    raise SystemExit(main())
