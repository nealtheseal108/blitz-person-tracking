"""
Depth-aware person detector.

The privacy contract:

    1. RGB enters this module.
    2. YOLOv8 produces person bboxes from RGB.
    3. We sample the aligned depth map at each bbox foot-point to get
       true distance in meters.
    4. We immediately call frame.release_rgb() -- the RGB array is
       overwritten with a 1x1 black pixel so any later code paths that
       still hold a reference cannot leak it.
    5. Only depth + Detection objects (boxes + distance) leave this
       module.

That means the rest of the pipeline (tracker, funnel, video writer)
never receives an identifiable image. The annotated video downstream
is built from the colorized depth map, not RGB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from depth_source import DepthFrame, sample_depth
from detector import Detection, PersonDetector


@dataclass
class DepthDetection(Detection):
    """A Detection enriched with true-meters distance from the camera."""
    distance_m: float = float("nan")     # median depth at foot-point, meters
    head_distance_m: float = float("nan")  # median depth at head, meters
    face_visible: bool = False           # set by gaze detector if/when wired

    def is_valid_distance(self) -> bool:
        return np.isfinite(self.distance_m) and self.distance_m > 0


class DepthPersonDetector:
    """Wraps PersonDetector, adds per-detection distance from a depth map."""

    def __init__(
        self,
        detector: Optional[PersonDetector] = None,
        weights: str = "yolov8n.pt",
        confidence: float = 0.4,
        sample_window: int = 7,
        max_distance_m: float = 8.0,
        min_distance_m: float = 0.2,
        device: Optional[str] = None,
    ):
        """
        Args:
            sample_window: side length of the kxk window we median-sample
                in the depth map. 7px is a good balance for 480p depth.
            max_distance_m / min_distance_m: detections whose foot-point
                depth falls outside this range are dropped. Filters
                spurious YOLO hits in posters, mirrors, and reflections.
        """
        self.detector = detector or PersonDetector(
            weights=weights, confidence=confidence, device=device
        )
        self.sample_window = sample_window
        self.max_distance_m = max_distance_m
        self.min_distance_m = min_distance_m

    def detect(self, frame: DepthFrame, release_rgb: bool = True) -> List[DepthDetection]:
        """Detect persons + sample distance.

        Args:
            release_rgb: if True (default), the RGB array on the frame
                is overwritten with a black pixel before this returns,
                so the rest of the pipeline cannot use it. Pass False
                if you want to run gaze detection on RGB right after
                this call -- in that case YOU are responsible for
                calling frame.release_rgb() before any writer touches
                the frame.
        """
        # 1. Detect persons in RGB
        rgb_detections = self.detector.detect(frame.rgb)

        # 2. Enrich with depth samples
        depth_dets: List[DepthDetection] = []
        for det in rgb_detections:
            fx, fy = det.foot_point
            head_x = (det.x1 + det.x2) // 2
            head_y = det.y1 + max(1, int((det.y2 - det.y1) * 0.10))

            d_foot = sample_depth(frame.depth_meters, fx, fy, k=self.sample_window)
            d_head = sample_depth(frame.depth_meters, head_x, head_y, k=self.sample_window)

            # Filter implausible distances (mirrors, posters of people, etc.)
            if not (np.isfinite(d_foot) and self.min_distance_m <= d_foot <= self.max_distance_m):
                continue

            depth_dets.append(
                DepthDetection(
                    x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                    confidence=det.confidence,
                    distance_m=float(d_foot),
                    head_distance_m=float(d_head) if np.isfinite(d_head) else float("nan"),
                )
            )

        if release_rgb:
            # PRIVACY: drop the RGB array. Anything downstream that tries
            # to read frame.rgb gets a 1x1 black pixel.
            frame.release_rgb()

        return depth_dets
