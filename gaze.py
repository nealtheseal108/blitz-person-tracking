"""
Lightweight "is this person looking at the machine?" detector.

Insight: the camera is mounted on TOP of the machine, pointing outward.
A frontal face detection inside a person's bounding box therefore
implies the person is facing the machine -- profiles and back-of-head
won't trigger a frontal-face detector. So we don't need full head-pose
estimation for a clean signal.

Implementation uses OpenCV's bundled Haar cascade for frontal faces.
That cascade ships with `opencv-python`; no extra weights to download
and no RGB ever leaves the engine. The check runs only on the head
region (top 35%) of each detected person bbox so it's cheap.

If you later want a more accurate "looked at" signal:
  - swap in a small frontal-face DNN (yolov8n-face.pt, ~7MB), or
  - run MediaPipe FaceMesh and compute yaw to threshold at +/- 25 deg.

The interface stays the same: gaze.is_looking(rgb, det) -> bool.

Privacy note: this module receives the RGB frame BEFORE
DepthFrame.release_rgb() is called. It returns only a boolean per
detection -- no facial features, embeddings, or crops are retained.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np

from detector import Detection


def _default_cascade_path() -> str:
    """OpenCV ships its cascades alongside the package."""
    base = os.path.dirname(cv2.__file__)
    candidate = os.path.join(base, "data", "haarcascade_frontalface_default.xml")
    return candidate


@dataclass
class GazeConfig:
    enabled: bool = True
    head_region_frac: float = 0.35   # top fraction of person bbox to scan
    min_face_size_frac: float = 0.10  # min face side as frac of bbox width
    scale_factor: float = 1.15
    min_neighbors: int = 4


class GazeDetector:
    def __init__(self, cfg: GazeConfig | None = None, cascade_path: str | None = None):
        self.cfg = cfg or GazeConfig()
        path = cascade_path or _default_cascade_path()
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(
                f"Could not load Haar cascade at {path!r}. "
                "Install OpenCV with: pip install opencv-python"
            )

    def is_looking(self, rgb: np.ndarray, det: Detection) -> bool:
        """Return True if a frontal face is found inside the head region."""
        if not self.cfg.enabled:
            return False

        h, w = rgb.shape[:2]
        x1 = max(0, det.x1)
        y1 = max(0, det.y1)
        x2 = min(w, det.x2)
        y2 = min(h, det.y2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return False

        head_h = max(16, int((y2 - y1) * self.cfg.head_region_frac))
        head_roi = rgb[y1:y1 + head_h, x1:x2]
        if head_roi.size == 0:
            return False

        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        min_side = max(16, int((x2 - x1) * self.cfg.min_face_size_frac))
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.cfg.scale_factor,
            minNeighbors=self.cfg.min_neighbors,
            minSize=(min_side, min_side),
        )
        return len(faces) > 0

    def evaluate_batch(
        self, rgb: np.ndarray, detections: Iterable[Tuple[int, Detection]]
    ) -> dict[int, bool]:
        """Convenience: returns {track_id -> looking} for a batch of (tid, det)."""
        return {tid: self.is_looking(rgb, det) for tid, det in detections}
