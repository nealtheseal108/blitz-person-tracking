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


def _default_profile_cascade_path() -> str:
    base = os.path.dirname(cv2.__file__)
    return os.path.join(base, "data", "haarcascade_profileface.xml")


@dataclass
class GazeConfig:
    enabled: bool = True
    head_region_frac: float = 0.35   # top fraction of person bbox to scan
    min_face_size_frac: float = 0.10  # min face side as frac of bbox width
    scale_factor: float = 1.15
    min_neighbors: int = 2
    min_turn_offset_frac: float = 0.04  # face center offset toward machine side


class GazeDetector:
    def __init__(self, cfg: GazeConfig | None = None, cascade_path: str | None = None):
        self.cfg = cfg or GazeConfig()
        path = cascade_path or _default_cascade_path()
        self._cascade = cv2.CascadeClassifier(path)
        profile_path = _default_profile_cascade_path()
        self._profile = cv2.CascadeClassifier(profile_path)
        if self._cascade.empty():
            raise RuntimeError(
                f"Could not load Haar cascade at {path!r}. "
                "Install OpenCV with: pip install opencv-python"
            )
        if self._profile.empty():
            raise RuntimeError(
                f"Could not load Haar profile cascade at {profile_path!r}. "
                "Install OpenCV with: pip install opencv-python"
            )

    def _head_roi(self, rgb: np.ndarray, det: Detection) -> tuple[int, int, np.ndarray] | None:
        h, w = rgb.shape[:2]
        x1 = max(0, det.x1)
        y1 = max(0, det.y1)
        x2 = min(w, det.x2)
        y2 = min(h, det.y2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        head_h = max(16, int((y2 - y1) * self.cfg.head_region_frac))
        head_roi = rgb[y1:y1 + head_h, x1:x2]
        if head_roi.size == 0:
            return None
        return x1, y1, head_roi

    def is_looking(self, rgb: np.ndarray, det: Detection) -> bool:
        """Back-compat path: any detectable face in head region."""
        if not self.cfg.enabled:
            return False
        roi_data = self._head_roi(rgb, det)
        if roi_data is None:
            return False
        _, _, head_roi = roi_data
        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_side = max(16, int((det.x2 - det.x1) * self.cfg.min_face_size_frac))
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.cfg.scale_factor,
            minNeighbors=self.cfg.min_neighbors,
            minSize=(min_side, min_side),
        )
        return len(faces) > 0

    def is_looking_toward_machine(
        self,
        rgb: np.ndarray,
        det: Detection,
        machine_point: tuple[int, int],
    ) -> bool:
        """True when head orientation appears turned toward machine side."""
        if not self.cfg.enabled:
            return False
        roi_data = self._head_roi(rgb, det)
        if roi_data is None:
            return False
        x1, _, head_roi = roi_data

        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_side = max(16, int((det.x2 - det.x1) * self.cfg.min_face_size_frac))
        frontal = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.cfg.scale_factor,
            minNeighbors=self.cfg.min_neighbors,
            minSize=(min_side, min_side),
        )
        profile = self._profile.detectMultiScale(
            gray,
            scaleFactor=self.cfg.scale_factor,
            minNeighbors=max(3, self.cfg.min_neighbors - 1),
            minSize=(min_side, min_side),
        )
        flipped = cv2.flip(gray, 1)
        profile_flipped = self._profile.detectMultiScale(
            flipped,
            scaleFactor=self.cfg.scale_factor,
            minNeighbors=max(3, self.cfg.min_neighbors - 1),
            minSize=(min_side, min_side),
        )

        candidates: list[tuple[float, float]] = []
        for (fx, fy, fw, fh) in frontal:
            candidates.append((x1 + fx + fw / 2.0, fy + fh / 2.0))
        for (fx, fy, fw, fh) in profile:
            candidates.append((x1 + fx + fw / 2.0, fy + fh / 2.0))
        for (fx, fy, fw, fh) in profile_flipped:
            # Map back from flipped ROI coordinates.
            mapped_x = head_roi.shape[1] - (fx + fw / 2.0)
            candidates.append((x1 + mapped_x, fy + fh / 2.0))

        if not candidates:
            return False

        machine_x, _ = machine_point
        body_center_x = (det.x1 + det.x2) / 2.0
        side = 1.0 if machine_x >= body_center_x else -1.0
        bw = max(1.0, float(det.x2 - det.x1))
        for face_cx, _ in candidates:
            offset = (face_cx - body_center_x) / bw
            if side * offset >= self.cfg.min_turn_offset_frac:
                return True
        return False

    def evaluate_batch(
        self, rgb: np.ndarray, detections: Iterable[Tuple[int, Detection]]
    ) -> dict[int, bool]:
        """Convenience: returns {track_id -> looking} for a batch of (tid, det)."""
        return {tid: self.is_looking(rgb, det) for tid, det in detections}
