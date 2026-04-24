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


def _iou4(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


@dataclass
class GazeConfig:
    enabled: bool = True
    head_region_frac: float = 0.35   # top fraction of person bbox to scan
    min_face_size_frac: float = 0.10  # min face side as frac of bbox width
    scale_factor: float = 1.15
    min_neighbors: int = 2
    min_turn_offset_frac: float = 0.04  # face center offset toward machine side
    use_mediapipe_headpose: bool = True
    strict_yaw_deg: float = 16.0
    attention_yaw_deg: float = 32.0


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
        self._mp_face_mesh = None
        if self.cfg.use_mediapipe_headpose:
            try:
                import mediapipe as mp  # type: ignore
                self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                )
            except Exception:
                self._mp_face_mesh = None

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

    def attention_look_and_face_boxes(
        self, rgb: np.ndarray, det: Detection
    ) -> tuple[bool, bool, list[tuple[int, int, int, int, str]]]:
        """Single pass: Haar + one MediaPipe FaceMesh call, attention/look + overlay boxes.

        Overlay always includes at least the head search region (tag ``HEAD``) when the
        person bbox is valid, so debug view is never empty when cascades miss the face.
        """
        if not self.cfg.enabled:
            return False, False, []

        h_img, w_img = rgb.shape[:2]
        roi_data = self._head_roi(rgb, det)
        if roi_data is None:
            return False, False, []
        x1, y1, head_roi = roi_data
        head_h = head_roi.shape[0]
        row_w = head_roi.shape[1]
        head_strip = (
            int(x1),
            int(y1),
            int(min(w_img, x1 + row_w)),
            int(min(h_img, y1 + head_h)),
        )

        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_side = max(14, int((det.x2 - det.x1) * self.cfg.min_face_size_frac))

        frontal = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.cfg.scale_factor,
            minNeighbors=self.cfg.min_neighbors,
            minSize=(min_side, min_side),
        )
        profile = self._profile.detectMultiScale(
            gray,
            scaleFactor=self.cfg.scale_factor,
            minNeighbors=max(1, self.cfg.min_neighbors - 1),
            minSize=(min_side, min_side),
        )
        flipped = cv2.flip(gray, 1)
        profile_flipped = self._profile.detectMultiScale(
            flipped,
            scaleFactor=self.cfg.scale_factor,
            minNeighbors=max(1, self.cfg.min_neighbors - 1),
            minSize=(min_side, min_side),
        )

        has_frontal = len(frontal) > 0
        has_profile = (len(profile) + len(profile_flipped)) > 0
        attention = has_frontal or has_profile
        looked = has_frontal

        mesh_box: tuple[int, int, int, int] | None = None
        if self._mp_face_mesh is not None:
            try:
                roi_rgb = cv2.cvtColor(head_roi, cv2.COLOR_BGR2RGB)
                out = self._mp_face_mesh.process(roi_rgb)
                if out.multi_face_landmarks:
                    lm = out.multi_face_landmarks[0].landmark
                    rh, rw = head_roi.shape[:2]
                    xs = [p.x * rw for p in lm]
                    ys = [p.y * rh for p in lm]
                    pad = 3
                    mx1 = int(np.clip(x1 + int(min(xs)) - pad, 0, w_img - 1))
                    my1 = int(np.clip(y1 + int(min(ys)) - pad, 0, h_img - 1))
                    mx2 = int(np.clip(x1 + int(max(xs)) + pad, 0, w_img - 1))
                    my2 = int(np.clip(y1 + int(max(ys)) + pad, 0, h_img - 1))
                    if mx2 > mx1 and my2 > my1:
                        mesh_box = (mx1, my1, mx2, my2)

                    nose_x = lm[1].x
                    left_x = lm[33].x
                    right_x = lm[263].x
                    eye_mid = 0.5 * (left_x + right_x)
                    eye_half = max(1e-4, abs(right_x - left_x) * 0.5)
                    yaw_proxy = (nose_x - eye_mid) / eye_half
                    yaw_deg = float(np.clip(yaw_proxy * 35.0, -60.0, 60.0))
                    attention = abs(yaw_deg) <= self.cfg.attention_yaw_deg
                    looked = abs(yaw_deg) <= self.cfg.strict_yaw_deg
            except Exception:
                pass

        haar_boxes: list[tuple[int, int, int, int, str]] = []
        for (fx, fy, fw, fh) in frontal:
            bb = (x1 + int(fx), y1 + int(fy), x1 + int(fx + fw), y1 + int(fy + fh))
            haar_boxes.append((*bb, "FACE"))
        for (fx, fy, fw, fh) in profile:
            bb = (x1 + int(fx), y1 + int(fy), x1 + int(fx + fw), y1 + int(fy + fh))
            haar_boxes.append((*bb, "FACE"))
        rw = head_roi.shape[1]
        for (fx, fy, fw, fh) in profile_flipped:
            ox1 = int(rw - (fx + fw))
            ox2 = int(rw - fx)
            bb = (x1 + ox1, y1 + int(fy), x1 + ox2, y1 + int(fy + fh))
            haar_boxes.append((*bb, "FACE"))

        out_boxes: list[tuple[int, int, int, int, str]] = []
        if mesh_box is not None:
            out_boxes.append((*mesh_box, "MESH"))

        for bb5 in haar_boxes:
            bb4 = bb5[:4]
            if mesh_box is not None and _iou4(bb4, mesh_box) > 0.5:
                continue
            out_boxes.append(bb5)

        if not out_boxes:
            out_boxes.append((*head_strip, "HEAD"))

        return attention, looked, out_boxes

    def attention_and_look(self, rgb: np.ndarray, det: Detection) -> tuple[bool, bool]:
        """Return (attention_intent, looked_at_strict).

        - attention_intent: broader "paying attention to machine/camera side"
        - looked_at_strict: tighter forward/head-facing condition
        """
        a, l, _ = self.attention_look_and_face_boxes(rgb, det)
        return a, l

    def detect_face_boxes(self, rgb: np.ndarray, det: Detection) -> list[tuple[int, int, int, int]]:
        """Return face boxes in full-frame coordinates for debugging/overlay (4-tuple, no tag)."""
        _, _, tagged = self.attention_look_and_face_boxes(rgb, det)
        return [b[:4] for b in tagged]

    def evaluate_batch(
        self, rgb: np.ndarray, detections: Iterable[Tuple[int, Detection]]
    ) -> dict[int, bool]:
        """Convenience: returns {track_id -> looking} for a batch of (tid, det)."""
        return {tid: self.is_looking(rgb, det) for tid, det in detections}
