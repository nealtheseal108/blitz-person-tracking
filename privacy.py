"""
Privacy redaction.

The defining principle: the engine never has to keep an image of a real
person. Frames are processed in memory, telemetry is extracted, then
the frame is either dropped or redacted before it touches disk.

This module returns a redacted *copy* of a frame given the person
detections. Three redaction modes:

    "off"            – no change. Use only during calibration.
    "face_blur"      – heavy gaussian blur on the face region only
                       (top ~28% of each person bbox). Body still visible.
    "body_pixelate"  – mosaic the entire person bbox. Identifiable
                       features (face, clothing logos, jewelry) gone.
    "silhouette"     – replace the person bbox with a flat-color
                       silhouette. Strongest visual privacy; used by
                       most retail-vision vendors for compliance demos.

By default the engine is configured with `save_raw_frames: false` AND
`mode: silhouette`, so even debugging videos contain no biometrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from detector import Detection


VALID_MODES = ("off", "face_blur", "body_pixelate", "silhouette")


@dataclass
class PrivacyConfig:
    mode: str = "silhouette"
    blur_kernel: int = 51        # odd; larger = more blur
    pixelate_blocks: int = 12    # mosaic blocks across the bbox width
    silhouette_color: Tuple[int, int, int] = (60, 60, 60)  # BGR

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"invalid privacy mode: {self.mode!r}; pick from {VALID_MODES}")
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _blur_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, k: int) -> None:
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def _pixelate_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, blocks: int) -> None:
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    bw = max(1, w // blocks)
    bh = max(1, h // blocks)
    # Down-sample then upsample with nearest-neighbor to get a chunky mosaic
    small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _silhouette_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                       color: Tuple[int, int, int]) -> None:
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)


def redact(frame: np.ndarray, detections: Iterable[Detection],
           cfg: PrivacyConfig) -> np.ndarray:
    """Return a redacted copy of `frame`. Original is not mutated."""
    if cfg.mode == "off":
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = _clip_box(det.x1, det.y1, det.x2, det.y2, w, h)
        if cfg.mode == "face_blur":
            # Top ~28% of bbox is the head/face region for an upright person
            face_h = max(1, int((y2 - y1) * 0.28))
            _blur_region(out, x1, y1, x2, y1 + face_h, cfg.blur_kernel)
        elif cfg.mode == "body_pixelate":
            _pixelate_region(out, x1, y1, x2, y2, cfg.pixelate_blocks)
        elif cfg.mode == "silhouette":
            _silhouette_region(out, x1, y1, x2, y2, cfg.silhouette_color)
    return out


def redact_inplace(frame: np.ndarray, detections: Iterable[Detection],
                   cfg: PrivacyConfig) -> None:
    """In-place version for hot paths where we don't need the original."""
    if cfg.mode == "off":
        return
    h, w = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = _clip_box(det.x1, det.y1, det.x2, det.y2, w, h)
        if cfg.mode == "face_blur":
            face_h = max(1, int((y2 - y1) * 0.28))
            _blur_region(frame, x1, y1, x2, y1 + face_h, cfg.blur_kernel)
        elif cfg.mode == "body_pixelate":
            _pixelate_region(frame, x1, y1, x2, y2, cfg.pixelate_blocks)
        elif cfg.mode == "silhouette":
            _silhouette_region(frame, x1, y1, x2, y2, cfg.silhouette_color)
