"""
Person detector built on top of YOLOv8 (ultralytics).

Wraps the model so the rest of the telemetry engine never has to know
about ultralytics-specific result objects. Returns a clean list of
(x1, y1, x2, y2, confidence) tuples in pixel coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# COCO class index for "person"
PERSON_CLASS_ID = 0


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def centroid(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def foot_point(self) -> tuple[int, int]:
        """Bottom-center of the bbox. Better than centroid for ground-plane
        zones because a person's feet localize them in space, while their
        torso centroid drifts with body angle/lean."""
        return ((self.x1 + self.x2) // 2, self.y2)


class PersonDetector:
    """Thin YOLOv8 wrapper that returns only person detections."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.4,
        iou: float = 0.5,
        device: str | None = None,
        imgsz: int = 640,
        verbose: bool = False,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for PersonDetector. "
                "Install with: pip install ultralytics"
            ) from e
        self.model = YOLO(weights)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.verbose = verbose

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=self.verbose,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        detections: List[Detection] = []
        for (x1, y1, x2, y2), conf in zip(boxes, confs):
            detections.append(
                Detection(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    confidence=float(conf),
                )
            )
        return detections
