"""
Zone manager.

Two flavors of zone:

  Polygon zones  – pixel polygons over the camera frame. Used by the
                   plain RGB pipeline. Containment via foot-point.

  Distance-band  – meters from the camera (which is mounted ON TOP of
   zones           the machine), used by the depth-camera pipeline.
                   Resolution-independent and survives camera repositioning
                   without recalibration.

Both implement the same `contains(...)` interface; the funnel state
machine doesn't care which kind it's working with.

We use polygons (not rectangles) for the RGB case because vending
machines sit at angles, in hallways, and against curved walls -- a
rectangular ROI almost always either over-counts the corridor or
under-counts the machine itself. With depth, we don't need polygons
at all -- "within 0.6 m of the machine" is a perfect engagement
definition that works on day 1 with no calibration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# ── Polygon zones (RGB pipeline) ────────────────────────────────────────────

@dataclass
class Zone:
    name: str
    polygon: np.ndarray  # shape (N, 2), int32, pixel coords
    color: Tuple[int, int, int]  # BGR

    def contains(self, point: Tuple[int, int], distance_m: Optional[float] = None) -> bool:
        # cv2.pointPolygonTest returns +1 inside, 0 on edge, -1 outside.
        # measureDist=False is much faster -- we only need a boolean.
        return (
            cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False)
            >= 0
        )


# ── Distance-band zones (depth pipeline) ────────────────────────────────────

@dataclass
class DistanceBandZone:
    """A zone defined by 'how far from the camera' in meters.

    With the camera mounted on top of the machine, distance from camera
    ≈ distance from machine, which is exactly what we want for funnel
    semantics. No floor calibration needed.
    """
    name: str
    min_m: float
    max_m: float
    color: Tuple[int, int, int]

    def contains(self, point: Tuple[int, int], distance_m: Optional[float] = None) -> bool:
        if distance_m is None or not math.isfinite(distance_m):
            return False
        return self.min_m <= distance_m <= self.max_m


AnyZone = Union[Zone, DistanceBandZone]


class ZoneManager:
    """Holds the three funnel zones and renders the overlay.

    Accepts a mix of polygon and distance-band zones. The funnel state
    machine is uniform -- it just asks "is this person in zone X?"
    """

    # Conventional names used throughout the engine. Renaming these will
    # break the funnel state machine -- update funnel.py too.
    FOOT_TRAFFIC = "foot_traffic"
    ENGAGEMENT = "engagement"
    TRANSACTION = "transaction"

    def __init__(self, zones: List[AnyZone]):
        self.zones: Dict[str, AnyZone] = {z.name: z for z in zones}
        self._validate()

    def _validate(self) -> None:
        required = {self.FOOT_TRAFFIC, self.ENGAGEMENT, self.TRANSACTION}
        missing = required - set(self.zones.keys())
        if missing:
            raise ValueError(
                f"ZoneManager missing required zones: {sorted(missing)}. "
                f"Run calibrate.py to define them."
            )

    def zones_containing(
        self,
        point: Tuple[int, int],
        distance_m: Optional[float] = None,
    ) -> List[str]:
        """Return names of all zones containing the point.

        Zones are nested (transaction ⊂ engagement ⊂ foot_traffic), so a
        person standing at the touchscreen is in all three. The funnel
        state machine handles the precedence.
        """
        return [name for name, zone in self.zones.items()
                if zone.contains(point, distance_m=distance_m)]

    def draw(self, frame: np.ndarray, alpha: float = 0.18) -> np.ndarray:
        """Draw zone overlays. Polygon zones get filled polygons;
        distance-band zones get a small legend in the corner since
        they have no pixel geometry."""
        overlay = frame.copy()
        polygon_zones = [z for z in self.zones.values() if isinstance(z, Zone)]
        band_zones = [z for z in self.zones.values() if isinstance(z, DistanceBandZone)]

        for zone in polygon_zones:
            cv2.fillPoly(overlay, [zone.polygon], zone.color)
        out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        for zone in polygon_zones:
            cv2.polylines(out, [zone.polygon], isClosed=True, color=zone.color, thickness=2)
            label_pt = tuple(zone.polygon[zone.polygon[:, 1].argmin()])
            cv2.putText(
                out, zone.name.upper(),
                (label_pt[0], max(label_pt[1] - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone.color, 2, cv2.LINE_AA,
            )

        if band_zones:
            # Legend: bottom-right, one row per distance band
            x = out.shape[1] - 280
            y = out.shape[0] - 12 - 22 * len(band_zones)
            cv2.rectangle(out, (x - 10, y - 22), (out.shape[1] - 10, out.shape[0] - 8),
                          (0, 0, 0), -1)
            for i, z in enumerate(band_zones):
                yi = y + i * 22
                cv2.rectangle(out, (x, yi - 14), (x + 18, yi + 2), z.color, -1)
                txt = f"{z.name.upper():13s} {z.min_m:.1f}-{z.max_m:.1f} m"
                cv2.putText(out, txt, (x + 26, yi),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    @classmethod
    def from_config(cls, config: dict, frame_shape: Tuple[int, int]) -> "ZoneManager":
        """Build from a parsed YAML config block.

        Each zone entry can be either:
          - polygon:  {name, color, polygon: [[x,y], ...]}
          - distance: {name, color, min_m, max_m}

        Polygon coords may be absolute pixels (ints) or normalized 0-1
        fractions (floats). Normalized is preferred because it survives
        camera-resolution changes.
        """
        h, w = frame_shape[:2] if frame_shape else (0, 0)
        zones: List[AnyZone] = []
        for entry in config.get("zones", []):
            color = tuple(entry.get("color", [0, 255, 0]))
            if "min_m" in entry or "max_m" in entry:
                zones.append(DistanceBandZone(
                    name=entry["name"],
                    min_m=float(entry.get("min_m", 0.0)),
                    max_m=float(entry.get("max_m", float("inf"))),
                    color=color,
                ))
                continue

            poly = []
            for x, y in entry["polygon"]:
                if isinstance(x, float) and 0 <= x <= 1 and isinstance(y, float) and 0 <= y <= 1:
                    poly.append([int(x * w), int(y * h)])
                else:
                    poly.append([int(x), int(y)])
            zones.append(
                Zone(name=entry["name"], polygon=np.array(poly, dtype=np.int32), color=color)
            )
        return cls(zones)
