"""
Centroid tracker.

Assigns persistent integer IDs to detections across frames so we can
attribute dwell time, zone transitions, and funnel events to a single
"person" rather than a per-frame bounding box.

Algorithm: Hungarian assignment on a Euclidean distance matrix between
existing tracked centroids and new detections, with a max-distance gate
to prevent ID swaps when somebody walks past at speed. Tracks that go
unmatched for `max_disappeared` frames are deregistered.

This is intentionally lightweight. For dense, occluded scenes you'd
swap this for DeepSORT or ByteTrack -- the Tracker.update() interface
stays the same.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from detector import Detection


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    centroid: Tuple[int, int]
    foot_point: Tuple[int, int]
    confidence: float
    appearance_hist: np.ndarray | None = None
    disappeared: int = 0
    history: List[Tuple[int, int]] = field(default_factory=list)  # foot_point trail

    def update(self, det: Detection) -> None:
        self.bbox = (det.x1, det.y1, det.x2, det.y2)
        self.centroid = det.centroid
        self.foot_point = det.foot_point
        self.confidence = det.confidence
        if det.appearance_hist is not None:
            if self.appearance_hist is None:
                self.appearance_hist = det.appearance_hist
            else:
                # Exponential moving average to stabilize clothing signature.
                self.appearance_hist = 0.7 * self.appearance_hist + 0.3 * det.appearance_hist
        self.disappeared = 0
        self.history.append(det.foot_point)
        # Keep trail bounded so memory doesn't grow forever on long runs
        if len(self.history) > 64:
            self.history = self.history[-64:]


class CentroidTracker:
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 120.0,
        appearance_weight: float = 0.45,
        edge_margin_px: int = 8,
        edge_grace_multiplier: float = 2.5,
    ):
        """
        Args:
            max_disappeared: number of frames a track can be missing before
                we deregister it. At 30 FPS this is 1 second of occlusion.
            max_distance: max pixel distance allowed between an existing
                track and a candidate detection for them to be matched.
                Prevents teleportation across the frame.
        """
        self.next_id = 0
        self.tracks: "OrderedDict[int, Track]" = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.appearance_weight = appearance_weight
        self.edge_margin_px = edge_margin_px
        self.edge_grace_multiplier = edge_grace_multiplier

    def _is_near_frame_edge(self, track: Track, frame_shape: tuple[int, int] | None) -> bool:
        if frame_shape is None:
            return False
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = track.bbox
        m = self.edge_margin_px
        return x1 <= m or y1 <= m or x2 >= (w - m) or y2 >= (h - m)

    def _register(self, det: Detection) -> Track:
        track = Track(
            track_id=self.next_id,
            bbox=(det.x1, det.y1, det.x2, det.y2),
            centroid=det.centroid,
            foot_point=det.foot_point,
            confidence=det.confidence,
            appearance_hist=det.appearance_hist,
            history=[det.foot_point],
        )
        self.tracks[self.next_id] = track
        self.next_id += 1
        return track

    def _deregister(self, track_id: int) -> None:
        if track_id in self.tracks:
            del self.tracks[track_id]

    def update(
        self, detections: List[Detection], frame_shape: tuple[int, int] | None = None
    ) -> Dict[int, Track]:
        # No detections this frame -- age all existing tracks
        if not detections:
            stale = []
            for tid, track in self.tracks.items():
                track.disappeared += 1
                max_age = self.max_disappeared
                if self._is_near_frame_edge(track, frame_shape):
                    max_age = int(self.max_disappeared * self.edge_grace_multiplier)
                if track.disappeared > max_age:
                    stale.append(tid)
            for tid in stale:
                self._deregister(tid)
            return dict(self.tracks)

        # No existing tracks -- register all detections as new
        if not self.tracks:
            for det in detections:
                self._register(det)
            return dict(self.tracks)

        # Build cost matrix between existing track foot-points and new detections.
        # Foot-points work better than centroids for ground-plane zones because
        # they're invariant to whether somebody bends over the touchscreen.
        track_ids = list(self.tracks.keys())
        track_pts = np.array([self.tracks[tid].foot_point for tid in track_ids])
        det_pts = np.array([det.foot_point for det in detections])

        # cost[i, j] = distance from track i to detection j
        cost = np.linalg.norm(
            track_pts[:, None, :] - det_pts[None, :, :], axis=2
        )

        # Appearance penalty based on clothing color histogram similarity.
        # Lower is better, range approx [0, 1].
        if self.appearance_weight > 0:
            app_penalty = np.zeros_like(cost, dtype=np.float32)
            for i, tid in enumerate(track_ids):
                th = self.tracks[tid].appearance_hist
                if th is None:
                    continue
                for j, det in enumerate(detections):
                    dh = det.appearance_hist
                    if dh is None:
                        continue
                    dist = cv2.compareHist(
                        th.astype(np.float32),
                        dh.astype(np.float32),
                        cv2.HISTCMP_BHATTACHARYYA,
                    )
                    app_penalty[i, j] = float(dist)
            cost = cost + (self.appearance_weight * self.max_distance * app_penalty)

        # Gate: any pairing further than max_distance is forbidden
        cost_for_assign = cost.copy()
        cost_for_assign[cost > self.max_distance] = 1e6

        row_ind, col_ind = linear_sum_assignment(cost_for_assign)

        matched_tracks = set()
        matched_dets = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > self.max_distance:
                continue
            tid = track_ids[r]
            self.tracks[tid].update(detections[c])
            matched_tracks.add(tid)
            matched_dets.add(c)

        # Unmatched existing tracks -> age them, possibly deregister
        stale = []
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].disappeared += 1
                max_age = self.max_disappeared
                if self._is_near_frame_edge(self.tracks[tid], frame_shape):
                    max_age = int(self.max_disappeared * self.edge_grace_multiplier)
                if self.tracks[tid].disappeared > max_age:
                    stale.append(tid)
        for tid in stale:
            self._deregister(tid)

        # Unmatched detections -> register as new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._register(det)

        return dict(self.tracks)
