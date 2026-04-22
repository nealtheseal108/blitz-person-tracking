"""
Funnel state machine.

Each tracked person walks through a 3-stage funnel:

    VISITOR  ──►  ENGAGED  ──►  CONVERTED
    (entered     (dwelt at     (sustained
     foot         the           presence at
     traffic      machine)      touchscreen,
     zone)                      OR external
                                POS confirm)

We require *dwell time* to advance, not a single frame of containment.
That filters out people who are just walking past on their way to class
-- which is the dominant noise source in any retail-vision deployment.

Events are emitted as JSON-serializable dicts and pushed to the
`event_sink` callable so the main engine can stream them to a JSONL
file, a webhook, Kafka, etc.

A "transaction" can also be triggered externally via
FunnelTracker.confirm_transaction(track_id) -- wire that up to the
vending-machine POS to get a ground-truth conversion signal instead of
relying on dwell-time inference.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from tracker import Track
from zones import ZoneManager


class FunnelStage(str, Enum):
    VISITOR = "visitor"
    ENGAGED = "engaged"
    CONVERTED = "converted"


@dataclass
class FunnelState:
    track_id: int
    stage: FunnelStage = FunnelStage.VISITOR
    first_seen_ts: float = 0.0
    engagement_zone_entered_ts: Optional[float] = None
    engaged_ts: Optional[float] = None
    transaction_zone_entered_ts: Optional[float] = None
    converted_ts: Optional[float] = None
    last_seen_ts: float = 0.0
    # Distance metrics from depth pipeline (NaN if RGB-only)
    min_distance_m: float = float("nan")
    last_distance_m: float = float("nan")
    # "Looked at machine" counter -- a debounced count of glance episodes.
    # Set by the gaze module via FunnelTracker.note_glance(track_id).
    look_count: int = 0
    _looking_now: bool = False
    _looking_streak: int = 0
    # Per-zone accumulated dwell in seconds (handy for analytics)
    counted_visitor: bool = False
    seen_frames: int = 0
    foot_traffic_frames: int = 0
    first_foot_point: Optional[tuple[int, int]] = None
    dwell: Dict[str, float] = field(default_factory=lambda: {
        ZoneManager.FOOT_TRAFFIC: 0.0,
        ZoneManager.ENGAGEMENT: 0.0,
        ZoneManager.TRANSACTION: 0.0,
    })


EventSink = Callable[[dict], None]


class FunnelTracker:
    def __init__(
        self,
        zone_manager: ZoneManager,
        engagement_dwell_sec: float = 2.0,
        transaction_dwell_sec: float = 6.0,
        visitor_dedupe_sec: float = 3.0,
        visitor_dedupe_px: float = 90.0,
        visitor_min_confidence: float = 0.45,
        visitor_min_motion_px: float = 10.0,
        visitor_min_seen_frames: int = 6,
        visitor_min_foot_traffic_frames: int = 4,
        visitor_min_displacement_px: float = 28.0,
        visitor_min_path_px: float = 42.0,
        event_sink: Optional[EventSink] = None,
    ):
        """
        Args:
            engagement_dwell_sec: how long a person must remain in the
                engagement zone before we count them as engaged.
                2s is a good default -- shorter and you count walk-bys;
                longer and you miss quick-purchase regulars.
            transaction_dwell_sec: same idea, for the transaction zone.
                6s roughly matches the time to tap through a vending
                purchase. If you have a real POS hook, set this very
                high (e.g. 999) so dwell never trips and only the POS
                signal counts as a conversion.
            event_sink: callable that receives every emitted event dict.
        """
        self.zones = zone_manager
        self.engagement_dwell_sec = engagement_dwell_sec
        self.transaction_dwell_sec = transaction_dwell_sec
        self.visitor_dedupe_sec = visitor_dedupe_sec
        self.visitor_dedupe_px = visitor_dedupe_px
        self.visitor_min_confidence = visitor_min_confidence
        self.visitor_min_motion_px = visitor_min_motion_px
        self.visitor_min_seen_frames = visitor_min_seen_frames
        self.visitor_min_foot_traffic_frames = visitor_min_foot_traffic_frames
        self.visitor_min_displacement_px = visitor_min_displacement_px
        self.visitor_min_path_px = visitor_min_path_px
        self.event_sink = event_sink or (lambda _: None)

        self.states: Dict[int, FunnelState] = {}
        self._last_update_ts: Optional[float] = None

        # Lifetime counters (unique IDs counted once per stage)
        self.totals = {
            FunnelStage.VISITOR: 0,
            FunnelStage.ENGAGED: 0,
            FunnelStage.CONVERTED: 0,
        }
        # Aggregate-only counters (no per-person retention required)
        self.unique_lookers: int = 0   # unique track ids that ever looked
        self.click_count: int = 0      # button presses reported by the machine
        self.approached_ids: set[int] = set()  # active motion toward machine
        # Recent visitor sightings for anti-double-counting when tracker IDs flicker.
        self._recent_visitor_points: deque[tuple[float, tuple[int, int]]] = deque(maxlen=300)

    def _is_duplicate_visitor(self, now: float, foot_point: tuple[int, int]) -> bool:
        # Drop stale sightings first.
        while self._recent_visitor_points and now - self._recent_visitor_points[0][0] > self.visitor_dedupe_sec:
            self._recent_visitor_points.popleft()
        fx, fy = foot_point
        for _, (px, py) in self._recent_visitor_points:
            if ((fx - px) ** 2 + (fy - py) ** 2) <= (self.visitor_dedupe_px ** 2):
                return True
        return False

    def _track_motion_px(self, track: Track) -> float:
        if len(track.history) < 4:
            return 0.0
        x0, y0 = track.history[-4]
        x1, y1 = track.history[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    def _track_path_px(self, track: Track) -> float:
        if len(track.history) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(track.history)):
            x0, y0 = track.history[i - 1]
            x1, y1 = track.history[i]
            total += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return total

    def _emit(self, event_type: str, state: FunnelState, **extra) -> None:
        ev = {
            "ts": time.time(),
            "event": event_type,
            "track_id": state.track_id,
            "stage": state.stage.value,
            **extra,
        }
        self.event_sink(ev)

    def update(
        self,
        tracks: Dict[int, Track],
        now: Optional[float] = None,
        distances_m: Optional[Dict[int, float]] = None,
    ) -> None:
        """Call once per frame with the current set of tracks.

        distances_m: optional {track_id -> distance_in_meters} from the
            depth pipeline. When supplied, distance-band zones become
            usable and per-track min_distance_m is recorded.
        """
        now = now if now is not None else time.time()
        dt = 0.0 if self._last_update_ts is None else max(0.0, now - self._last_update_ts)
        self._last_update_ts = now
        distances_m = distances_m or {}

        active_ids = set(tracks.keys())

        for tid, track in tracks.items():
            d = distances_m.get(tid)
            zones_in = set(self.zones.zones_containing(track.foot_point, distance_m=d))

            # New visitor: first time we've seen this track id at all
            if tid not in self.states:
                self.states[tid] = FunnelState(
                    track_id=tid,
                    first_seen_ts=now,
                    last_seen_ts=now,
                    counted_visitor=False,
                )

            state = self.states[tid]
            state.last_seen_ts = now
            state.seen_frames += 1
            if state.first_foot_point is None:
                state.first_foot_point = track.foot_point
            if ZoneManager.FOOT_TRAFFIC in zones_in:
                state.foot_traffic_frames += 1

            # Visitor counting gate: must be in foot_traffic, confident, and moving.
            if not state.counted_visitor:
                if ZoneManager.FOOT_TRAFFIC in zones_in and track.confidence >= self.visitor_min_confidence:
                    motion_px = self._track_motion_px(track)
                    path_px = self._track_path_px(track)
                    disp_px = 0.0
                    if state.first_foot_point is not None:
                        fx, fy = state.first_foot_point
                        cx, cy = track.foot_point
                        disp_px = ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5
                    if (
                        motion_px >= self.visitor_min_motion_px
                        and state.seen_frames >= self.visitor_min_seen_frames
                        and state.foot_traffic_frames >= self.visitor_min_foot_traffic_frames
                        and disp_px >= self.visitor_min_displacement_px
                        and path_px >= self.visitor_min_path_px
                    ):
                        is_duplicate = self._is_duplicate_visitor(now, track.foot_point)
                        if not is_duplicate:
                            state.counted_visitor = True
                            self.totals[FunnelStage.VISITOR] += 1
                            self._recent_visitor_points.append((now, track.foot_point))
                            self._emit(
                                "visitor_seen",
                                state,
                                foot_point=track.foot_point,
                                zones=sorted(zones_in),
                            )
                        else:
                            self._emit(
                                "visitor_deduped",
                                state,
                                foot_point=track.foot_point,
                            )

            # Track distance metrics (NaN-safe min)
            if d is not None and d == d:  # not NaN
                state.last_distance_m = d
                if not (state.min_distance_m == state.min_distance_m):  # currently NaN
                    state.min_distance_m = d
                else:
                    state.min_distance_m = min(state.min_distance_m, d)

            # Accumulate per-zone dwell time
            for zone_name in zones_in:
                state.dwell[zone_name] = state.dwell.get(zone_name, 0.0) + dt

            # ── Stage transition: VISITOR → ENGAGED ────────────────────────
            if state.stage == FunnelStage.VISITOR:
                if ZoneManager.ENGAGEMENT in zones_in:
                    if state.engagement_zone_entered_ts is None:
                        state.engagement_zone_entered_ts = now
                    elif now - state.engagement_zone_entered_ts >= self.engagement_dwell_sec:
                        state.stage = FunnelStage.ENGAGED
                        state.engaged_ts = now
                        if state.counted_visitor:
                            self.totals[FunnelStage.ENGAGED] += 1
                        self._emit(
                            "engagement_start",
                            state,
                            dwell_sec=round(now - state.engagement_zone_entered_ts, 2),
                        )
                else:
                    # Reset dwell timer if they leave engagement zone before
                    # the threshold. Walk-bys shouldn't accumulate credit.
                    state.engagement_zone_entered_ts = None

            # ── Stage transition: ENGAGED → CONVERTED ──────────────────────
            if state.stage == FunnelStage.ENGAGED:
                if ZoneManager.TRANSACTION in zones_in:
                    if state.transaction_zone_entered_ts is None:
                        state.transaction_zone_entered_ts = now
                    elif now - state.transaction_zone_entered_ts >= self.transaction_dwell_sec:
                        self._mark_converted(state, now, source="dwell")
                else:
                    state.transaction_zone_entered_ts = None

        # Tracks that disappeared this frame -- emit lifecycle events but
        # keep their FunnelState around so late POS confirms can still
        # attribute a conversion to them.
        for tid in list(self.states.keys()):
            if tid not in active_ids:
                state = self.states[tid]
                # Only emit "track_lost" once -- we mark by setting
                # last_seen_ts > 0 and stamping a sentinel.
                if not getattr(state, "_lost_emitted", False):
                    self._emit(
                        "track_lost",
                        state,
                        total_dwell=round(sum(state.dwell.values()), 2),
                    )
                    state._lost_emitted = True  # type: ignore[attr-defined]

    # ── Glance / "looking at machine" tracking ─────────────────────────────

    def note_glance(self, track_id: int, looking: bool, now: Optional[float] = None) -> None:
        """Called by the gaze module each frame with the per-track gaze
        decision. We hysteresis-debounce by counting one glance per
        on-edge transition (looking=False -> True), so a continuous
        stare counts as ONE glance, not 30/sec.
        """
        state = self.states.get(track_id)
        if state is None:
            return
        if looking:
            state._looking_streak += 1
        else:
            state._looking_streak = 0
        # Require a tiny temporal confirmation window (2 frames) to reduce
        # cascade jitter while still being responsive.
        confirmed_looking = state._looking_streak >= 2
        if confirmed_looking and not state._looking_now:
            state.look_count += 1
            if state.look_count == 1 and state.counted_visitor:
                self.unique_lookers += 1
            self._emit("glance", state, look_count=state.look_count)
        state._looking_now = confirmed_looking

    def note_approach_motion(self, track_id: int) -> None:
        """Count approach only after at least one look event.

        This enforces the product rule:
          looked_at (head movement) should be >= approached (full-body movement).
        """
        state = self.states.get(track_id)
        if state is None:
            return
        if state.look_count <= 0:
            return
        if not state.counted_visitor:
            return
        if track_id in self.approached_ids:
            return
        self.approached_ids.add(track_id)
        self._emit("approach_motion", state)

    # ── External click signal from the machine itself ──────────────────────

    def note_click(self, source: str = "machine") -> None:
        """Called whenever the machine reports a button press / tap.
        These are ground-truth interactions independent of the camera.
        """
        self.click_count += 1
        self._emit("click", FunnelState(track_id=-1), source=source)

    def confirm_transaction(self, track_id: int, source: str = "pos") -> bool:
        """Externally mark a track as converted (e.g. POS webhook).

        Returns True if a state was updated, False if no such track.
        """
        state = self.states.get(track_id)
        if state is None or state.stage == FunnelStage.CONVERTED:
            return False
        self._mark_converted(state, time.time(), source=source)
        return True

    def confirm_most_recent_engaged(self, source: str = "pos") -> Optional[int]:
        """Convenience: the POS doesn't know our track IDs, so when a
        purchase fires, attribute it to whichever currently-engaged
        track has been engaged longest. Returns the track_id (or None).
        """
        candidates = [
            s for s in self.states.values()
            if s.stage == FunnelStage.ENGAGED and s.engaged_ts is not None
        ]
        if not candidates:
            return None
        winner = min(candidates, key=lambda s: s.engaged_ts or 0)
        self._mark_converted(winner, time.time(), source=source)
        return winner.track_id

    def _mark_converted(self, state: FunnelState, now: float, source: str) -> None:
        state.stage = FunnelStage.CONVERTED
        state.converted_ts = now
        if state.counted_visitor:
            self.totals[FunnelStage.CONVERTED] += 1
        self._emit("transaction", state, source=source)

    # ── Reporting ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        # Aliases mirror the user-facing funnel names:
        #   passed_by    = anyone seen at all (entered foot_traffic)
        #   looked_at    = unique people whose face the gaze module saw
        #   approached   = unique people who reached engagement (close)
        #   clicked      = button presses reported by the machine
        #   purchased    = converted transactions (POS-confirmed if wired)
        v = self.totals[FunnelStage.VISITOR]
        e = self.totals[FunnelStage.ENGAGED]
        c = self.totals[FunnelStage.CONVERTED]
        ul = self.unique_lookers
        clicks = self.click_count
        # If motion-based approach detection is wired, use it; otherwise
        # fall back to ENGAGED for backward compatibility.
        approached_raw = len(self.approached_ids) if self.approached_ids else e
        approached = min(approached_raw, ul)
        return {
            "passed_by": v,
            "looked_at": ul,
            "approached": approached,
            "clicked": clicks,
            "purchased": c,
            "look_rate":        round(ul / v, 4) if v else 0.0,    # of passed-by
            "approach_rate":    round(approached / v, 4) if v else 0.0,
            "click_rate":       round(clicks / approached, 4) if approached else 0.0,
            "purchase_rate":    round(c / approached, 4) if approached else 0.0,
            "overall_conversion": round(c / v, 4) if v else 0.0,
        }

    def per_track_rows(self) -> List[dict]:
        """Flattened per-track records suitable for a CSV dump.

        Note: distance + look_count fields are NaN/0 when the depth or
        gaze modules aren't wired in -- that's intentional, the schema
        is stable across configurations.
        """
        rows = []
        for s in self.states.values():
            rows.append({
                "track_id": s.track_id,
                "stage": s.stage.value,
                "first_seen_ts": s.first_seen_ts,
                "engaged_ts": s.engaged_ts or "",
                "converted_ts": s.converted_ts or "",
                "last_seen_ts": s.last_seen_ts,
                "min_distance_m": round(s.min_distance_m, 2)
                    if s.min_distance_m == s.min_distance_m else "",
                "look_count": s.look_count,
                "dwell_foot_traffic_sec": round(s.dwell.get(ZoneManager.FOOT_TRAFFIC, 0.0), 2),
                "dwell_engagement_sec": round(s.dwell.get(ZoneManager.ENGAGEMENT, 0.0), 2),
                "dwell_transaction_sec": round(s.dwell.get(ZoneManager.TRANSACTION, 0.0), 2),
            })
        return rows
