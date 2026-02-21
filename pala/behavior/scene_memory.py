from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .models import SceneObservation


@dataclass(frozen=True)
class SceneMemorySnapshot:
    now_s: float
    person_recently_seen: bool
    last_seen_age_s: Optional[float]
    present_streak_s: float
    absent_streak_s: float
    likely_zone: Optional[str]
    zone_transitions_recent: int
    dominant_activity: Optional[str]
    recent_event_count: int


@dataclass(frozen=True)
class _MemoryEvent:
    ts_s: float
    event: str
    zone: Optional[str]
    activity: Optional[str]
    person_present: bool


class SceneMemory:
    """Rolling scene/user-action memory for behavior-time planning."""

    def __init__(self, *, history_max: int = 256, recently_seen_s: float = 2.5):
        self._events: deque[_MemoryEvent] = deque(maxlen=max(32, int(history_max)))
        self._recently_seen_s = max(0.2, float(recently_seen_s))
        self._last_seen_s: Optional[float] = None
        self._last_zone: Optional[str] = None
        self._last_ts_s: Optional[float] = None
        self._present_streak_s = 0.0
        self._absent_streak_s = 0.0

    def update(self, obs: SceneObservation) -> SceneMemorySnapshot:
        event = _MemoryEvent(
            ts_s=float(obs.ts_mono_s),
            event=obs.event,
            zone=obs.zone,
            activity=obs.activity_hint,
            person_present=bool(obs.person_present),
        )
        self._events.append(event)
        dt_s = 0.0
        if self._last_ts_s is not None:
            dt_s = max(0.0, float(obs.ts_mono_s) - self._last_ts_s)
        self._last_ts_s = float(obs.ts_mono_s)
        if obs.person_present:
            self._present_streak_s += dt_s
            self._absent_streak_s = 0.0
        else:
            self._absent_streak_s += dt_s
            self._present_streak_s = 0.0
        if obs.person_present:
            self._last_seen_s = obs.ts_mono_s
            if obs.zone is not None:
                self._last_zone = obs.zone

        now = float(obs.ts_mono_s)
        last_seen_age_s: Optional[float] = None
        person_recently_seen = False
        if self._last_seen_s is not None:
            last_seen_age_s = max(0.0, now - self._last_seen_s)
            person_recently_seen = last_seen_age_s <= self._recently_seen_s

        likely_zone = self._dominant_zone(now_s=now, window_s=6.0) or self._last_zone
        zone_transitions_recent = self._zone_transitions(now_s=now, window_s=4.0)
        dominant_activity = self._dominant_activity(now_s=now, window_s=8.0)
        recent_event_count = self._event_count(now_s=now, window_s=8.0)
        return SceneMemorySnapshot(
            now_s=now,
            person_recently_seen=person_recently_seen,
            last_seen_age_s=last_seen_age_s,
            present_streak_s=self._present_streak_s,
            absent_streak_s=self._absent_streak_s,
            likely_zone=likely_zone,
            zone_transitions_recent=zone_transitions_recent,
            dominant_activity=dominant_activity,
            recent_event_count=recent_event_count,
        )

    def _window(self, *, now_s: float, window_s: float) -> list[_MemoryEvent]:
        cutoff = now_s - max(0.1, float(window_s))
        return [e for e in self._events if e.ts_s >= cutoff]

    def _event_count(self, *, now_s: float, window_s: float) -> int:
        return len(self._window(now_s=now_s, window_s=window_s))

    def _dominant_zone(self, *, now_s: float, window_s: float) -> Optional[str]:
        counts: dict[str, int] = {}
        for e in self._window(now_s=now_s, window_s=window_s):
            if not e.person_present or e.zone is None:
                continue
            counts[e.zone] = counts.get(e.zone, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _dominant_activity(self, *, now_s: float, window_s: float) -> Optional[str]:
        counts: dict[str, int] = {}
        for e in self._window(now_s=now_s, window_s=window_s):
            if e.activity is None:
                continue
            counts[e.activity] = counts.get(e.activity, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _zone_transitions(self, *, now_s: float, window_s: float) -> int:
        seq: list[str] = []
        for e in self._window(now_s=now_s, window_s=window_s):
            if not e.person_present or e.zone is None:
                continue
            if not seq or seq[-1] != e.zone:
                seq.append(e.zone)
        return max(0, len(seq) - 1)
