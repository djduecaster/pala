from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Optional

from ..perception.frame_cache import LatestFrameCache
from ..types import PerceptionState
from .state_models import SceneSummary


@dataclass
class _SummarizerRequest:
    state: PerceptionState
    frame_age_ms: Optional[float]


class AsyncSceneSummarizer:
    """Background scene summarizer with latest-only request semantics."""

    def __init__(
        self,
        *,
        frame_cache: LatestFrameCache,
        max_hz: float = 2.0,
        response_ttl_ms: int = 1200,
    ) -> None:
        self._frame_cache = frame_cache
        self._period_s = 1.0 / max(0.2, float(max_hz))
        self._ttl_s = max(0.1, float(response_ttl_ms) / 1000.0)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._pending: Optional[_SummarizerRequest] = None
        self._last_submit_s = 0.0

        self._latest: Optional[SceneSummary] = None
        self._latest_ts_s: Optional[float] = None
        self._zone_history: deque[str] = deque(maxlen=6)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, st: PerceptionState) -> None:
        now = time.monotonic()
        if now - self._last_submit_s < self._period_s:
            return
        snap = self._frame_cache.get(max_age_ms=1000)
        frame_age_ms = None if snap is None else (time.monotonic_ns() - snap.mono_ns) / 1_000_000.0
        req = _SummarizerRequest(state=st, frame_age_ms=frame_age_ms)
        with self._lock:
            self._pending = req
            self._last_submit_s = now
            self._cond.notify_all()

    def latest(self) -> Optional[SceneSummary]:
        with self._lock:
            summary = self._latest
            ts = self._latest_ts_s
        if summary is None or ts is None:
            return None
        if (time.monotonic() - ts) > self._ttl_s:
            return None
        return summary

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                while not self._stop.is_set() and self._pending is None:
                    self._cond.wait(timeout=0.1)
                if self._stop.is_set():
                    break
                req = self._pending
                self._pending = None
            if req is None:
                continue
            summary = self._summarize(req)
            with self._lock:
                self._latest = summary
                self._latest_ts_s = time.monotonic()

    def _summarize(self, req: _SummarizerRequest) -> SceneSummary:
        st = req.state
        zone = None
        if isinstance(st.debug, dict):
            raw_zone = st.debug.get("zone_hint")
            if isinstance(raw_zone, str) and raw_zone:
                zone = raw_zone
        if zone is not None:
            self._zone_history.append(zone)

        person_present = st.primary_person is not None
        uncertainty_flags: list[str] = []
        conf = st.primary_person_conf
        if person_present and conf is not None and conf < 0.5:
            uncertainty_flags.append("low_person_conf")
        if req.frame_age_ms is not None and req.frame_age_ms > 250.0:
            uncertainty_flags.append("stale_frame")

        activity_hint = self._infer_activity_hint(person_present, zone)
        return SceneSummary(
            timestamp_monotonic_s=time.monotonic(),
            person_present=person_present,
            zone_hint=zone,
            primary_person_conf=conf,
            activity_hint=activity_hint,
            uncertainty_flags=uncertainty_flags,
            frame_age_ms=req.frame_age_ms,
        )

    def _infer_activity_hint(self, person_present: bool, zone: Optional[str]) -> Optional[str]:
        if not person_present:
            return "away"
        if len(self._zone_history) < 3:
            return "engaged"
        recent = list(self._zone_history)[-3:]
        if len(set(recent)) >= 2:
            return "transitioning"
        if zone == "center":
            return "focused_work"
        return "engaged"
