from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Deque, Dict, Mapping, Optional


@dataclass(frozen=True)
class ModeHealthSnapshot:
    state: str
    mode: str
    ts_wall_s: Optional[float]
    transition_count_window: int
    churn_score: float
    transition_reason: str = ""


class ModeHealthFSM:
    """Small telemetry-side FSM for Behavior V4 mode stability health."""

    def __init__(
        self,
        *,
        churn_window_s: float = 12.0,
        transitioning_hold_s: float = 2.0,
        churn_threshold: int = 3,
    ) -> None:
        self._churn_window_s = max(2.0, float(churn_window_s))
        self._transitioning_hold_s = max(0.1, float(transitioning_hold_s))
        self._churn_threshold = max(2, int(churn_threshold))

        self._transition_ts: Deque[float] = collections.deque()
        self._last_transition_s: Optional[float] = None
        self._last_event_s: Optional[float] = None
        self._logical_ts = 0.0

        self._state_counts: Dict[str, int] = {}
        self._reason_counts: Dict[str, int] = {}
        self._mode_counts: Dict[str, int] = {}

        self._last_snapshot = ModeHealthSnapshot(
            state="unknown",
            mode="",
            ts_wall_s=None,
            transition_count_window=0,
            churn_score=0.0,
            transition_reason="",
        )

    @property
    def last_snapshot(self) -> ModeHealthSnapshot:
        return self._last_snapshot

    def state_counts(self) -> Dict[str, int]:
        return dict(self._state_counts)

    def transition_reason_counts(self) -> Dict[str, int]:
        return dict(self._reason_counts)

    def mode_counts(self) -> Dict[str, int]:
        return dict(self._mode_counts)

    def ingest(
        self,
        *,
        ts_wall_s: Optional[float],
        mode: str,
        transitioned: bool,
        transition_reason: str = "",
        planner_error: str = "",
        guard_fallback: bool = False,
    ) -> ModeHealthSnapshot:
        now_s = self._coerce_now(ts_wall_s)
        self._last_event_s = now_s

        mode_norm = str(mode or "").strip().lower()
        if mode_norm:
            self._mode_counts[mode_norm] = self._mode_counts.get(mode_norm, 0) + 1

        reason = str(transition_reason or "").strip().lower()
        if transitioned:
            self._last_transition_s = now_s
            self._transition_ts.append(now_s)
            if reason:
                self._reason_counts[reason] = self._reason_counts.get(reason, 0) + 1

        cutoff = now_s - self._churn_window_s
        while self._transition_ts and self._transition_ts[0] < cutoff:
            self._transition_ts.popleft()

        transition_count = len(self._transition_ts)
        churn_score = min(1.0, float(transition_count) / float(self._churn_threshold))

        state = self._classify_state(
            now_s=now_s,
            mode=mode_norm,
            transition_count=transition_count,
            planner_error=str(planner_error or "").strip(),
            guard_fallback=bool(guard_fallback),
        )

        self._state_counts[state] = self._state_counts.get(state, 0) + 1
        self._last_snapshot = ModeHealthSnapshot(
            state=state,
            mode=mode_norm,
            ts_wall_s=ts_wall_s,
            transition_count_window=transition_count,
            churn_score=round(churn_score, 3),
            transition_reason=reason,
        )
        return self._last_snapshot

    def _classify_state(
        self,
        *,
        now_s: float,
        mode: str,
        transition_count: int,
        planner_error: str,
        guard_fallback: bool,
    ) -> str:
        if mode == "boot_awaken":
            return "boot"
        if mode == "recover_reset":
            return "recovering"
        if transition_count >= self._churn_threshold:
            return "churn"
        if self._last_transition_s is not None and (now_s - self._last_transition_s) <= self._transitioning_hold_s:
            return "transitioning"
        if planner_error:
            return "planner_blocked"
        if guard_fallback:
            return "fallback_active"
        if mode:
            return "stable"
        return "unknown"

    def _coerce_now(self, ts_wall_s: Optional[float]) -> float:
        if isinstance(ts_wall_s, (int, float)):
            now_s = float(ts_wall_s)
            if self._last_event_s is not None and now_s < self._last_event_s:
                return self._last_event_s
            return now_s
        self._logical_ts += 0.001
        if self._last_event_s is None:
            return self._logical_ts
        return self._last_event_s + 0.001


def ingest_mode_event(
    fsm: ModeHealthFSM,
    event: Mapping[str, Any],
) -> ModeHealthSnapshot:
    return fsm.ingest(
        ts_wall_s=_as_optional_float(event.get("ts_wall_s")),
        mode=str(event.get("mode") or ""),
        transitioned=_as_boolish(event.get("mode_transitioned")),
        transition_reason=str(event.get("mode_transition_reason") or ""),
        planner_error=str(event.get("planner_last_error") or ""),
        guard_fallback=_as_boolish(event.get("guard_fallback")),
    )


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(text)
