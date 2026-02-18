from __future__ import annotations

from collections import deque
import time
from typing import Optional

from .state_models import SceneSummary, SessionMemory


class SessionMemoryManager:
    """Deterministic short/session memory manager for planner arbitration."""

    def __init__(self, *, away_timeout_s: float = 3.0, recent_intents_max: int = 8) -> None:
        self._away_timeout_s = max(0.5, float(away_timeout_s))
        self._recent_intents: deque[str] = deque(maxlen=max(1, int(recent_intents_max)))
        now = time.monotonic()
        self._memory = SessionMemory(
            interaction_state="idle",
            task_hypothesis=None,
            last_transition_s=now,
            staleness_ms=0.0,
            recent_intents=[],
        )
        self._last_seen_person_s: Optional[float] = None

    def update(self, summary: SceneSummary) -> SessionMemory:
        now = time.monotonic()
        prev_state = self._memory.interaction_state
        state = prev_state

        if summary.person_present:
            self._last_seen_person_s = now
            if prev_state in {"idle", "searching"}:
                state = "engaged"
            elif summary.activity_hint == "transitioning":
                state = "transitioning"
            else:
                state = "engaged"
        else:
            last_seen = self._last_seen_person_s
            if last_seen is None or (now - last_seen) > self._away_timeout_s:
                state = "idle"
            else:
                state = "searching"

        task = _infer_task(summary, state)
        if state != prev_state:
            last_transition_s = now
        else:
            last_transition_s = self._memory.last_transition_s

        self._memory = SessionMemory(
            interaction_state=state,
            task_hypothesis=task,
            last_transition_s=last_transition_s,
            staleness_ms=max(0.0, (now - summary.timestamp_monotonic_s) * 1000.0),
            recent_intents=list(self._recent_intents),
        )
        return self._memory

    def note_intent(self, intent: str) -> None:
        token = str(intent).strip()
        if token:
            self._recent_intents.append(token)
            self._memory.recent_intents = list(self._recent_intents)

    @property
    def memory(self) -> SessionMemory:
        return self._memory


def _infer_task(summary: SceneSummary, interaction_state: str) -> Optional[str]:
    if interaction_state == "idle":
        return None
    if summary.activity_hint == "focused_work":
        return "focus_work"
    if summary.activity_hint == "transitioning":
        return "context_shift"
    if summary.person_present:
        return "engaged_presence"
    return "searching"
