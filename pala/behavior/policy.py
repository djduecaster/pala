from __future__ import annotations

import time
from typing import Optional

from ..types import PerceptionState, ActionPlan
from ..planner import PlannerInterface
from ..control.primitives import (
    PRIMITIVE_HOLD,
    PRIMITIVE_GLANCE_LEFT,
    PRIMITIVE_GLANCE_RIGHT,
    PRIMITIVE_ACKNOWLEDGE,
)


class BehaviorPolicy:
    """Simple dwell-based zone policy."""

    def __init__(self, planner: PlannerInterface, dwell_s: float = 2.0, cooldown_s: float = 1.0):
        self.planner = planner
        self.dwell_s = float(dwell_s)
        self.cooldown_s = float(cooldown_s)
        self._curr_zone: Optional[str] = None
        self._zone_start = time.monotonic()
        self._last_trigger = 0.0

    def step(self, st: Optional[PerceptionState]) -> ActionPlan:
        now = time.monotonic()
        if st is None or st.primary_person is None:
            return ActionPlan(primitive=PRIMITIVE_HOLD, params={}, confidence=0.3)

        zone = _zone_from_cx(st.primary_person.cx)
        if zone != self._curr_zone:
            self._curr_zone = zone
            self._zone_start = now

        dwell = now - self._zone_start
        if dwell >= self.dwell_s and (now - self._last_trigger) >= self.cooldown_s:
            self._last_trigger = now
            if zone == "left":
                return ActionPlan(
                    primitive=PRIMITIVE_GLANCE_LEFT,
                    params={"duration_s": 0.6},
                    confidence=0.8,
                    explanation="user dwell left"
                )
            if zone == "right":
                return ActionPlan(
                    primitive=PRIMITIVE_GLANCE_RIGHT,
                    params={"duration_s": 0.6},
                    confidence=0.8,
                    explanation="user dwell right"
                )
            return ActionPlan(
                primitive=PRIMITIVE_ACKNOWLEDGE,
                params={"duration_s": 0.4, "amp_rad": 0.2},
                confidence=0.7,
                explanation="user dwell center"
            )

        # Default: consult planner (heuristic) for small idle actions
        return self.planner.plan(st)


def _zone_from_cx(cx: float) -> str:
    if cx < 0.33:
        return "left"
    if cx < 0.66:
        return "center"
    return "right"
