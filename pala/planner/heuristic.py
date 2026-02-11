from __future__ import annotations

from ..control.primitives import PRIMITIVE_BREATH, PRIMITIVE_HOLD
from ..types import ActionPlan, PerceptionState
from .protocol import PlannerInterface


class HeuristicPlanner(PlannerInterface):
    """Default local planner (no cloud calls)."""

    def plan(self, st: PerceptionState) -> ActionPlan:
        if st.primary_person is None:
            return ActionPlan(primitive=PRIMITIVE_HOLD, params={}, confidence=0.2)
        return ActionPlan(
            primitive=PRIMITIVE_BREATH,
            params={"amp_rad": 0.08, "period_s": 7.0, "rate_rad_s": 1.0},
            confidence=0.4,
            explanation="idle presence",
        )
