from __future__ import annotations

import time
from typing import Optional

from ..types import PerceptionState, ActionPlan
from ..planner import PlannerInterface
from ..control.primitives import (
    PrimitiveKind,
    HoldCommand,
    GlanceCommand,
    NodCommand,
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
        self._active_signature: Optional[str] = None
        self._planner_owns_semantics = bool(getattr(planner, "owns_semantic_behavior", False))

    def step(self, st: Optional[PerceptionState]) -> ActionPlan:
        now = time.monotonic()
        if st is None:
            proposed = ActionPlan(
                primitive=PrimitiveKind.HOLD,
                command=HoldCommand(),
                confidence=0.3,
                style="calm",
            )
            return self._arbitrate(proposed)

        # In orchestrator-owned mode, behavior policy avoids additional semantic triggers.
        if self._planner_owns_semantics:
            proposed = self.planner.plan(st)
            return self._arbitrate(proposed)

        if st.primary_person is None:
            proposed = ActionPlan(
                primitive=PrimitiveKind.HOLD,
                command=HoldCommand(),
                confidence=0.3,
                style="calm",
            )
            return self._arbitrate(proposed)

        zone = _zone_from_cx(st.primary_person.cx)
        if zone != self._curr_zone:
            self._curr_zone = zone
            self._zone_start = now

        dwell = now - self._zone_start
        if dwell >= self.dwell_s and (now - self._last_trigger) >= self.cooldown_s:
            self._last_trigger = now
            if zone == "left":
                proposed = ActionPlan(
                    primitive=PrimitiveKind.GLANCE,
                    command=GlanceCommand(direction="left", duration_s=0.6),
                    confidence=0.8,
                    explanation="user dwell left",
                    style="curious",
                    cancel_current=True,
                )
                return self._arbitrate(proposed)
            if zone == "right":
                proposed = ActionPlan(
                    primitive=PrimitiveKind.GLANCE,
                    command=GlanceCommand(direction="right", duration_s=0.6),
                    confidence=0.8,
                    explanation="user dwell right",
                    style="curious",
                    cancel_current=True,
                )
                return self._arbitrate(proposed)
            proposed = ActionPlan(
                primitive=PrimitiveKind.NOD,
                command=NodCommand(duration_s=0.4, amp_rad=0.2),
                confidence=0.7,
                explanation="user dwell center",
                style="focused",
                cancel_current=True,
            )
            return self._arbitrate(proposed)

        # Default: consult planner (heuristic) for small idle actions
        proposed = self.planner.plan(st)
        return self._arbitrate(proposed)

    def _arbitrate(self, proposed: ActionPlan) -> ActionPlan:
        sig = self._signature(proposed)
        if self._active_signature == sig:
            return ActionPlan(
                primitive=proposed.primitive,
                command=proposed.command,
                confidence=proposed.confidence,
                explanation=proposed.explanation,
                style=proposed.style,
                action_id=proposed.action_id,
                cancel_current=False,
            )

        self._active_signature = sig
        return ActionPlan(
            primitive=proposed.primitive,
            command=proposed.command,
            confidence=proposed.confidence,
            explanation=proposed.explanation,
            style=proposed.style,
            action_id=proposed.action_id,
            cancel_current=bool(proposed.cancel_current),
        )

    @staticmethod
    def _signature(action: ActionPlan) -> str:
        return f"{action.primitive.value}:{action.style}:{action.command!r}"


def _zone_from_cx(cx: float) -> str:
    if cx < 0.33:
        return "left"
    if cx < 0.66:
        return "center"
    return "right"
