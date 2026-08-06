from __future__ import annotations

from typing import Optional

from ..types import ActionPlan, HoldCommand, PerceptionState, PrimitiveKind


class HoldBehaviorPolicy:
    """Blank-slate behavior boundary used until the new agent is designed."""

    def __init__(self) -> None:
        self._action = ActionPlan(
            primitive=PrimitiveKind.HOLD,
            command=HoldCommand(),
            confidence=1.0,
            explanation="behavior_reset_hold",
            style="calm",
        )

    def step(self, _state: Optional[PerceptionState]) -> ActionPlan:
        return self._action

    def shutdown(self) -> None:
        return None
