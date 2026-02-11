from __future__ import annotations

from typing import Protocol

from ..types import ActionPlan, PerceptionState


class PlannerInterface(Protocol):
    def plan(self, st: PerceptionState) -> ActionPlan:
        ...
