from __future__ import annotations

from typing import Any, Optional

from ..types import ActionPlan, PerceptionState
from .heuristic import HeuristicPlanner


class AsyncOrchestratorPlanner:
    """
    Compatibility wrapper for prior orchestrator entry point.

    Current behavior falls back to local heuristic planning until the
    remote planner pipeline is wired in.
    """

    owns_semantic_behavior = True

    def __init__(self, *, fallback: Optional[HeuristicPlanner] = None, **_: Any):
        self._fallback = fallback or HeuristicPlanner()

    def plan(self, st: PerceptionState | None) -> ActionPlan:
        return self._fallback.plan(st)

    def shutdown(self) -> None:
        return None
