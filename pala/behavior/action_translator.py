from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..types import ActionPlan, action_plan_from_dict
from .planner_client import PlannerDecision


@dataclass
class ActionTranslationResult:
    action: Optional[ActionPlan]
    error: Optional[str] = None


class ActionTranslator:
    """Translate strict planner decision payloads into typed ActionPlan objects."""

    def translate(self, decision: PlannerDecision) -> ActionTranslationResult:
        if not decision.act_now:
            return ActionTranslationResult(action=None)
        if not decision.primitive:
            return ActionTranslationResult(action=None, error="missing primitive")
        if decision.primitive == "orient_to_zone":
            zone = decision.command.get("zone")
            if not isinstance(zone, str) or zone.strip().lower() not in {"left", "center", "right"}:
                return ActionTranslationResult(action=None, error="missing zone")

        payload = {
            "primitive": decision.primitive,
            "command": dict(decision.command),
            "confidence": decision.confidence,
            "style": decision.style,
            "explanation": decision.rationale_short,
            # Phase-1 policy: no planner-driven cancellation semantics.
            "cancel_current": False,
        }
        action = action_plan_from_dict(payload)
        if action is None:
            return ActionTranslationResult(action=None, error="invalid action payload")
        if not action.explanation and decision.rationale_short:
            action.explanation = decision.rationale_short
        action.cancel_current = False
        return ActionTranslationResult(action=action)
