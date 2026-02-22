from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..types import ActionPlan, action_plan_from_dict
from .types import IntentProposal


@dataclass
class CompileResult:
    action: Optional[ActionPlan]
    error: Optional[str]


class ActionCompiler:
    """Compile intent proposals into validated ActionPlan objects."""

    def compile(self, proposal: IntentProposal) -> CompileResult:
        payload = {
            "primitive": proposal.primitive,
            "command": dict(proposal.command),
            "confidence": float(proposal.confidence),
            "style": proposal.style,
            "cancel_current": False,
            "explanation": proposal.rationale_short,
        }
        action = action_plan_from_dict(payload)
        if action is None:
            return CompileResult(action=None, error="invalid_action_payload")
        action.cancel_current = False
        return CompileResult(action=action, error=None)
