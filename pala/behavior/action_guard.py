from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ..types import ActionPlan, action_plan_from_dict
from .decision_schema_v4 import BehaviorDecision
from .mode_fsm_v4 import MacroMode
from .skills_v4 import (
    allowed_moods_for_mode,
    allowed_primitives_for,
    allowed_skills_for_mode,
    default_action_payload_for_mode,
    default_skill_for_mode,
)


@dataclass
class ActionGuardConfig:
    min_action_dwell_s: float = 0.8
    stale_after_s: float = 6.0
    cooldowns_s: Dict[str, float] = field(
        default_factory=lambda: {
            "orient_to_zone": 1.2,
            "glance": 2.0,
            "nod": 4.0,
            "home": 4.0,
        }
    )


@dataclass(frozen=True)
class GuardContext:
    mode: MacroMode
    active_skill: str
    current_action: ActionPlan
    action_age_s: float
    model_age_s: float = 0.0
    health_degraded: bool = False
    breaker_open: bool = False


@dataclass(frozen=True)
class GuardResult:
    accepted: bool
    used_fallback: bool
    reason: str
    action: ActionPlan
    skill: str


class ActionGuard:
    """Small deterministic validator for model-selected actions."""

    def __init__(self, config: ActionGuardConfig | None = None) -> None:
        self._cfg = config or ActionGuardConfig()
        self._last_commit_by_primitive: Dict[str, float] = {}

    def evaluate(self, *, decision: BehaviorDecision, context: GuardContext, now_mono_s: float) -> GuardResult:
        if context.breaker_open or context.health_degraded:
            return self._fallback(mode=context.mode, reason="health_degraded")

        if context.model_age_s > max(0.2, float(self._cfg.stale_after_s)):
            return self._fallback(mode=context.mode, reason="stale_model_decision")

        if decision.mode != context.mode.value:
            return self._fallback(mode=context.mode, reason="mode_not_current")

        if context.action_age_s < max(0.0, float(self._cfg.min_action_dwell_s)):
            current_primitive = context.current_action.primitive.value
            if decision.action.primitive != current_primitive:
                return self._fallback(mode=context.mode, reason="min_action_dwell")

        allowed_moods = allowed_moods_for_mode(context.mode)
        if decision.mood not in allowed_moods:
            return self._fallback(mode=context.mode, reason="mood_not_allowed")

        allowed_skills = allowed_skills_for_mode(context.mode)
        if decision.skill not in allowed_skills:
            return self._fallback(mode=context.mode, reason="skill_not_allowed")

        allowed_primitives = allowed_primitives_for(context.mode, decision.skill)
        if decision.action.primitive not in allowed_primitives:
            return self._fallback(mode=context.mode, reason="primitive_not_allowed")

        last_commit = self._last_commit_by_primitive.get(decision.action.primitive)
        cooldown_s = max(0.0, float(self._cfg.cooldowns_s.get(decision.action.primitive, 0.0)))
        if last_commit is not None and (now_mono_s - last_commit) < cooldown_s:
            return self._fallback(mode=context.mode, reason="primitive_cooldown")

        payload = {
            "primitive": decision.action.primitive,
            "command": dict(decision.action.command),
            "style": decision.action.style,
            "confidence": float(decision.confidence),
            "cancel_current": False,
            "explanation": decision.rationale_short,
        }
        action = action_plan_from_dict(payload)
        if action is None:
            return self._fallback(mode=context.mode, reason="invalid_action_payload")
        action.cancel_current = False

        return GuardResult(
            accepted=True,
            used_fallback=False,
            reason="accepted",
            action=action,
            skill=decision.skill,
        )

    def mark_committed(self, *, action: ActionPlan, now_mono_s: float) -> None:
        self._last_commit_by_primitive[action.primitive.value] = now_mono_s

    def fallback_action(self, *, mode: MacroMode, reason: str) -> ActionPlan:
        return self._build_fallback_action(mode=mode, reason=reason)

    def _fallback(self, *, mode: MacroMode, reason: str) -> GuardResult:
        action = self._build_fallback_action(mode=mode, reason=reason)
        return GuardResult(
            accepted=False,
            used_fallback=True,
            reason=reason,
            action=action,
            skill=default_skill_for_mode(mode),
        )

    def _build_fallback_action(self, *, mode: MacroMode, reason: str) -> ActionPlan:
        payload = default_action_payload_for_mode(mode, reason=reason)
        action = action_plan_from_dict(payload)
        if action is not None:
            action.cancel_current = False
            return action
        # Last-resort guardrail fallback; this should never happen with fixed defaults.
        hold = action_plan_from_dict(
            {
                "primitive": "hold",
                "command": {},
                "style": "calm",
                "confidence": 0.1,
                "cancel_current": False,
                "explanation": f"fallback:{reason}",
            }
        )
        assert hold is not None
        hold.cancel_current = False
        return hold
