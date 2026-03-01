from __future__ import annotations

from pala.behavior.action_guard import ActionGuard, GuardContext
from pala.behavior.decision_schema_v4 import BehaviorActionDecision, BehaviorDecision
from pala.behavior.mode_fsm_v4 import MacroMode
from pala.types import action_plan_from_dict


def _current_action():
    action = action_plan_from_dict(
        {
            "primitive": "hold",
            "command": {},
            "style": "calm",
            "confidence": 0.5,
            "cancel_current": False,
        }
    )
    assert action is not None
    return action


def _decision(*, primitive: str, skill: str = "greet_user", mode: str = "social_interact", confidence: float = 0.8):
    return BehaviorDecision(
        schema_version="pala.behavior_decision.v1",
        mode=mode,
        mood="curious",
        skill=skill,
        action=BehaviorActionDecision(
            primitive=primitive,
            command={"zone": "center"} if primitive == "orient_to_zone" else {},
            style="curious",
        ),
        confidence=confidence,
        rationale_short="test decision",
        mode_transition="stay",
        alternatives=[],
    )


def test_action_guard_accepts_valid_social_action():
    guard = ActionGuard()
    ctx = GuardContext(
        mode=MacroMode.SOCIAL_INTERACT,
        active_skill="greet_user",
        current_action=_current_action(),
        action_age_s=2.0,
        model_age_s=0.1,
    )
    result = guard.evaluate(decision=_decision(primitive="orient_to_zone"), context=ctx, now_mono_s=10.0)
    assert result.accepted is True
    assert result.used_fallback is False
    assert result.action.primitive.value == "orient_to_zone"


def test_action_guard_fallback_when_primitive_not_allowed():
    guard = ActionGuard()
    ctx = GuardContext(
        mode=MacroMode.SOCIAL_INTERACT,
        active_skill="greet_user",
        current_action=_current_action(),
        action_age_s=2.0,
        model_age_s=0.1,
    )
    result = guard.evaluate(decision=_decision(primitive="home"), context=ctx, now_mono_s=20.0)
    assert result.accepted is False
    assert result.used_fallback is True
    assert result.reason == "primitive_not_allowed"
    assert result.action.primitive.value in {"orient_to_zone", "breath", "hold"}


def test_action_guard_enforces_cooldown_after_commit():
    guard = ActionGuard()
    ctx = GuardContext(
        mode=MacroMode.SOCIAL_INTERACT,
        active_skill="greet_user",
        current_action=_current_action(),
        action_age_s=2.0,
        model_age_s=0.1,
    )
    first = guard.evaluate(decision=_decision(primitive="orient_to_zone"), context=ctx, now_mono_s=30.0)
    assert first.accepted is True
    guard.mark_committed(action=first.action, now_mono_s=30.0)

    second = guard.evaluate(decision=_decision(primitive="orient_to_zone"), context=ctx, now_mono_s=30.4)
    assert second.accepted is False
    assert second.reason == "primitive_cooldown"
    assert second.used_fallback is True


def test_action_guard_fallback_when_decision_is_stale():
    guard = ActionGuard()
    ctx = GuardContext(
        mode=MacroMode.IDLE_PRESENCE,
        active_skill="social_ack",
        current_action=_current_action(),
        action_age_s=5.0,
        model_age_s=9.0,
    )
    result = guard.evaluate(decision=_decision(primitive="glance", skill="social_ack", mode="idle_presence"), context=ctx, now_mono_s=40.0)
    assert result.accepted is False
    assert result.reason == "stale_model_decision"
    assert result.used_fallback is True


def test_action_guard_rejects_mode_mismatch():
    guard = ActionGuard()
    ctx = GuardContext(
        mode=MacroMode.SOCIAL_INTERACT,
        active_skill="greet_user",
        current_action=_current_action(),
        action_age_s=2.0,
        model_age_s=0.1,
    )
    result = guard.evaluate(
        decision=_decision(primitive="orient_to_zone", mode="idle_presence"),
        context=ctx,
        now_mono_s=50.0,
    )
    assert result.accepted is False
    assert result.reason == "mode_not_current"
    assert result.used_fallback is True


def test_action_guard_rejects_mood_not_allowed_for_mode():
    guard = ActionGuard()
    ctx = GuardContext(
        mode=MacroMode.SEARCH_ASSIST,
        active_skill="expressive_search",
        current_action=_current_action(),
        action_age_s=2.0,
        model_age_s=0.1,
    )
    decision = _decision(primitive="orient_to_zone", skill="expressive_search", mode="search_assist")
    decision = BehaviorDecision(
        schema_version=decision.schema_version,
        mode=decision.mode,
        mood="excited",
        skill=decision.skill,
        action=decision.action,
        confidence=decision.confidence,
        rationale_short=decision.rationale_short,
        mode_transition=decision.mode_transition,
        alternatives=decision.alternatives,
    )
    result = guard.evaluate(decision=decision, context=ctx, now_mono_s=60.0)
    assert result.accepted is False
    assert result.reason == "mood_not_allowed"
    assert result.used_fallback is True
