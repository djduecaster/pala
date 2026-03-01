from __future__ import annotations

import json

from pala.behavior.policy_v4 import BehaviorPolicyV4, BehaviorPolicyV4Config
from pala.types import BBoxNorm, PerceptionState


def test_policy_v4_step_exits_boot_and_sets_idle_fallback():
    now = {"t": 0.0}

    def clock():
        return now["t"]

    policy = BehaviorPolicyV4(
        config=BehaviorPolicyV4Config(startup_wake_enabled=False, startup_min_s=0.5),
        clock=clock,
    )
    action0 = policy.step(None)
    assert action0.primitive.value in {"breath", "hold", "glance"}

    now["t"] = 0.8
    action1 = policy.step(None)
    assert policy.current_mode.value == "idle_presence"
    assert action1.primitive.value in {"breath", "hold", "glance"}


def test_policy_v4_apply_model_output_commits_when_valid():
    now = {"t": 0.0}

    def clock():
        return now["t"]

    policy = BehaviorPolicyV4(
        config=BehaviorPolicyV4Config(startup_min_s=0.1),
        clock=clock,
    )
    now["t"] = 0.2
    _ = policy.step(
        PerceptionState(
            timestamp_monotonic_s=now["t"],
            primary_person=BBoxNorm(cx=0.5, cy=0.5, w=0.3, h=0.6),
            primary_person_conf=0.9,
        )
    )
    now["t"] = 1.7
    _ = policy.step(
        PerceptionState(
            timestamp_monotonic_s=now["t"],
            primary_person=BBoxNorm(cx=0.5, cy=0.5, w=0.3, h=0.6),
            primary_person_conf=0.9,
        )
    )

    raw = json.dumps(
        {
            "schema_version": "pala.behavior_decision.v1",
            "mode": "social_interact",
            "mood": "curious",
            "skill": "greet_user",
            "action": {
                "primitive": "orient_to_zone",
                "command": {"zone": "center", "amp_rad": 0.2, "rate_rad_s": 1.2},
                "style": "curious",
            },
            "confidence": 0.8,
            "rationale_short": "User is centered; orient toward them.",
            "mode_transition": "to_social_interact",
        }
    )
    result = policy.apply_model_output(raw, model_age_s=0.1)
    assert result.accepted is True
    assert policy.current_action.primitive.value == "orient_to_zone"


def test_policy_v4_boot_sequence_emits_wake_actions():
    now = {"t": 0.0}

    def clock():
        return now["t"]

    policy = BehaviorPolicyV4(
        config=BehaviorPolicyV4Config(
            startup_wake_enabled=True,
            startup_wake_left_s=0.2,
            startup_wake_right_s=0.2,
            startup_wake_loop_s=0.2,
            startup_wake_settle_s=0.2,
            startup_min_s=0.8,
        ),
        clock=clock,
    )

    a0 = policy.step(None)
    assert a0.explanation == "startup_wake_left"
    assert a0.primitive.value == "orient_to_zone"

    now["t"] = 0.25
    a1 = policy.step(None)
    assert a1.explanation == "startup_wake_right"
    assert a1.primitive.value == "orient_to_zone"

    now["t"] = 0.5
    a2 = policy.step(None)
    assert a2.explanation == "startup_wake_loop"
    assert a2.primitive.value == "nod"

    now["t"] = 0.75
    a3 = policy.step(None)
    assert a3.explanation == "startup_observe_settle"
    assert a3.primitive.value == "breath"


def test_policy_v4_honors_model_mode_transition():
    now = {"t": 0.0}

    def clock():
        return now["t"]

    policy = BehaviorPolicyV4(
        config=BehaviorPolicyV4Config(startup_wake_enabled=False, startup_min_s=0.0),
        clock=clock,
    )
    _ = policy.step(None)
    assert policy.current_mode.value == "idle_presence"

    now["t"] = 2.0
    raw = json.dumps(
        {
            "schema_version": "pala.behavior_decision.v1",
            "mode": "search_assist",
            "mood": "focused",
            "skill": "expressive_search",
            "action": {
                "primitive": "orient_to_zone",
                "command": {"zone": "left", "amp_rad": 0.2, "rate_rad_s": 1.1},
                "style": "focused",
            },
            "confidence": 0.8,
            "rationale_short": "Switch to search and scan left first.",
            "mode_transition": "to_search_assist",
        }
    )
    result = policy.apply_model_output(raw, model_age_s=0.1)
    assert result.accepted is True
    assert policy.current_mode.value == "search_assist"


def test_policy_v4_uses_structured_signals_not_local_person_fallback():
    now = {"t": 0.0}

    def clock():
        return now["t"]

    policy = BehaviorPolicyV4(
        config=BehaviorPolicyV4Config(startup_wake_enabled=False, startup_min_s=0.0),
        clock=clock,
    )
    _ = policy.step(None)
    assert policy.current_mode.value == "idle_presence"

    now["t"] = 2.0
    st = PerceptionState(
        timestamp_monotonic_s=now["t"],
        primary_person=BBoxNorm(cx=0.5, cy=0.5, w=0.3, h=0.6),
        primary_person_conf=0.99,
        debug={},
    )
    _ = policy.step(st)
    assert policy.current_mode.value == "idle_presence"


def test_policy_v4_mode_lifecycle_search_social_home_idle():
    now = {"t": 0.0}

    def clock():
        return now["t"]

    policy = BehaviorPolicyV4(
        config=BehaviorPolicyV4Config(startup_wake_enabled=False, startup_min_s=0.0),
        clock=clock,
    )

    _ = policy.step(
        PerceptionState(timestamp_monotonic_s=now["t"], debug={"person_present": False, "person_conf": 0.0})
    )
    assert policy.current_mode.value == "idle_presence"

    now["t"] = 2.0
    _ = policy.step(
        PerceptionState(timestamp_monotonic_s=now["t"], debug={"person_present": True, "person_conf": 0.9})
    )
    assert policy.current_mode.value == "social_interact"

    now["t"] = 4.0
    _ = policy.step(
        PerceptionState(
            timestamp_monotonic_s=now["t"],
            debug={"person_present": True, "person_conf": 0.9, "search_requested": True},
        )
    )
    assert policy.current_mode.value == "search_assist"

    now["t"] = 6.0
    _ = policy.step(
        PerceptionState(
            timestamp_monotonic_s=now["t"],
            debug={"person_present": True, "person_conf": 0.8, "search_complete": True, "user_ack": True},
        )
    )
    assert policy.current_mode.value == "social_interact"

    now["t"] = 8.0
    _ = policy.step(
        PerceptionState(
            timestamp_monotonic_s=now["t"],
            debug={"person_present": True, "person_conf": 0.8, "home_requested": True},
        )
    )
    assert policy.current_mode.value == "return_home"

    now["t"] = 10.0
    _ = policy.step(
        PerceptionState(
            timestamp_monotonic_s=now["t"],
            debug={"person_present": False, "person_conf": 0.0, "home_completed": True},
        )
    )
    assert policy.current_mode.value == "idle_presence"


def test_policy_v4_skill_timeout_forces_timeout_fallback_mode():
    now = {"t": 0.0}

    def clock():
        return now["t"]

    policy = BehaviorPolicyV4(
        config=BehaviorPolicyV4Config(startup_wake_enabled=False, startup_min_s=0.0),
        clock=clock,
    )
    _ = policy.step(None)
    now["t"] = 2.0
    raw = json.dumps(
        {
            "schema_version": "pala.behavior_decision.v1",
            "mode": "search_assist",
            "mood": "focused",
            "skill": "expressive_search",
            "action": {
                "primitive": "orient_to_zone",
                "command": {"zone": "left", "amp_rad": 0.2, "rate_rad_s": 1.1},
                "style": "focused",
            },
            "confidence": 0.8,
            "rationale_short": "Begin search sweep on left zone.",
            "mode_transition": "to_search_assist",
        }
    )
    result = policy.apply_model_output(raw, model_age_s=0.1)
    assert result.accepted is True
    assert policy.current_mode.value == "search_assist"
    assert result.skill == "expressive_search"

    now["t"] = 19.5
    _ = policy.step(
        PerceptionState(
            timestamp_monotonic_s=now["t"],
            debug={"search_requested": True, "person_present": True, "person_conf": 0.7},
        )
    )
    assert policy.current_mode.value == "return_home"
    assert policy.current_action.explanation == "skill_timeout:expressive_search"
