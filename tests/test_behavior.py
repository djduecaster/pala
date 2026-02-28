from __future__ import annotations

from dataclasses import dataclass

from pala.behavior import (
    BehaviorPolicy,
    BehaviorPolicyConfig,
    DecisionSnapshot,
    EnvironmentSnapshot,
    WorldStateStore,
    WorldStateStoreConfig,
    parse_env_summary_response,
    parse_intent_proposer_response,
)
from pala.types import PerceptionState


@dataclass
class _FakeClock:
    now_s: float = 0.0

    def __call__(self) -> float:
        return self.now_s

    def set(self, value: float) -> None:
        self.now_s = value


def test_world_state_store_persists_markdown(tmp_path):
    identity_path = tmp_path / "identity.md"
    world_path = tmp_path / "world_state.md"
    digest_path = tmp_path / "session_digest.md"
    identity_path.write_text("PALA identity test", encoding="utf-8")

    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(identity_path),
            world_state_path=str(world_path),
            session_digest_path=str(digest_path),
            max_events=3,
            max_decisions=2,
        )
    )

    store.update_environment(
        EnvironmentSnapshot(
            scene="desk with monitor and notebook",
            events="user sits down and starts typing",
            hypotheses="focused work session",
            summary="user transitioned into focused work",
            delta_score=0.7,
            features={"person_present": True, "zone_hint": "center", "activity_level": 0.5, "novelty": 0.6},
        )
    )
    store.append_event("user reached for notebook")
    store.append_decision(
        DecisionSnapshot(
            primitive="orient_to_zone",
            style="calm",
            confidence=0.72,
            rationale_short="keep user centered",
        )
    )
    store.rewrite_session_digest("Session stable: focused work with intermittent movement.")

    assert world_path.exists()
    world_md = world_path.read_text(encoding="utf-8")
    assert "PALA identity test" in world_md
    assert "orient_to_zone" in world_md
    assert "focused work" in world_md
    assert "Features" in world_md


def test_parse_env_summary_response_strict_json():
    raw = (
        '{"schema_version":"pala.env_summary.v1","scene":"Desk scene","events":"User leans forward",'
        '"hypotheses":"User preparing to type","summary_short":"User begins focused desk work",'
        '"delta_score":0.78,"features":{"person_present":true,"zone_hint":"left","activity_level":0.66,"novelty":0.71}}'
    )
    parsed = parse_env_summary_response(raw)
    assert parsed is not None
    assert parsed.summary.delta_score == 0.78
    assert parsed.summary.features["zone_hint"] == "left"


def test_parse_intent_proposer_response_strict_json():
    raw = (
        '{"schema_version":"pala.intent_proposals.v2","notes_short":"User moved left",'
        '"proposals":[{"intent":"track_user","primitive":"orient_to_zone",'
        '"command":{"zone":"left","amp_rad":0.22,"rate_rad_s":1.3},'
        '"style":"focused","score":0.74,"confidence":0.69,"urgency":0.5,'
        '"risk":"low","allow_interrupt":false,"evidence":["frame:latest"],'
        '"rationale_short":"user moved left, reorient gently"},'
        '{"intent":"idle_presence","primitive":"breath","command":{"amp_rad":0.06,"period_s":6.5,"rate_rad_s":1.0},'
        '"style":"calm","score":0.4,"confidence":0.6,"urgency":0.15,'
        '"risk":"low","allow_interrupt":true,"evidence":[],"rationale_short":"maintain subtle presence"},'
        '{"intent":"scan_environment","primitive":"glance","command":{"direction":"right","amp_rad":0.2,"duration_s":0.4,"rate_rad_s":1.3},'
        '"style":"curious","score":0.38,"confidence":0.55,"urgency":0.3,'
        '"risk":"low","allow_interrupt":true,"evidence":[],"rationale_short":"check opposite side"}]}'
    )
    parsed = parse_intent_proposer_response(raw)
    assert parsed is not None
    assert parsed.response.schema_version == "pala.intent_proposals.v2"
    assert len(parsed.response.proposals) == 3
    assert parsed.response.proposals[0].primitive == "orient_to_zone"


def test_parse_intent_proposer_response_rejects_non_schema_json():
    raw = (
        '{"schema_version":"pala.intent_proposals.v2","proposals":['
        '{"intent":"social desk companion","likelihood":0.9,"description":"Gently glance left toward the user to acknowledge presence."},'
        '{"intent":"idle presence","description":"Maintain calm breathing presence."}'
        ']}'
    )
    parsed = parse_intent_proposer_response(raw)
    assert parsed is None


def test_parse_env_summary_response_accepts_common_alias_keys():
    raw = (
        '{"schema_version":"pala.env_summary.v1",'
        '"description":"A desk scene with a person at left using a laptop.",'
        '"changes":"Person shifted posture and leaned forward.",'
        '"summary":"User appears focused on computer work."}'
    )
    parsed = parse_env_summary_response(raw)
    assert parsed is not None
    assert parsed.summary.scene.startswith("A desk scene")
    assert parsed.summary.events.startswith("Person shifted posture")


def test_behavior_policy_commits_only_on_actual_action_change(tmp_path):
    clock = _FakeClock()
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    cfg = BehaviorPolicyConfig(
        remote_enabled=False,
        idle_after_s=1.0,
        idle_glance_after_s=2.0,
        arbiter_base_margin=0.05,
        planner_max_proposals=3,
        env_log_path=None,
        planner_log_path=None,
        reasoning_log_path=None,
        trace_log_path=None,
    )
    policy = BehaviorPolicy(world_state=store, config=cfg, clock=clock)

    st = PerceptionState(timestamp_monotonic_s=0.0, debug={"zone_hint": "left"})

    # Startup tick keeps initial hold action; no decision commit yet.
    a0 = policy.step(st)
    assert a0.primitive.value == "hold"
    assert store.snapshot()["decision_tail"] == []

    # Advance enough to trigger idle engine commit.
    clock.set(2.5)
    a1 = policy.step(st)
    decisions_after_commit = store.snapshot()["decision_tail"]
    assert len(decisions_after_commit) == 1
    assert a1.primitive.value in {"breath", "glance", "orient_to_zone"}

    # Immediate next tick should usually keep current action; no extra commit.
    clock.set(2.6)
    policy.step(st)
    assert len(store.snapshot()["decision_tail"]) == 1


def test_behavior_policy_startup_awaken_sequence_then_handoff(tmp_path):
    clock = _FakeClock()
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    cfg = BehaviorPolicyConfig(
        remote_enabled=False,
        startup_wake_enabled=True,
        startup_wake_left_s=0.2,
        startup_wake_right_s=0.2,
        startup_wake_loop_s=0.2,
        startup_wake_settle_s=0.2,
        idle_after_s=0.0,
        idle_glance_after_s=2.0,
        arbiter_base_margin=0.02,
        env_log_path=None,
        planner_log_path=None,
        reasoning_log_path=None,
        trace_log_path=None,
    )
    policy = BehaviorPolicy(world_state=store, config=cfg, clock=clock)

    a0 = policy.step(st=None)
    assert a0.primitive.value == "move_to"
    assert a0.explanation == "startup_wake_left"
    assert store.snapshot()["decision_tail"] == []

    clock.set(0.25)
    a1 = policy.step(st=None)
    assert a1.primitive.value == "move_to"
    assert a1.explanation == "startup_wake_right"

    clock.set(0.50)
    a2 = policy.step(st=None)
    assert a2.primitive.value == "move_to"
    assert a2.explanation == "startup_wake_loop"

    clock.set(0.75)
    a3 = policy.step(st=None)
    assert a3.primitive.value == "move_to"
    assert a3.explanation == "startup_observe_settle"

    clock.set(1.10)
    a4 = policy.step(st=None)
    assert policy._startup_done is True  # noqa: SLF001
    assert a4.primitive.value in {"hold", "breath", "glance"}


def test_behavior_policy_startup_fast_exit_is_latched(tmp_path):
    clock = _FakeClock()
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    cfg = BehaviorPolicyConfig(
        remote_enabled=False,
        startup_wake_enabled=True,
        startup_wake_left_s=0.2,
        startup_wake_right_s=0.2,
        startup_wake_loop_s=0.2,
        startup_wake_settle_s=0.2,
        startup_person_conf_fast_exit=0.6,
        idle_after_s=0.0,
        idle_glance_after_s=2.0,
        env_log_path=None,
        planner_log_path=None,
        reasoning_log_path=None,
        trace_log_path=None,
    )
    policy = BehaviorPolicy(world_state=store, config=cfg, clock=clock)

    store.update_environment(
        EnvironmentSnapshot(
            scene="person appears",
            events="person entered frame",
            hypotheses="possible interaction",
            summary="person present",
            delta_score=0.7,
            features={"person_present": True, "zone_hint": "center", "activity_level": 0.4, "novelty": 0.6},
        )
    )
    action0 = policy.step(st=None)
    assert action0.explanation == "startup_observe_settle"

    store.update_environment(
        EnvironmentSnapshot(
            scene="person no longer visible",
            events="person moved out of frame",
            hypotheses="no immediate interaction",
            summary="person absent",
            delta_score=0.2,
            features={"person_present": False, "zone_hint": "unknown", "activity_level": 0.1, "novelty": 0.1},
        )
    )
    clock.set(0.1)
    action1 = policy.step(st=None)
    assert action1.explanation == "startup_observe_settle"
