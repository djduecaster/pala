from __future__ import annotations

from dataclasses import dataclass

from pala.behavior import (
    ActionTranslator,
    BehaviorPolicy,
    BehaviorPolicyConfig,
    CosmosEnvProcessor,
    CosmosPlannerClient,
    DecisionSnapshot,
    EnvironmentSnapshot,
    PlannerDecision,
    RemoteCallResult,
    RollingFrameWindow,
    WorldStateStore,
    WorldStateStoreConfig,
    parse_env_processor_response,
    parse_planner_response,
)
from pala.types import ActionPlan, BBoxNorm, PerceptionState
from pala.control.primitives import PrimitiveKind, OrientToZoneCommand
import numpy as np
import time


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
            opportunities="maintain key light on desk",
            uncertainties="no explicit user request yet",
            summary="user transitioned into focused work",
            delta_score=0.7,
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
    assert digest_path.exists()
    world_md = world_path.read_text(encoding="utf-8")
    assert "PALA identity test" in world_md
    assert "orient_to_zone" in world_md
    assert "focused work" in world_md
    assert "Control State" in world_md


def test_parse_env_processor_response_from_tagged_text():
    raw = (
        "<think>private reasoning</think>"
        "<scene>Desk scene with user and monitor.</scene>"
        "<events>User posture changed to forward lean.</events>"
        "<hypotheses>User is about to type.</hypotheses>"
        "<opportunities>Provide focused desk lighting.</opportunities>"
        "<uncertainties>No confirmed request yet.</uncertainties>"
        "<delta_score>0.82</delta_score>"
        "<summary>User engaged in work posture.</summary>"
    )
    parsed = parse_env_processor_response(raw)
    assert parsed is not None
    assert parsed.snapshot.delta_score == 0.82
    assert parsed.snapshot.summary == "User engaged in work posture."
    assert parsed.reasoning_text == "private reasoning"


def test_parse_planner_response_and_translate_to_action():
    raw = (
        "<think>private reasoning</think>"
        "<decision_json>"
        '{"act_now": true, "primitive": "orient_to_zone", "command": {"zone": "left"}, '
        '"style": "curious", "confidence": 0.88}'
        "</decision_json>"
        "<rationale_short>Track user shift to the left.</rationale_short>"
    )
    decision = parse_planner_response(raw)
    assert decision is not None
    translator = ActionTranslator()
    translated = translator.translate(decision)
    assert translated.error is None
    assert translated.action is not None
    assert translated.action.primitive == PrimitiveKind.ORIENT_TO_ZONE
    assert isinstance(translated.action.command, OrientToZoneCommand)
    assert translated.action.command.zone == "left"
    assert translated.action.cancel_current is False


def test_planner_parser_normalizes_orient_command_keys():
    raw = (
        "<decision_json>"
        '{"act_now": true, "primitive": "orient_to_zone", '
        '"command": {"target": "right", "rate": 0.6}, "style": "calm", "confidence": 0.8}'
        "</decision_json>"
        "<rationale_short>Track user rightward.</rationale_short>"
    )
    decision = parse_planner_response(raw)
    assert decision is not None
    assert decision.primitive == "orient_to_zone"
    assert decision.command.get("zone") == "right"
    assert decision.command.get("rate_rad_s") == 0.6


def test_action_translator_rejects_orient_without_zone():
    translator = ActionTranslator()
    decision = PlannerDecision(
        act_now=True,
        primitive="orient_to_zone",
        command={},
        style="calm",
        confidence=0.7,
        rationale_short="turn toward user",
        reasoning_text=None,
        raw_text="raw",
    )
    result = translator.translate(decision)
    assert result.action is None
    assert result.error == "missing zone"


@dataclass
class _Planner:
    def plan(self, _st: PerceptionState) -> ActionPlan:
        return ActionPlan(
            primitive=PrimitiveKind.ORIENT_TO_ZONE,
            command=OrientToZoneCommand(zone="right"),
            confidence=0.6,
            style="calm",
        )


def test_behavior_policy_planner_passthrough_and_memory(tmp_path):
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    policy = BehaviorPolicy(planner=_Planner(), world_state=store)
    st = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.8, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.91,
        debug={"zone_hint": "right"},
    )
    out = policy.step(st)
    assert out.primitive == PrimitiveKind.ORIENT_TO_ZONE

    snap = store.snapshot()
    assert snap["event_tail"] == []
    assert len(snap["decision_tail"]) >= 1


def test_latest_only_bookkeeping_helpers():
    env = CosmosEnvProcessor()
    planner = CosmosPlannerClient()

    assert env.submit_or_replace({"tick": 1}) is True
    assert env.submit_or_replace({"tick": 2}) is False
    assert env.take_latest_pending() == {"tick": 2}

    assert planner.submit_or_replace({"tick": 1}) is True
    assert planner.submit_or_replace({"tick": 2}) is False
    assert planner.take_latest_pending() == {"tick": 2}


def test_env_parser_accepts_partial_tags_with_core_fields():
    raw = (
        "<scene>Desk scene</scene>"
        "<events>User moved.</events>"
        "<hypotheses>User is engaged.</hypotheses>"
    )
    parsed = parse_env_processor_response(raw)
    assert parsed is not None
    assert parsed.snapshot.scene == "Desk scene"
    assert parsed.snapshot.events == "User moved."
    assert parsed.snapshot.hypotheses == "User is engaged."
    assert parsed.snapshot.summary == "User moved."


def test_env_parser_non_numeric_delta_score_falls_back_to_default():
    raw = (
        "<scene>Desk scene</scene>"
        "<events>User moved.</events>"
        "<hypotheses>User is engaged.</hypotheses>"
        "<opportunities>Maintain visibility.</opportunities>"
        "<uncertainties>No explicit request.</uncertainties>"
        "<delta_score>high</delta_score>"
        "<summary>user moved</summary>"
    )
    parsed = parse_env_processor_response(raw)
    assert parsed is not None
    assert 0.0 <= parsed.snapshot.delta_score <= 1.0


def test_env_parser_clamps_delta_score_to_unit_interval():
    raw = (
        "<scene>Desk scene</scene>"
        "<events>User moved.</events>"
        "<hypotheses>User is engaged.</hypotheses>"
        "<opportunities>Maintain visibility.</opportunities>"
        "<uncertainties>No explicit request.</uncertainties>"
        "<delta_score>3.2</delta_score>"
        "<summary>user moved</summary>"
    )
    parsed = parse_env_processor_response(raw)
    assert parsed is not None
    assert parsed.snapshot.delta_score == 1.0


def test_env_parser_accepts_json_fallback_with_answer_tag():
    raw = (
        "<think>reasoning</think>"
        "<answer>"
        '{"scene":"desk","events":"user reached for keyboard","hypotheses":"starting work",'
        '"opportunities":"aim light","uncertainties":"none","delta_score":0.73,"summary":"user engaged"}'
        "</answer>"
    )
    parsed = parse_env_processor_response(raw)
    assert parsed is not None
    assert parsed.snapshot.events == "user reached for keyboard"
    assert parsed.snapshot.delta_score == 0.73


def test_env_parser_accepts_malformed_closing_tag_via_fuzzy_extraction():
    raw = (
        "<scene>desk scene</scene>"
        "<events>user reached toward keyboard</events>"
        "<hypotheses>focused work</hypothese>"
        "<summary>user transition</summary>"
    )
    parsed = parse_env_processor_response(raw)
    assert parsed is not None
    assert parsed.snapshot.hypotheses == "focused work"
    assert parsed.snapshot.summary == "user transition"


def test_env_parser_rejects_template_echo_without_core_content():
    raw = (
        "<scene>...</scene>"
        "<events>...</events>"
        "<hypotheses>...</hypotheses>"
        "<summary>The PALA lamp adopts a home pose to maintain stability.</summary>"
    )
    parsed = parse_env_processor_response(raw)
    assert parsed is None


def test_planner_parser_fail_closed_for_missing_or_invalid_decision_block():
    missing_decision = "<rationale_short>ok</rationale_short>"
    invalid_json = (
        "<decision_json>{invalid}</decision_json>"
        "<rationale_short>ok</rationale_short>"
    )
    assert parse_planner_response(missing_decision) is None
    assert parse_planner_response(invalid_json) is None


def test_planner_parser_accepts_plain_json_fallback():
    raw = (
        '{"act_now":true,"primitive":"orient_to_zone","command":{"zone":"left"},'
        '"style":"curious","confidence":0.82,"rationale_short":"track user"}'
    )
    decision = parse_planner_response(raw)
    assert decision is not None
    assert decision.act_now is True
    assert decision.primitive == "orient_to_zone"
    assert decision.command["zone"] == "left"
    assert decision.rationale_short == "track user"


def test_planner_parser_accepts_nested_prediction_schema():
    raw = (
        '{"inference":"person is moving right","prediction":{"primitive":"orient_to_zone","target_zone":"right",'
        '"style":"focused","confidence":0.71}}'
    )
    decision = parse_planner_response(raw)
    assert decision is not None
    assert decision.primitive == "orient_to_zone"
    assert decision.command.get("zone") == "right"
    assert decision.style == "focused"
    assert decision.confidence == 0.71


def test_planner_parser_accepts_malformed_decision_closing_tag():
    raw = (
        '<decision_json>{"act_now":true,"primitive":"hold","command":{},"style":"calm","confidence":0.8}</decision_tail>'
        "<rationale_short>hold posture</rationale_short>"
    )
    decision = parse_planner_response(raw)
    assert decision is not None
    assert decision.act_now is True
    assert decision.primitive == "hold"
    assert decision.rationale_short == "hold posture"


def test_planner_parser_extracts_rationale_from_trailing_text_when_tag_missing():
    raw = (
        '<decision_json>{"act_now":true,"primitive":"home","command":{},"style":"calm","confidence":0.7}</decision_tail>'
        " Return to neutral after completing the prior glance."
    )
    decision = parse_planner_response(raw)
    assert decision is not None
    assert decision.primitive == "home"
    assert decision.rationale_short == "Return to neutral after completing the prior glance."


def test_planner_parser_deprioritizes_fence_heavy_low_energy_decision():
    raw = (
        "```json {\"act_now\":true,\"primitive\":\"breath\",\"command\":{},\"style\":\"calm\",\"confidence\":0.8} ```"
    )
    decision = parse_planner_response(raw)
    assert decision is not None
    assert decision.act_now is False
    assert decision.primitive is None
    assert decision.command == {}


def test_planner_parser_coercion_defaults_and_bounds():
    raw = (
        "<decision_json>"
        '{"act_now":"false","primitive":"orient_to_zone","command":"bad","style":" ","confidence":2.4}'
        "</decision_json>"
        "<rationale_short>keep steady</rationale_short>"
    )
    decision = parse_planner_response(raw)
    assert decision is not None
    assert decision.act_now is False
    assert decision.command == {}
    assert decision.style == "calm"
    assert decision.confidence == 1.0


def test_action_translator_fail_closed_for_missing_or_invalid_primitive_payload():
    translator = ActionTranslator()
    missing_primitive = PlannerDecision(
        act_now=True,
        primitive=None,
        command={},
        style="calm",
        confidence=0.5,
        rationale_short="missing primitive",
        reasoning_text=None,
        raw_text="raw",
    )
    missing_result = translator.translate(missing_primitive)
    assert missing_result.action is None
    assert missing_result.error == "missing primitive"

    invalid_payload = PlannerDecision(
        act_now=True,
        primitive="orient_to_zone",
        command={"zone": "desk"},
        style="calm",
        confidence=0.7,
        rationale_short="invalid zone",
        reasoning_text=None,
        raw_text="raw",
    )
    invalid_result = translator.translate(invalid_payload)
    assert invalid_result.action is None
    assert invalid_result.error == "missing zone"


def test_action_translator_returns_no_action_when_act_now_is_false():
    translator = ActionTranslator()
    deferred = PlannerDecision(
        act_now=False,
        primitive="orient_to_zone",
        command={"zone": "left"},
        style="calm",
        confidence=0.7,
        rationale_short="wait",
        reasoning_text=None,
        raw_text="raw",
    )
    result = translator.translate(deferred)
    assert result.action is None
    assert result.error is None


def test_behavior_policy_falls_back_to_hold_when_planner_unavailable_or_failing(tmp_path):
    class _BrokenPlanner:
        def plan(self, _st):
            raise RuntimeError("planner down")

    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    failing_policy = BehaviorPolicy(planner=_BrokenPlanner(), world_state=store)
    no_planner_policy = BehaviorPolicy(planner=None, world_state=store)

    st = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.5, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.4,
        debug={"zone_hint": "center"},
    )
    failed = failing_policy.step(st)
    missing = no_planner_policy.step(st)

    assert failed.primitive == PrimitiveKind.HOLD
    assert failed.explanation == "fallback hold"
    assert missing.primitive == PrimitiveKind.HOLD
    assert missing.explanation == "fallback hold"


def test_world_state_store_enforces_tail_bounds_and_control_state_format(tmp_path):
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
            max_events=2,
            max_decisions=1,
        )
    )
    store.append_event("event-one")
    store.append_event("  ")
    store.append_event("event-two")
    store.append_event("event-three")
    store.append_decision(
        DecisionSnapshot(
            primitive="hold",
            style="calm",
            confidence=0.2,
            rationale_short="first",
        )
    )
    store.append_decision(
        DecisionSnapshot(
            primitive="breath",
            style="curious",
            confidence=0.8,
            rationale_short="second",
        )
    )
    store.set_control_state({"active": "hold"})

    snap = store.snapshot()
    assert [row["summary"] for row in snap["event_tail"]] == ["event-two", "event-three"]
    assert [row["source"] for row in snap["event_tail"]] == ["manual", "manual"]
    assert len(snap["decision_tail"]) == 1
    assert snap["decision_tail"][0]["primitive"] == "breath"
    assert "active=hold" in snap["control_state_latest"]

    class _Control:
        active_kind = "hold"
        status = "running"
        reason = None
        started_monotonic_s = 1.23

    store.set_control_state(_Control())
    snap = store.snapshot()
    assert "active_kind=hold" in snap["control_state_latest"]
    assert "status=running" in snap["control_state_latest"]


def test_world_state_store_uses_default_identity_when_identity_file_missing(tmp_path):
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "missing_identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    assert "You are PALA" in store.identity_core


def test_rolling_frame_window_sampling_includes_newest():
    window = RollingFrameWindow(window_s=10.0)
    base_ns = time.monotonic_ns()
    for i in range(9):
        frame = np.full((8, 8, 3), i, dtype=np.uint8)
        window.add_frame(frame, mono_ns=base_ns + (i * 10_000_000))
    sampled = window.sample(max_frames=4)
    assert len(sampled) == 4
    assert int(sampled[-1].frame[0, 0, 0]) == 8


class _FakeSnapshot:
    def __init__(self, frame: np.ndarray, mono_ns: int):
        self.frame = frame
        self.mono_ns = mono_ns
        self.pts_ns = None


class _FakeFrameCache:
    def __init__(self, frame: np.ndarray):
        self._snap = _FakeSnapshot(frame=frame, mono_ns=time.monotonic_ns())

    def get(self, *, max_age_ms=None):  # noqa: ANN001
        _ = max_age_ms
        self._snap.mono_ns = time.monotonic_ns()
        return self._snap


def test_behavior_policy_remote_workers_update_world_state_and_action(monkeypatch, tmp_path):
    def _fake_post_chat_json(*, url, payload, timeout_s, api_key):
        _ = (url, timeout_s, api_key)
        user_content = payload["messages"][1]["content"]
        user_text = user_content[-1]["text"]
        if "environment processor" in user_text:
            content = (
                "<scene>desk with user</scene>"
                "<events>user shifted right</events>"
                "<hypotheses>user preparing to work</hypotheses>"
                "<opportunities>track user and light desk</opportunities>"
                "<uncertainties>none</uncertainties>"
                "<delta_score>0.88</delta_score>"
                "<summary>user moved to right workspace area</summary>"
            )
            return RemoteCallResult(
                ok=True,
                status_code=200,
                latency_ms=40.0,
                response_json={"choices": [{"message": {"content": content, "reasoning_content": "env-think"}}]},
                error=None,
            )
        planner_content = (
            "<decision_json>"
            '{"act_now":true,"primitive":"orient_to_zone","command":{"zone":"right"},"style":"curious","confidence":0.81}'
            "</decision_json>"
            "<rationale_short>track user on right side</rationale_short>"
        )
        return RemoteCallResult(
            ok=True,
            status_code=200,
            latency_ms=45.0,
            response_json={"choices": [{"message": {"content": planner_content, "reasoning_content": "plan-think"}}]},
            error=None,
        )

    monkeypatch.setattr("pala.behavior.policy.post_chat_json", _fake_post_chat_json)

    frame_cache = _FakeFrameCache(np.zeros((16, 16, 3), dtype=np.uint8))
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    cfg = BehaviorPolicyConfig(
        remote_enabled=True,
        base_url="http://127.0.0.1:8000",
        env_hz=50.0,
        planner_hz=50.0,
        request_timeout_ms=500,
        request_min_fresh_frames=1,
        env_log_path=str(tmp_path / "env.jsonl"),
        planner_log_path=str(tmp_path / "planner.jsonl"),
        reasoning_log_path=str(tmp_path / "reasoning.jsonl"),
    )
    policy = BehaviorPolicy(planner=None, world_state=store, config=cfg, frame_cache=frame_cache)
    st = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.8, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.9,
        debug={"zone_hint": "right"},
    )
    try:
        out = policy.step(st)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            time.sleep(0.01)
            out = policy.step(st)
            if out.primitive == PrimitiveKind.ORIENT_TO_ZONE:
                break
        assert out.primitive == PrimitiveKind.ORIENT_TO_ZONE
        assert isinstance(out.command, OrientToZoneCommand)
        assert out.command.zone == "right"
        snap = store.snapshot()
        assert snap["latest_env_snapshot"] is not None
        assert len(snap["decision_tail"]) >= 1
    finally:
        policy.shutdown()


def test_world_state_store_clamps_non_positive_tail_limits(tmp_path):
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
            max_events=0,
            max_decisions=-4,
        )
    )
    store.append_event("one")
    store.append_event("two")
    store.append_decision(DecisionSnapshot(primitive="hold", style="calm", confidence=0.3, rationale_short="one"))
    store.append_decision(DecisionSnapshot(primitive="breath", style="calm", confidence=0.6, rationale_short="two"))
    snap = store.snapshot()
    assert len(snap["event_tail"]) == 1
    assert snap["event_tail"][0]["summary"] == "two"
    assert snap["event_tail"][0]["source"] == "manual"
    assert len(snap["decision_tail"]) == 1
    assert snap["decision_tail"][0]["primitive"] == "breath"


def test_world_state_store_records_env_processor_dense_event_tail(tmp_path):
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    store.update_environment(
        EnvironmentSnapshot(
            scene="desk with monitor and mug",
            events="person leans in and reaches toward keyboard",
            hypotheses="focused work is starting",
            opportunities="light keyboard area",
            uncertainties="no direct request yet",
            summary="engagement increased near keyboard",
            delta_score=0.8,
        )
    )
    snap = store.snapshot()
    assert len(snap["event_tail"]) == 1
    event = snap["event_tail"][0]
    assert event["source"] == "env_processor"
    assert "events=person leans in and reaches toward keyboard" in event["summary"]
    assert "summary=engagement increased near keyboard" in event["summary"]


def test_world_state_store_markdown_uses_second_precision_timestamps(tmp_path):
    world_path = tmp_path / "world_state.md"
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(world_path),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    store.append_event("event-one")
    md = world_path.read_text(encoding="utf-8")
    assert "." not in md.split("Last updated: ", 1)[1].splitlines()[0]
