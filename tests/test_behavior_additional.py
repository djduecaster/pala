from __future__ import annotations

from pala.behavior import (
    BehaviorPolicy,
    BehaviorPolicyConfig,
    CosmosEnvProcessor,
    CosmosPlannerClient,
    PlannerDecision,
    WorldStateStore,
    WorldStateStoreConfig,
    parse_env_processor_response,
    parse_planner_response,
)
from pala.control.primitives import PrimitiveKind
from pala.types import BBoxNorm, PerceptionState


def test_behavior_policy_persist_every_step_writes_memory_files(tmp_path):
    world_path = tmp_path / "world_state.md"
    digest_path = tmp_path / "session_digest.md"
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(world_path),
            session_digest_path=str(digest_path),
        )
    )
    policy = BehaviorPolicy(
        planner=None,
        world_state=store,
        config=BehaviorPolicyConfig(persist_every_step=True),
    )
    state = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.5, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.8,
        debug={"zone_hint": "center"},
    )

    out = policy.step(state)
    assert out.primitive == PrimitiveKind.HOLD
    assert world_path.exists()
    assert digest_path.exists()


def test_world_state_store_loads_existing_session_digest(tmp_path):
    digest_path = tmp_path / "session_digest.md"
    digest_path.write_text("prior digest", encoding="utf-8")
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(digest_path),
        )
    )
    assert store.session_digest == "prior digest"


def test_env_processor_complete_request_clears_inflight_and_parses():
    env = CosmosEnvProcessor()
    assert env.submit_or_replace({"tick": 1}) is True
    assert env.in_flight is True

    raw = (
        "<scene>desk</scene>"
        "<events>user moved</events>"
        "<hypotheses>engaged</hypotheses>"
        "<opportunities>light</opportunities>"
        "<uncertainties>none</uncertainties>"
        "<delta_score>0.7</delta_score>"
        "<summary>movement detected</summary>"
    )
    parsed = env.complete_request(raw)
    assert env.in_flight is False
    assert parsed is not None
    assert parsed.snapshot.delta_score == 0.7


def test_planner_client_complete_request_clears_inflight_and_parses():
    planner = CosmosPlannerClient()
    assert planner.submit_or_replace({"tick": 1}) is True
    assert planner.in_flight is True

    raw = (
        "<decision_json>{\"act_now\": true, \"primitive\": \"hold\", \"command\": {}, \"confidence\": 0.9}</decision_json>"
        "<rationale_short>stay still</rationale_short>"
    )
    decision = planner.complete_request(raw)
    assert planner.in_flight is False
    assert decision is not None
    assert decision.primitive == "hold"


def test_planner_parser_coerces_yes_and_clamps_negative_confidence():
    raw = (
        "<decision_json>"
        "{\"act_now\":\"yes\",\"primitive\":\"hold\",\"command\":{},\"style\":\"Calm\",\"confidence\":-2}"
        "</decision_json>"
        "<rationale_short>hold posture</rationale_short>"
    )
    decision = parse_planner_response(raw)
    assert decision is not None
    assert decision.act_now is True
    assert decision.confidence == 0.0
    assert decision.style == "calm"


def test_planner_parser_rejects_non_object_decision_json():
    raw = "<decision_json>[1,2,3]</decision_json><rationale_short>bad</rationale_short>"
    assert parse_planner_response(raw) is None


def test_planner_decision_dataclass_shape():
    decision = PlannerDecision(
        act_now=False,
        primitive=None,
        command={},
        style="calm",
        confidence=0.5,
        rationale_short="wait",
        reasoning_text=None,
        raw_text="x",
    )
    assert decision.act_now is False
    assert decision.command == {}


def test_env_parser_normalizes_multi_image_language_into_sequence_language():
    raw = (
        "<scene>Multiple images depict an office. In the first frame, the user sits. "
        "In the second frame, the user leans forward.</scene>"
        "<events>There are four frames depicting subtle posture changes.</events>"
        "<hypotheses>User remains focused on desk work.</hypotheses>"
    )
    parsed = parse_env_processor_response(raw)
    assert parsed is not None
    scene = parsed.snapshot.scene.lower()
    events = parsed.snapshot.events.lower()
    assert "multiple images" not in scene
    assert "first frame" not in scene
    assert "four frames depicting" not in events
    assert "sequence" in (scene + " " + events)


def test_planner_parser_accepts_fenced_python_literal_json_with_rationale():
    raw = (
        "```json {'act_now': True, 'primitive': 'orient_to_zone', 'command': {}, "
        "'style': 'calm', 'confidence': 0.9} ```"
        "<rationale_short>person shifts left</rationale_short>"
    )
    decision = parse_planner_response(raw)
    assert decision is not None
    assert decision.act_now is True
    assert decision.primitive == "orient_to_zone"
    assert decision.command.get("zone") == "left"


def test_planner_parser_accepts_single_quote_json_with_lowercase_booleans():
    raw = (
        "```json {'act_now': false, 'primitive': 'hold', 'command': {}, "
        "'style': 'calm', 'confidence': 0.4} ```"
        "<rationale_short>remain still</rationale_short>"
    )
    decision = parse_planner_response(raw)
    assert decision is not None
    assert decision.act_now is False
    assert decision.primitive == "hold"
