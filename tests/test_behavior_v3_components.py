from __future__ import annotations

import json

from pala.behavior import (
    ActionCompiler,
    Arbiter,
    ArbiterConfig,
    ComponentHealth,
    ContextBuilder,
    EnvSummarizer,
    Governor,
    GovernorConfig,
    HealthManager,
    IntentProposal,
    IntentProposer,
    ProposalCandidate,
    ProposerResponse,
    TraceBus,
    parse_env_summary_response,
    parse_intent_proposer_response,
)
from pala.behavior.decision_types import BehaviorMode
from pala.behavior.json_parse import parse_json_flexible
from pala.behavior.schemas import env_response_format, intent_response_format
from pala.behavior.types import clamp_float, clamp_int, clamp01
from pala.types import ActionPlan, HoldCommand, PrimitiveKind


def _valid_env_json(**overrides):
    base = {
        "schema_version": "pala.env_summary.v1",
        "scene": "desk scene",
        "events": "user leaned forward",
        "hypotheses": "user preparing to type",
        "summary_short": "user shifted into focused posture",
        "delta_score": 0.7,
        "features": {
            "person_present": True,
            "zone_hint": "left",
            "activity_level": 0.6,
            "novelty": 0.4,
        },
    }
    base.update(overrides)
    return json.dumps(base)


def _valid_proposal(**overrides):
    item = {
        "intent": "track_user",
        "primitive": "orient_to_zone",
        "command": {"zone": "left", "amp_rad": 0.2, "rate_rad_s": 1.1},
        "style": "focused",
        "score": 0.8,
        "confidence": 0.7,
        "urgency": 0.5,
        "risk": "low",
        "allow_interrupt": True,
        "evidence": ["frame:latest"],
        "rationale_short": "track user movement",
    }
    item.update(overrides)
    return item


def _valid_proposer_json(proposals):
    return json.dumps(
        {
            "schema_version": "pala.intent_proposals.v2",
            "notes_short": "ok",
            "proposals": proposals,
        }
    )


def test_behavior_type_clamp_helpers():
    assert clamp01(1.8) == 1.0
    assert clamp01(-2.0) == 0.0
    assert clamp01("bad", default=0.25) == 0.25
    assert clamp_float("bad", lo=0.0, hi=2.0, default=0.5) == 0.5
    assert clamp_float(5.0, lo=0.0, hi=2.0, default=0.5) == 2.0
    assert clamp_int("bad", lo=1, hi=3, default=2) == 2
    assert clamp_int(7, lo=1, hi=3, default=2) == 3


def test_trace_bus_fail_closed_and_payload_copy():
    class _Logger:
        def __init__(self):
            self.items = []
            self.closed = 0
            self.fail_write = False
            self.fail_close = False

        def write(self, obj):
            if self.fail_write:
                raise RuntimeError("write fail")
            self.items.append(obj)

        def close(self):
            self.closed += 1
            if self.fail_close:
                raise RuntimeError("close fail")

    logger = _Logger()
    bus = TraceBus(logger)
    payload = {"x": 1}
    bus.emit(payload)
    payload["x"] = 2
    assert logger.items == [{"x": 1}]

    logger.fail_write = True
    bus.emit({"y": 1})
    logger.fail_close = True
    bus.close()
    TraceBus(None).emit({"noop": True})
    TraceBus(None).close()


def test_context_builder_env_and_planner_context_shapes():
    builder = ContextBuilder()
    current = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.3,
        style="calm",
    )
    world = {
        "latest_env_snapshot": {
            "summary": " ".join(["x"] * 300),
            "scene": "Desk with user and monitor",
            "events": "User moved left",
            "hypotheses": "User reaching for keyboard",
            "delta_score": 0.8,
            "features": {"activity_level": "0.6", "novelty": "0.2", "person_present": 1, "zone_hint": "right"},
        },
        "event_tail": [
            {"timestamp_wall_s": "bad-ts", "summary": "first"},
            {"timestamp_wall_s": 1_700_000_000.0, "summary": "second"},
            {"timestamp_wall_s": 1_700_000_001.0, "summary": "third"},
        ],
        "decision_tail": [
            {
                "timestamp_wall_s": 1_700_000_002.0,
                "primitive": "hold",
                "style": "calm",
                "confidence": 0.4,
                "rationale_short": " ".join(["r"] * 200),
            }
        ],
        "control_state_latest": "active_kind=hold",
        "session_digest": " ".join(["d"] * 500),
    }

    env_ctx = builder.build_env_context(world_snapshot=world, current_action=current, frame_timeline=[{"ordinal": 1}])
    assert len(env_ctx["recent_env_events"]) == 2
    assert env_ctx["recent_env_events"][0]["summary"] == "second"
    assert env_ctx["frame_timeline"] == [{"ordinal": 1}]

    planner_ctx = builder.build_planner_context(
        st=None,
        world_snapshot=world,
        current_action=current,
        planner_health={"state": "HEALTHY"},
        now_mono_s=10.0,
        last_commit_mono_s=8.0,
        no_commit_s=2.0,
    )
    assert planner_ctx["signals"]["zone_hint"] == "right"
    assert planner_ctx["signals"]["person_present"] is True
    assert planner_ctx["latest_env"]["summary"].endswith("...")
    assert planner_ctx["anti_collapse"]["no_commit_s"] == 2.0


def test_context_builder_omits_unknown_zone_from_model_context():
    builder = ContextBuilder()
    current = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.3,
        style="calm",
    )
    world = {
        "latest_env_snapshot": {
            "summary": "steady scene",
            "delta_score": 0.1,
            "features": {"person_present": False, "zone_hint": "unknown", "activity_level": 0.1, "novelty": 0.1},
        },
        "control_state_latest": "active_kind=hold",
    }

    planner_ctx = builder.build_planner_context(
        st=None,
        world_snapshot=world,
        current_action=current,
        planner_health={"state": "HEALTHY"},
        now_mono_s=10.0,
        last_commit_mono_s=8.0,
        no_commit_s=2.0,
    )
    assert "zone_hint" not in planner_ctx["signals"]
    assert "perception:zone:unknown" not in planner_ctx["evidence_index"]["available"]


def test_env_summarizer_valid_and_error_paths(monkeypatch):
    parsed = parse_env_summary_response(_valid_env_json())
    assert parsed is not None
    assert parsed.summary.features["zone_hint"] == "left"

    bad = parse_env_summary_response("[]")
    assert bad is None

    # Schema allows non-empty whitespace strings; parser should fail after cleaning.
    whitespace = parse_env_summary_response(
        _valid_env_json(scene="   ", events="   ", hypotheses="   ", summary_short="   ")
    )
    assert whitespace is None

    # Non-dict features should be normalized to default feature values.
    monkeypatch.setattr("pala.behavior.env_summarizer.validate", lambda instance, schema: None)
    summarizer = EnvSummarizer()
    summarizer.submit_or_replace({"tick": 1})
    parsed = summarizer.complete_request(
        json.dumps(
            {
                "schema_version": "pala.env_summary.v1",
                "scene": "s",
                "events": "e",
                "hypotheses": "h",
                "summary_short": "ss",
                "delta_score": 0.5,
                "features": [],
            }
        )
    )
    assert parsed is not None
    assert parsed.summary.features["zone_hint"] == "unknown"


def test_env_summarizer_accepts_wrapped_payload_shape():
    raw_wrapped = json.dumps(
        {
            "pala.env_summary.v1": {
                "scene": "desk scene",
                "events": "user leaned forward",
                "hypotheses": "user preparing to type",
                "summary_short": "user shifted into focused posture",
                "delta_score": 0.5,
                "features": {
                    "person_present": True,
                    "zone_hint": "center",
                    "activity_level": 0.6,
                    "novelty": 0.2,
                },
            }
        }
    )
    summarizer = EnvSummarizer()
    summarizer.submit_or_replace({"tick": 1})
    parsed = summarizer.complete_request(raw_wrapped)
    assert parsed is not None
    assert parsed.summary.summary_short == "user shifted into focused posture"


def test_intent_proposer_valid_normalized_and_error_paths():
    parsed = parse_intent_proposer_response(
        _valid_proposer_json(
            [
                _valid_proposal(),
                _valid_proposal(
                    primitive="glance",
                    command={"direction": "left", "amp_rad": 0.2, "duration_s": 0.4, "rate_rad_s": 1.3},
                    intent="scan_environment",
                    style="curious",
                    rationale_short="scan left quickly",
                ),
                _valid_proposal(
                    primitive="breath",
                    command={"amp_rad": 0.08, "period_s": 7.0, "rate_rad_s": 1.0},
                    intent="idle_presence",
                    style="calm",
                    rationale_short="fallback breathe",
                ),
            ]
        )
    )
    assert parsed is not None
    assert parsed.response.proposals[0].primitive == "orient_to_zone"

    # Invalid proposal items are dropped; valid items remain.
    normalized = parse_intent_proposer_response(
        _valid_proposer_json(
            [
                _valid_proposal(
                    style="unknown_style",
                    primitive="glance",
                    command={"direction": "right", "amp_rad": 0.2, "duration_s": 0.4, "rate_rad_s": 1.0},
                    intent="scan_environment",
                    rationale_short="bad style test",
                )
            ]
            + [
                _valid_proposal(
                    primitive="breath",
                    command={"amp_rad": 0.09, "period_s": 9.0, "rate_rad_s": 1.2},
                    intent="idle_presence",
                    style="calm",
                    rationale_short="fallback 1",
                ),
                _valid_proposal(
                    primitive="hold",
                    command={},
                    intent="idle_presence",
                    style="calm",
                    rationale_short="fallback 2",
                ),
            ]
        )
    )
    assert normalized is not None
    assert len(normalized.response.proposals) == 2

    # Extra top-level keys still fail the response envelope schema.
    salvage_raw = json.dumps(
        {
            "schema_version": "pala.intent_proposals.v2",
            "notes_short": "ok",
            "extra": 1,
            "proposals": [_valid_proposal(), _valid_proposal(), {"intent": "bad"}],
        }
    )
    salvage = parse_intent_proposer_response(salvage_raw)
    assert salvage is None

    assert parse_intent_proposer_response("") is None
    assert parse_intent_proposer_response("[]") is None
    assert parse_intent_proposer_response(_valid_proposer_json([])) is None
    assert parse_intent_proposer_response(_valid_proposer_json([{"intent": "bad"}])) is None

    proposer = IntentProposer()
    proposer.submit_or_replace({"tick": 1})
    assert proposer.complete_request(_valid_proposer_json([{"intent": "bad"}])) is None
    err = proposer.last_parse_error or ""
    assert err == "all_proposals_invalid"


def test_intent_proposer_accepts_wrapped_payload_shape():
    wrapped = json.dumps(
        {
            "pala.intent_proposals.v2": {
                "notes_short": "ok",
                "proposals": [
                    _valid_proposal(),
                    _valid_proposal(
                        primitive="glance",
                        command={"direction": "left", "amp_rad": 0.2, "duration_s": 0.4, "rate_rad_s": 1.1},
                        intent="scan_environment",
                        style="curious",
                        rationale_short="scan left",
                    ),
                    _valid_proposal(
                        primitive="breath",
                        command={"amp_rad": 0.08, "period_s": 7.0, "rate_rad_s": 1.0},
                        intent="idle_presence",
                        style="calm",
                        rationale_short="fallback breathe",
                    ),
                ],
            }
        }
    )
    proposer = IntentProposer()
    proposer.submit_or_replace({"tick": 1})
    parsed = proposer.complete_request(wrapped)
    assert parsed is not None
    assert len(parsed.response.proposals) == 3


def test_intent_proposer_parses_fenced_and_alt_shape():
    raw = (
        "```json\n"
        + json.dumps(
            {
                "intent_proposals": [
                    _valid_proposal(
                        primitive="glance",
                        command={"direction": "right", "amp_rad": 0.3, "duration_s": 0.5, "rate_rad_s": 1.4},
                        intent="scan_environment",
                        style="curious",
                        rationale_short="brief acknowledge",
                    ),
                    _valid_proposal(
                        primitive="orient_to_zone",
                        command={"zone": "left", "amp_rad": 0.2, "rate_rad_s": 1.2},
                        intent="track_user",
                        rationale_short="track left",
                    ),
                    _valid_proposal(
                        primitive="breath",
                        command={"amp_rad": 0.08, "period_s": 6.8, "rate_rad_s": 1.0},
                        intent="idle_presence",
                        style="calm",
                        rationale_short="fallback breathe",
                    ),
                ]
            }
        )
        + "\n```"
    )
    parsed = parse_intent_proposer_response(raw)
    assert parsed is not None
    assert len(parsed.response.proposals) == 3
    assert parsed.response.proposals[0].primitive == "glance"
    assert parsed.parse_stage in {"defenced", "extracted"}


def test_intent_proposer_rejects_invalid_zone_alias_and_keeps_reset_pose_primitive():
    raw = json.dumps(
        {
            "schema_version": "pala.intent_proposals.v2",
            "proposals": [
                {
                    "intent": "react_to_change",
                    "primitive": "orient_to_zone",
                    "command": {"direction": "forward"},
                    "style": "calm",
                    "score": 0.7,
                    "confidence": 0.6,
                    "urgency": 0.5,
                    "risk": "low",
                    "allow_interrupt": True,
                    "evidence": ["frame:latest"],
                    "rationale_short": "center user",
                },
                {
                    "intent": "reset_pose",
                    "primitive": "home",
                    "command": {"rate_rad_s": 1.2},
                    "style": "calm",
                    "score": 0.6,
                    "confidence": 0.7,
                    "urgency": 0.4,
                    "risk": "low",
                    "allow_interrupt": True,
                    "evidence": ["frame:latest"],
                    "rationale_short": "return home",
                },
                {
                    "intent": "idle_presence",
                    "primitive": "hold",
                    "command": {},
                    "style": "calm",
                    "score": 0.2,
                    "confidence": 0.5,
                    "urgency": 0.1,
                    "risk": "low",
                    "allow_interrupt": True,
                    "evidence": ["frame:latest"],
                    "rationale_short": "fallback hold",
                },
            ],
        }
    )
    parsed = parse_intent_proposer_response(raw)
    assert parsed is not None
    assert len(parsed.response.proposals) == 2
    assert all(item.primitive != "orient_to_zone" for item in parsed.response.proposals)


def test_intent_proposer_keeps_reset_pose_alias_when_payload_is_valid():
    raw = json.dumps(
        {
            "schema_version": "pala.intent_proposals.v2",
            "proposals": [
                {
                    "intent": "reset_pose",
                    "primitive": "home",
                    "command": {"rate_rad_s": 1.3},
                    "style": "calm",
                    "score": 0.6,
                    "confidence": 0.7,
                    "urgency": 0.4,
                    "risk": "low",
                    "allow_interrupt": True,
                    "evidence": ["frame:latest"],
                    "rationale_short": "return home",
                },
                {
                    "intent": "track_user",
                    "primitive": "glance",
                    "command": {"direction": "left", "amp_rad": 0.2, "duration_s": 0.4, "rate_rad_s": 1.1},
                    "style": "curious",
                    "score": 0.7,
                    "confidence": 0.8,
                    "urgency": 0.5,
                    "risk": "low",
                    "allow_interrupt": True,
                    "evidence": ["frame:latest"],
                    "rationale_short": "check user side",
                },
                {
                    "intent": "idle_presence",
                    "primitive": "hold",
                    "command": {},
                    "style": "calm",
                    "score": 0.2,
                    "confidence": 0.6,
                    "urgency": 0.1,
                    "risk": "low",
                    "allow_interrupt": True,
                    "evidence": ["frame:latest"],
                    "rationale_short": "fallback hold",
                },
            ],
        }
    )
    parsed = parse_intent_proposer_response(raw)
    assert parsed is not None
    assert parsed.response.proposals[0].primitive == "home"


def test_intent_proposer_rejects_invalid_glance_direction():
    raw = _valid_proposer_json(
        [
            _valid_proposal(
                primitive="glance",
                command={"direction": "forward", "amp_rad": 0.3, "duration_s": 0.4, "rate_rad_s": 1.6},
            )
        ]
    )
    assert parse_intent_proposer_response(raw) is None


def test_env_summarizer_parses_fenced_relaxed_shape():
    raw = (
        "```json\n"
        + json.dumps(
            {
                "schema_version": "1",
                "scene": {"person_present": True, "zone_hint": "left", "activity_level": 0.5, "novelty": 0.4},
                "events": [{"description": "person looked left"}],
                "summary": "person shifted posture",
            }
        )
        + "\n```"
    )
    parsed = parse_env_summary_response(raw)
    assert parsed is not None
    assert parsed.summary.features["person_present"] is True
    assert parsed.summary.summary_short.startswith("person shifted posture")
    assert parsed.parse_stage in {"defenced", "extracted"}


def test_parse_json_flexible_accepts_answer_wrappers():
    raw = "<answer>```json\n{\"ok\":true}\n```</answer>"
    data, err, stage = parse_json_flexible(raw)
    assert err is None
    assert data == {"ok": True}
    assert stage in {"defenced", "extracted"}


def test_governor_and_action_compiler_and_arbiter_branches():
    gov = Governor(GovernorConfig(block_high_risk=False, block_home_unless_reset_pose=True))
    base = _valid_proposal()
    candidates = [
        ProposalCandidate(
            proposal=IntentProposal(**{**base, "primitive": "home", "intent": "track_user"}),
            source="remote",
        ),
        ProposalCandidate(
            proposal=IntentProposal(**{**base, "primitive": "orient_to_zone", "command": {"zone": "desk"}}),
            source="remote",
        ),
        ProposalCandidate(
            proposal=IntentProposal(**{**base, "risk": "medium"}),
            source="idle_engine",
        ),
    ]
    out = gov.evaluate(candidates, mode=BehaviorMode.ENGAGE_TRACK, signals={"person_present": True})
    assert out[0].valid is False and out[0].reject_reason == "home_blocked"
    assert out[1].valid is False and out[1].reject_reason == "invalid_zone"
    assert out[2].valid is True and out[2].utility > 0.0

    compiler = ActionCompiler()
    bad_compile = compiler.compile(
        IntentProposal(
            intent="track_user",
            primitive="orient_to_zone",
            command={"zone": "desk"},
            style="calm",
            score=0.5,
            confidence=0.5,
            urgency=0.5,
            risk="low",
            allow_interrupt=True,
            evidence=[],
            rationale_short="bad zone",
        )
    )
    assert bad_compile.action is None
    assert bad_compile.error == "invalid_action_payload"

    current = ActionPlan(primitive=PrimitiveKind.NOD, command={"amp_rad": 0.2}, confidence=0.5, style="calm")
    valid_candidate = ProposalCandidate(
        proposal=IntentProposal(
            intent="affirmation",
            primitive="nod",
            command={"amp_rad": 0.2, "duration_s": 0.4, "cycles": 1, "rate_rad_s": 1.8},
            style="calm",
            score=0.8,
            confidence=0.8,
            urgency=0.2,
            risk="low",
            allow_interrupt=True,
            evidence=[],
            rationale_short="affirm",
        ),
        source="remote",
    )
    governed = Governor(GovernorConfig(block_high_risk=False)).evaluate(
        [valid_candidate],
        mode=BehaviorMode.ACKNOWLEDGE,
        signals={"person_present": True},
    )
    arb = Arbiter(ArbiterConfig(min_dwell_s=0.0, base_margin=0.01, idle_after_s=1.0, terminal_retrigger_s=0.5))
    result = arb.select(
        candidates=governed,
        current_action=current,
        current_utility=0.1,
        action_age_s=1.0,
        no_commit_s=2.0,
        recent_switches=0,
        planner_open_breaker=False,
        same_primitive_streak=1,
        mode=BehaviorMode.ACKNOWLEDGE,
    )
    assert result.decision == "commit"
    assert result.reason == "same_signature_retrigger"
    assert result.chosen is not None


def test_health_manager_state_transitions_and_effective_hz():
    hm = HealthManager()
    hm.on_env_result(status="ok", latency_ms=12.0)
    assert hm.env.state == "HEALTHY"
    hm.on_env_result(status="transport_error", latency_ms=10.0)
    assert hm.env.state == "DEGRADED"
    hm.on_env_result(status="transport_error", latency_ms=10.0)
    hm.on_env_result(status="transport_error", latency_ms=10.0)
    assert hm.env.state == "OPEN_BREAKER"

    no_signal = ProposerResponse(
        schema_version="pala.intent_proposals.v2",
        proposals=[
            IntentProposal(
                intent="idle_presence",
                primitive="hold",
                command={},
                style="calm",
                score=0.1,
                confidence=0.2,
                urgency=0.1,
                risk="low",
                allow_interrupt=True,
                evidence=[],
                rationale_short="hold",
            )
        ],
        notes_short="",
    )
    for _ in range(6):
        hm.on_planner_result(status="ok", latency_ms=20.0, response=no_signal)
    assert hm.planner.state == "DEGRADED"
    assert hm.planner_effective_hz(1.0) <= 0.35

    hm.on_planner_result(status="transport_error", latency_ms=30.0, response=None)
    hm.on_planner_result(status="transport_error", latency_ms=30.0, response=None)
    hm.on_planner_result(status="transport_error", latency_ms=30.0, response=None)
    assert hm.planner_open_breaker() is True
    assert hm.planner_effective_hz(1.0) <= 0.15
    assert isinstance(ComponentHealth().as_dict(), dict)

    hm.on_perception_result(detector_alive=False, source_alive=True, stale_frame=False)
    hm.on_perception_result(detector_alive=False, source_alive=True, stale_frame=False)
    hm.on_perception_result(detector_alive=False, source_alive=True, stale_frame=False)
    assert hm.perception.state == "OPEN_BREAKER"
    assert hm.perception_degraded() is True


def test_schema_response_format_helpers():
    intent = intent_response_format()
    env = env_response_format()
    assert intent["type"] == "json_schema"
    assert env["type"] == "json_schema"
    assert intent["json_schema"]["name"] == "pala_intent_proposals_v2"
    assert env["json_schema"]["name"] == "pala_env_summary_v1"
