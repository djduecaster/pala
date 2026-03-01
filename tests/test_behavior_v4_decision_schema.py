from __future__ import annotations

import json

from pala.behavior.decision_schema_v4 import (
    BehaviorDecisionParser,
    behavior_decision_response_format,
)


def _decision_payload(**overrides):
    payload = {
        "schema_version": "pala.behavior_decision.v1",
        "mode": "social_interact",
        "mood": "curious",
        "skill": "greet_user",
        "action": {
            "primitive": "orient_to_zone",
            "command": {"zone": "center", "amp_rad": 0.2, "rate_rad_s": 1.1},
            "style": "curious",
        },
        "confidence": 0.72,
        "rationale_short": "A person is in front of me, so I should orient toward center.",
        "mode_transition": "stay",
    }
    payload.update(overrides)
    return payload


def test_parser_accepts_valid_behavior_decision():
    parser = BehaviorDecisionParser()
    raw = json.dumps(_decision_payload())
    parsed = parser.parse(raw)
    assert parsed is not None
    assert parsed.decision.mode == "social_interact"
    assert parsed.decision.skill == "greet_user"
    assert parsed.decision.action.primitive == "orient_to_zone"
    assert parser.last_parse_stage == "raw"
    assert parser.last_parse_error is None


def test_parser_accepts_wrapped_fenced_payload():
    parser = BehaviorDecisionParser()
    wrapped = {"pala.behavior_decision.v1": _decision_payload()}
    raw = "```json\n" + json.dumps(wrapped) + "\n```"
    parsed = parser.parse(raw)
    assert parsed is not None
    assert parsed.decision.schema_version == "pala.behavior_decision.v1"
    assert parsed.decision.mode_transition == "stay"
    assert parser.last_parse_stage == "defenced"


def test_parser_rejects_when_required_field_missing():
    parser = BehaviorDecisionParser()
    bad = _decision_payload()
    bad.pop("action")
    parsed = parser.parse(json.dumps(bad))
    assert parsed is None
    assert parser.last_parse_error is not None
    assert parser.last_parse_error.startswith("schema:")


def test_parser_accepts_schema_version_alias_and_canonicalizes():
    parser = BehaviorDecisionParser()
    payload = _decision_payload(schema_version="pala_behavior_decision_v1")
    parsed = parser.parse(json.dumps(payload))
    assert parsed is not None
    assert parsed.decision.schema_version == "pala.behavior_decision.v1"
    assert parser.last_parse_error is None


def test_behavior_decision_response_format_is_strict_json_schema():
    fmt = behavior_decision_response_format()
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "pala_behavior_decision_v1"
    assert fmt["json_schema"]["strict"] is True
