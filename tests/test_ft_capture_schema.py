from __future__ import annotations

import pytest

from tools.ft_capture.schema import parse_expected_decision_json, validate_label_record


def test_expected_decision_json_parse_and_validate() -> None:
    payload = (
        '{"schema_version":"pala.behavior_decision.v1",'
        '"mode":"task_lighting","mood":"focused","skill":"task_light_adjust",'
        '"action":{"primitive":"orient_to_zone","command":{"zone":"center","amp_rad":0.2,"rate_rad_s":1.4},"style":"focused"},'
        '"confidence":0.9,"rationale_short":"book lifted for reading","mode_transition":"to_task_lighting"}'
    )
    decision = parse_expected_decision_json(payload)
    assert decision.mode == "task_lighting"
    assert decision.skill == "task_light_adjust"
    assert decision.action.primitive == "orient_to_zone"
    assert decision.action.style == "focused"


def test_label_requires_decision_and_rationale_when_labeled() -> None:
    with pytest.raises(ValueError):
        validate_label_record(
            {
                "status": "labeled",
                "quality_flag": "usable",
                "expected_decision": None,
                "rationale_text": "",
            }
        )

    validated = validate_label_record(
        {
            "status": "labeled",
            "quality_flag": "usable",
            "expected_decision": {
                "schema_version": "pala.behavior_decision.v1",
                "mode": "idle_presence",
                "mood": "calm",
                "skill": "social_ack",
                "action": {
                    "primitive": "breath",
                    "command": {"amp_rad": 0.06, "period_s": 6.5, "rate_rad_s": 1.0},
                    "style": "calm",
                },
                "confidence": 0.8,
                "rationale_short": "stable idle scene",
                "mode_transition": "stay",
            },
            "rationale_text": "stable idle scene",
        }
    )
    assert validated.status == "labeled"
    assert validated.expected_decision is not None


def test_label_rejects_legacy_expected_action_key() -> None:
    with pytest.raises(ValueError, match="expected_action"):
        validate_label_record(
            {
                "status": "labeled",
                "quality_flag": "usable",
                "expected_action": {
                    "primitive": "hold",
                    "command": {},
                    "style": "calm",
                    "confidence": 0.2,
                },
                "rationale_text": "legacy key",
            }
        )
