from __future__ import annotations

import pytest

from tools.ft_capture.schema import parse_expected_action_json, validate_label_record


def test_expected_action_json_parse_and_validate() -> None:
    payload = (
        '{"intent":"track_user","primitive":"orient_to_zone",'
        '"command":{"zone":"center","amp_rad":0.2,"rate_rad_s":1.4},'
        '"style":"focused","confidence":0.9}'
    )
    action = parse_expected_action_json(payload)
    assert action.intent == "track_user"
    assert action.primitive == "orient_to_zone"
    assert action.style == "focused"


def test_label_requires_action_and_rationale_when_labeled() -> None:
    with pytest.raises(ValueError):
        validate_label_record(
            {
                "status": "labeled",
                "quality_flag": "usable",
                "expected_action": None,
                "rationale_text": "",
            }
        )

    validated = validate_label_record(
        {
            "status": "labeled",
            "quality_flag": "usable",
            "expected_action": {
                "intent": "idle_presence",
                "primitive": "breath",
                "command": {"amp_rad": 0.06, "period_s": 6.5, "rate_rad_s": 1.0},
                "style": "calm",
                "confidence": 0.8,
            },
            "rationale_text": "stable idle scene",
        }
    )
    assert validated.status == "labeled"
    assert validated.expected_action is not None
