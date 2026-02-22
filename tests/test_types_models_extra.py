from __future__ import annotations

import pytest

from pala.types.models import (
    ActionPlan,
    BreathCommand,
    GlanceCommand,
    GazeToCommand,
    HoldCommand,
    HomeCommand,
    MoveToCommand,
    NodCommand,
    OrientToZoneCommand,
    PrimitiveKind,
    action_plan_from_dict,
    command_from_dict,
)


def test_command_from_dict_supports_all_primitives():
    assert isinstance(command_from_dict(PrimitiveKind.HOLD, {}), HoldCommand)
    assert isinstance(command_from_dict(PrimitiveKind.HOME, {"rate_rad_s": 2.1}), HomeCommand)
    assert isinstance(
        command_from_dict(
            PrimitiveKind.MOVE_TO,
            {"target_rad": [0.1, -0.2], "relative": "yes", "rate_rad_s": 1.7, "timeout_s": 3.0},
        ),
        MoveToCommand,
    )
    assert isinstance(
        command_from_dict(
            PrimitiveKind.GAZE_TO,
            {"yaw_rad": 0.4, "pitch_rad": -0.2, "rate_rad_s": 1.1, "dwell_s": 0.2, "timeout_s": 2.2},
        ),
        GazeToCommand,
    )
    glance = command_from_dict(PrimitiveKind.GLANCE, {"direction": "UP", "amp_rad": 0.2, "duration_s": 0.3})
    assert isinstance(glance, GlanceCommand)
    assert glance.direction == "up"
    nod = command_from_dict(PrimitiveKind.NOD, {"cycles": "2"})
    assert isinstance(nod, NodCommand)
    assert nod.cycles == 2
    assert isinstance(command_from_dict(PrimitiveKind.BREATH, {"amp_rad": 0.09}), BreathCommand)
    orient = command_from_dict(PrimitiveKind.ORIENT_TO_ZONE, {"zone": "RIGHT"})
    assert isinstance(orient, OrientToZoneCommand)
    assert orient.zone == "right"


def test_command_from_dict_rejects_invalid_payloads():
    with pytest.raises(ValueError, match="target_rad is required"):
        command_from_dict(PrimitiveKind.MOVE_TO, {})
    with pytest.raises(ValueError, match="must not be empty"):
        command_from_dict(PrimitiveKind.MOVE_TO, {"target_rad": []})
    with pytest.raises(ValueError, match="direction must be"):
        command_from_dict(PrimitiveKind.GLANCE, {"direction": "forward"})
    with pytest.raises(ValueError, match="cycles must be >= 1"):
        command_from_dict(PrimitiveKind.NOD, {"cycles": 0})
    with pytest.raises(ValueError, match="zone must be"):
        command_from_dict(PrimitiveKind.ORIENT_TO_ZONE, {"zone": "desk"})
    with pytest.raises(ValueError, match="yaw_rad is required"):
        command_from_dict(PrimitiveKind.GAZE_TO, {"pitch_rad": 0.1})


def test_action_plan_from_dict_supports_wrapped_payload_and_defaults():
    parsed = action_plan_from_dict(
        {
            "action": {
                "primitive": "HOLD",
                "command": {},
                "confidence": "bad-number",
                "style": " ",
                "action_id": " ",
                "cancel_current": 1,
                "explanation": 42,
            }
        }
    )
    assert parsed is not None
    assert parsed.primitive == PrimitiveKind.HOLD
    assert isinstance(parsed.command, HoldCommand)
    assert parsed.confidence == 0.5
    assert parsed.style == "calm"
    assert parsed.cancel_current is True
    assert parsed.explanation is None
    assert parsed.action_id


def test_action_plan_from_dict_rejects_invalid_primitive_or_command():
    assert action_plan_from_dict({"primitive": "unknown", "command": {}}) is None
    assert action_plan_from_dict({"primitive": "hold", "command": 3}) is None


def test_action_plan_post_init_coercions_and_validation():
    action = ActionPlan(
        primitive="move_to",
        command={"target_rad": [0.1], "relative": "yes"},
        confidence=1.9,
        style=" FOCUSED ",
        action_id=" ",
        cancel_current="on",
    )
    assert action.primitive == PrimitiveKind.MOVE_TO
    assert isinstance(action.command, MoveToCommand)
    assert action.command.relative is True
    assert action.confidence == 1.0
    assert action.style == "focused"
    assert action.cancel_current is True
    assert action.action_id

    with pytest.raises(ValueError, match="Unknown primitive kind"):
        ActionPlan(primitive="bad", command={}, confidence=0.5)
    with pytest.raises(ValueError, match="must be HoldCommand"):
        ActionPlan(primitive=PrimitiveKind.HOLD, command=123, confidence=0.5)
