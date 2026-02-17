import json

from pala.types import (
    PerceptionState,
    BBoxNorm,
    PointNorm,
    ActionPlan,
    PrimitiveKind,
    BreathCommand,
    to_json_dict,
    to_json_line,
    action_plan_from_dict,
)


def test_types_json_roundtrip():
    st = PerceptionState(
        timestamp_monotonic_s=1.23,
        timestamp_wall_s=4.56,
        fps=30.0,
        latency_ms=12.0,
        primary_person=BBoxNorm(cx=0.4, cy=0.5, w=0.2, h=0.3),
        primary_person_conf=0.9,
        pointing_target=PointNorm(x=0.1, y=0.2),
        pointing_conf=0.8,
        debug={"zone_hint": "left"},
    )
    raw = to_json_line(st)
    payload = json.loads(raw)
    assert payload == to_json_dict(st)


def test_action_plan_typed_json_roundtrip():
    action = ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command=BreathCommand(amp_rad=0.1, period_s=6.0, rate_rad_s=1.2),
        confidence=0.7,
        explanation="demo",
    )

    payload = json.loads(to_json_line(action))
    assert payload["primitive"] == "breath"
    assert payload["command"]["amp_rad"] == 0.1

    parsed = action_plan_from_dict(payload)
    assert parsed is not None
    assert parsed.primitive == PrimitiveKind.BREATH
    assert parsed.command.period_s == 6.0
