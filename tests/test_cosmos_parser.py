from pala.control.primitives import PrimitiveKind
from pala.planner.cosmos_async import _parse_action_content


def test_parse_action_content_typed_payload():
    content = (
        '{"primitive":"breath","command":{"amp_rad":0.1,"period_s":6.0,"rate_rad_s":1.1},'
        '"confidence":0.65,"explanation":"ok"}'
    )
    action = _parse_action_content(content)
    assert action is not None
    assert action.primitive == PrimitiveKind.BREATH
    assert action.command.period_s == 6.0
    assert action.confidence == 0.65


def test_parse_action_content_rejects_missing_command():
    content = '{"primitive":"breath","confidence":0.5}'
    action = _parse_action_content(content)
    assert action is None


def test_parse_action_content_rejects_unknown_primitive():
    content = '{"primitive":"dance","command":{},"confidence":0.5}'
    action = _parse_action_content(content)
    assert action is None
