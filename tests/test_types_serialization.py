import json

from pala.types import PerceptionState, BBoxNorm, PointNorm, to_json_dict, to_json_line


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
