from __future__ import annotations

import json
from types import SimpleNamespace

from pala.types import ActionPlan
from pala.control.primitives import BreathCommand, PrimitiveKind
from tools.primitive_sim.run import (
    _default_joint_angles,
    _default_baseline,
    _extract_numeric_dh_params,
    _load_viewer_geometry_from_config,
    _normalize_baseline,
    _primitive_specs,
)
from tools.primitive_sim.simulate import (
    SimSegment,
    build_suite_segments,
    load_segments_from_json,
    simulate_segments,
)


def _joint_limits(count: int = 5) -> list[list[float]]:
    return [[-1.57, 1.57] for _ in range(count)]


def _joint_names(count: int = 5) -> list[str]:
    return [f"joint_{i}" for i in range(count)]


def test_simulate_segments_generates_samples_and_summary():
    segment = SimSegment(
        name="breath",
        action=ActionPlan(
            primitive=PrimitiveKind.BREATH,
            command=BreathCommand(amp_rad=0.08, period_s=3.0, rate_rad_s=1.0),
            confidence=1.0,
            explanation="test",
        ),
        max_s=1.2,
        stop_on_done=False,
    )

    trace = simulate_segments(
        joint_names=_joint_names(),
        joint_limits_rad=_joint_limits(),
        segments=[segment],
        hz=60.0,
    )

    assert trace["metadata"]["segment_count"] == 1
    assert len(trace["samples"]) > 10
    assert trace["summary"]["status_counts"]
    assert len(trace["summary"]["joint_stats"]) == 5


def test_load_segments_from_json_supports_wrapped_shape(tmp_path):
    script = {
        "segments": [
            {
                "name": "test_home",
                "max_s": 1.5,
                "stop_on_done": True,
                "action": {
                    "primitive": "home",
                    "command": {"rate_rad_s": 1.2},
                    "confidence": 1.0,
                    "style": "calm",
                },
            }
        ]
    }
    path = tmp_path / "sim_script.json"
    path.write_text(json.dumps(script), encoding="utf-8")

    segments = load_segments_from_json(path)
    assert len(segments) == 1
    assert segments[0].name == "test_home"
    assert segments[0].action.primitive == PrimitiveKind.HOME


def test_build_suite_segments_has_expected_coverage():
    segments = build_suite_segments(suite_breath_s=5.0, style="curious")
    names = [s.name for s in segments]

    assert "home" in names
    assert "breath" in names
    assert "glance_left" in names
    assert "nod" in names
    assert len(segments) >= 6


def test_load_viewer_geometry_from_config_parses_supported_keys(tmp_path):
    cfg = tmp_path / "robot.yaml"
    cfg.write_text(
        (
            "sim_viewer:\n"
            "  lamp_geometry:\n"
            "    mast_height_m: 1.31\n"
            "    upper_arm_len_m: 0.55\n"
            "    fore_arm_len_m: 0.44\n"
            "    shade_tip_radius_m: 0.041\n"
            "    pitch1_zero_offset_rad: 1.57079632679\n"
            "    pitch2_zero_offset_rad: -0.15\n"
            "    bad_value: nope\n"
        ),
        encoding="utf-8",
    )
    geom = _load_viewer_geometry_from_config(cfg)
    assert geom["mastHeight"] == 1.31
    assert geom["upperArmLen"] == 0.55
    assert geom["foreArmLen"] == 0.44
    assert geom["shadeFrontRadius"] == 0.041
    assert geom["pitch1ZeroOffsetRad"] == 1.57079632679
    assert geom["pitch2ZeroOffsetRad"] == -0.15


def test_load_viewer_geometry_from_config_derives_from_dh_params(tmp_path):
    cfg = tmp_path / "robot.yaml"
    cfg.write_text(
        (
            "dh_params:\n"
            "  yaw_d: 0.061\n"
            "  pitch1_a: 0.401\n"
            "  roll_d: 0.337\n"
            "  pitch1_theta0_deg: -90\n"
            "  pitch2_theta0_deg: 15\n"
            "  pitch3_theta0_deg: 8\n"
        ),
        encoding="utf-8",
    )
    geom = _load_viewer_geometry_from_config(cfg)
    assert geom["hubRise"] == 0.061
    assert geom["upperArmLen"] == 0.401
    assert geom["foreArmLen"] == 0.337
    assert abs(geom["pitch1ZeroOffsetRad"] + 1.57079632679) < 1e-9
    assert abs(geom["pitch2ZeroOffsetRad"] - 0.26179938779) < 1e-9
    assert abs(geom["pitch3ZeroOffsetRad"] - 0.13962634016) < 1e-9


def _cfg_stub() -> SimpleNamespace:
    return SimpleNamespace(
        joint_names=["yaw", "pitch1", "pitch2", "roll", "pitch3"],
        joint_limits_rad=[[-1.57, 1.57] for _ in range(5)],
        style_profiles={"calm": {"amp_scale": 1.0}},
    )


def test_baseline_normalization_restores_missing_and_invalid_fields():
    cfg = _cfg_stub()
    raw = {
        "version": 1,
        "primitives": {
            "breath": {"amp_rad": 0.11, "period_s": 8.1, "rate_rad_s": 1.05},
            "move_to": {"target_rad": [0.1, 0.2], "relative": "true", "timeout_s": 3.0},
            "glance": {"direction": "RIGHT", "amp_rad": 0.3, "duration_s": 0.6, "rate_rad_s": 1.4},
        },
    }

    baseline = _normalize_baseline(raw, cfg)
    defaults = _default_baseline(cfg)

    assert baseline["version"] == 1
    assert set(baseline["primitives"]) == set(defaults["primitives"])
    assert baseline["primitives"]["breath"]["amp_rad"] == 0.11
    assert baseline["primitives"]["move_to"]["relative"] is True
    assert len(baseline["primitives"]["move_to"]["target_rad"]) == 5
    assert baseline["primitives"]["home"] == defaults["primitives"]["home"]


def test_primitive_specs_include_all_runtime_primitives():
    specs = _primitive_specs(_cfg_stub())
    ids = {row["id"] for row in specs}
    assert ids == {
        "hold",
        "home",
        "move_to",
        "gaze_to",
        "glance",
        "nod",
        "breath",
        "orient_to_zone",
    }


def test_extract_numeric_dh_params_filters_non_numeric_values():
    raw = {
        "dh_params": {
            "yaw_d": 0.0635,
            "pitch1_theta0_deg": "90",
            "bad_token": "nope",
            "none_value": None,
        }
    }
    dh = _extract_numeric_dh_params(raw)
    assert dh == {"yaw_d": 0.0635, "pitch1_theta0_deg": 90.0}


def test_default_joint_angles_clamps_zero_to_limits():
    cfg = SimpleNamespace(
        joint_names=["j0", "j1", "j2"],
        joint_limits_rad=[[0.2, 1.0], [-1.0, 1.0], [-2.0, -0.1]],
    )
    assert _default_joint_angles(cfg) == [0.2, 0.0, -0.1]
