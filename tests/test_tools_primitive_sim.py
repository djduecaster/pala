from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pala.behavior.mode_fsm_v4 import ModeFsmV4Config
from pala.types import ActionPlan
from pala.control.primitives import BreathCommand, PrimitiveKind
from tools.primitive_sim.run import (
    _append_experiment_record,
    _expand_sweep_grid,
    _default_joint_angles,
    _default_baseline,
    _extract_numeric_dh_params,
    _load_baseline,
    _load_experiment_history,
    _load_viewer_geometry_from_config,
    _normalize_baseline,
    _primitive_specs,
    _score_trace_metrics,
    _scenario_segments_from_payload,
    _save_baseline,
    _target_step_index,
    _trace_metrics,
)
from tools.primitive_sim.simulate import (
    SimSegment,
    build_suite_segments,
    load_segments_from_json,
    simulate_segments,
)
from tools.primitive_sim.state_machine import LampStateMachineSimulator


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


def test_baseline_normalization_accepts_strict_v2_payload():
    cfg = _cfg_stub()
    raw = _default_baseline(cfg)
    raw["primitives"]["breath"] = {"amp_rad": 0.11, "period_s": 8.1, "rate_rad_s": 1.05}
    raw["primitives"]["move_to"]["relative"] = True
    raw["updated_by"] = "unit_test"
    raw["updated_at_utc"] = "2026-02-27T00:00:00Z"

    baseline = _normalize_baseline(raw, cfg)

    assert baseline["version"] == 2
    assert baseline.get("updated_by") == "unit_test"
    assert baseline.get("updated_at_utc") == "2026-02-27T00:00:00Z"
    assert baseline["primitives"]["breath"]["amp_rad"] == 0.11
    assert baseline["primitives"]["move_to"]["relative"] is True


def test_baseline_normalization_rejects_v1_payload():
    cfg = _cfg_stub()
    raw = {
        "version": 1,
        "primitives": {
            "breath": {"amp_rad": 0.11, "period_s": 8.1, "rate_rad_s": 1.05},
        },
    }
    with pytest.raises(ValueError, match="unsupported baseline version"):
        _normalize_baseline(raw, cfg)


def test_baseline_normalization_rejects_missing_primitive_payload():
    cfg = _cfg_stub()
    raw = _default_baseline(cfg)
    del raw["primitives"]["glance"]
    with pytest.raises(ValueError, match="missing baseline primitive payload: glance"):
        _normalize_baseline(raw, cfg)


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


def test_load_baseline_rejects_v1_schema(tmp_path):
    cfg = _cfg_stub()
    p = tmp_path / "baseline_v1.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "primitives": {
                    "breath": {"amp_rad": 0.09, "period_s": 8.0, "rate_rad_s": 1.0},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported baseline version"):
        _load_baseline(p, cfg)


def test_save_baseline_preserves_existing_metadata(tmp_path):
    cfg = _cfg_stub()
    baseline = _default_baseline(cfg)
    baseline["updated_by"] = "unit_test"
    baseline["updated_at_utc"] = "2026-02-27T00:00:00Z"

    p = tmp_path / "baseline.json"
    _save_baseline(p, baseline)
    saved = json.loads(p.read_text(encoding="utf-8"))

    assert saved["version"] == 2
    assert saved["updated_by"] == "unit_test"
    assert saved["updated_at_utc"] == "2026-02-27T00:00:00Z"


def test_scenario_segments_from_payload_parses_and_canonicalizes():
    cfg = _cfg_stub()
    baseline = _default_baseline(cfg)
    segments = _scenario_segments_from_payload(
        raw_steps=[
            {
                "name": "home_step",
                "primitive": "home",
                "duration_s": 1.6,
                "command": {"rate_rad_s": 1.4},
            },
            {
                "primitive": "glance",
                "style": "curious",
                "duration_s": 0.8,
                "command": {"direction": "right", "amp_rad": 0.24},
            },
        ],
        cfg=cfg,
        style_options=["calm", "curious"],
        baseline=baseline,
    )
    assert len(segments) == 2
    assert segments[0].name == "home_step"
    assert segments[0].action.primitive == PrimitiveKind.HOME
    assert float(getattr(segments[0].action.command, "rate_rad_s")) == 1.4
    assert segments[1].action.primitive == PrimitiveKind.GLANCE
    assert segments[1].action.style == "curious"
    assert str(getattr(segments[1].action.command, "direction")) == "right"


def test_scenario_segments_from_payload_rejects_empty_steps():
    cfg = _cfg_stub()
    with pytest.raises(ValueError, match="steps must be a non-empty array"):
        _scenario_segments_from_payload(
            raw_steps=[],
            cfg=cfg,
            style_options=["calm"],
            baseline=_default_baseline(cfg),
        )


def test_trace_metrics_returns_core_stats():
    segment = SimSegment(
        name="home",
        action=ActionPlan(
            primitive=PrimitiveKind.HOME,
            command={"rate_rad_s": 1.2},
            confidence=1.0,
            explanation="test",
            style="calm",
        ),
        max_s=1.5,
        stop_on_done=True,
    )
    trace = simulate_segments(
        joint_names=["yaw", "pitch1", "pitch2", "roll", "pitch3"],
        joint_limits_rad=[[-1.57, 1.57] for _ in range(5)],
        segments=[segment],
        hz=50.0,
    )
    metrics = _trace_metrics(trace)
    assert metrics["sample_count"] > 0
    assert metrics["duration_s"] >= 0.0
    assert metrics["peak_joint_vel_rad_s"] >= 0.0
    assert "limit_violation_count" in metrics
    assert "primitive_switch_count" in metrics


def test_experiment_history_appends_and_reads_newest_first(tmp_path):
    p = tmp_path / "experiments.jsonl"
    _append_experiment_record(
        p,
        {
            "saved_at_utc": "2026-02-27T00:00:00Z",
            "name": "exp_a",
            "metrics": {"sample_count": 10},
        },
    )
    _append_experiment_record(
        p,
        {
            "saved_at_utc": "2026-02-27T00:01:00Z",
            "name": "exp_b",
            "metrics": {"sample_count": 20},
        },
    )
    rows = _load_experiment_history(p, limit=2)
    assert len(rows) == 2
    assert rows[0]["name"] == "exp_b"
    assert rows[1]["name"] == "exp_a"


def _fsm_sim() -> LampStateMachineSimulator:
    return LampStateMachineSimulator.create(
        mode_config=ModeFsmV4Config(),
        idle_config=None,
    )


def test_state_machine_defaults_to_boot_awaken():
    sim = _fsm_sim()
    snap = sim.snapshot()
    assert snap["mode"] == "boot_awaken"
    assert "hold" in snap["allowed_primitives"]
    assert "breath" in snap["allowed_primitives"]


def test_state_machine_boot_holds_without_startup_complete_signal():
    sim = _fsm_sim()
    out = sim.step(
        dt_s=0.5,
        signals={
            "person_present": True,
            "person_conf": 0.95,
        },
    )
    assert out["mode_decision"]["to"] == "boot_awaken"
    assert out["mode_decision"]["transitioned"] is False
    assert out["mode_decision"]["reason"] in {"hold_mode", "min_mode_dwell_hold"}


def test_state_machine_presence_step_transitions_to_social_interact():
    sim = _fsm_sim()
    _ = sim.step(
        dt_s=0.5,
        signals={"startup_complete": True},
        zone_hint="center",
    )
    out = sim.step(
        dt_s=1.3,
        signals={
            "person_present": True,
            "person_conf": 0.85,
            "startup_complete": True,
        },
        zone_hint="center",
    )
    assert out["mode_decision"]["to"] == "social_interact"
    assert out["mode_decision"]["reason"] == "person_present_engage"
    assert any(item["primitive"] == "orient_to_zone" for item in out["proposals"])


def test_state_machine_health_breaker_forces_recover_reset():
    sim = _fsm_sim()
    out = sim.step(
        dt_s=1.1,
        signals={
            "person_present": False,
            "person_conf": 0.0,
            "planner_open_breaker": True,
            "perception_degraded": False,
        },
    )
    assert out["mode_decision"]["to"] == "recover_reset"
    assert out["mode_decision"]["reason"] == "health_degraded"
    assert out["recommended"]["primitive"] == "hold"


def test_state_machine_home_request_transitions_to_return_home():
    sim = _fsm_sim()
    _ = sim.step(dt_s=0.5, signals={"startup_complete": True})
    out = sim.step(
        dt_s=1.3,
        signals={
            "startup_complete": True,
            "home_requested": True,
        },
    )
    assert out["mode_decision"]["to"] == "return_home"
    assert out["mode_decision"]["reason"] == "home_requested"
    assert out["recommended"]["primitive"] == "home"


def test_state_machine_force_mode_transitions():
    sim = _fsm_sim()
    out = sim.force_mode(
        next_mode="recover_reset",
        reason="ops_force",
        dt_s=0.2,
        signals={"health_degraded": True},
    )
    assert out["mode_decision"]["to"] == "recover_reset"
    assert out["mode_decision"]["reason"] == "ops_force"
    assert out["tick_index"] == 1


def test_state_machine_force_mode_rejects_invalid_mode():
    sim = _fsm_sim()
    with pytest.raises(ValueError, match="invalid mode token"):
        sim.force_mode(next_mode="bad_mode")


def test_state_machine_commit_resets_no_commit_timer():
    sim = _fsm_sim()
    out1 = sim.step(dt_s=0.5, signals={})
    assert out1["no_commit_s"] > 0.0
    out2 = sim.step(dt_s=0.5, signals={}, commit=True)
    assert out2["no_commit_s"] == 0.0


def test_expand_sweep_grid_cartesian_product():
    grid = _expand_sweep_grid({"amp_rad": [0.05, 0.1], "period_s": [4.0, 6.0]})
    assert len(grid) == 4
    assert {"amp_rad": 0.05, "period_s": 4.0} in grid
    assert {"amp_rad": 0.1, "period_s": 6.0} in grid


def test_expand_sweep_grid_rejects_invalid_values():
    with pytest.raises(ValueError, match="param_grid must be a non-empty object"):
        _expand_sweep_grid({})
    with pytest.raises(ValueError, match="must be a non-empty array"):
        _expand_sweep_grid({"amp_rad": []})
    with pytest.raises(ValueError, match=r"param_grid\[amp_rad\]\[0\] must be numeric"):
        _expand_sweep_grid({"amp_rad": ["oops", 0.1]})


def test_target_step_index_by_index_or_name():
    steps = [
        {"name": "home_step"},
        {"name": "breath_step"},
        {"name": "glance_step"},
    ]
    assert _target_step_index(steps, None) == 2
    assert _target_step_index(steps, 1) == 1
    assert _target_step_index(steps, -1) == 2
    assert _target_step_index(steps, "breath_step") == 1
    with pytest.raises(ValueError, match="target_step not found"):
        _target_step_index(steps, "missing_step")


def test_score_trace_metrics_applies_weights():
    metrics = {
        "min_limit_margin_rad": 0.2,
        "limit_violation_count": 1,
        "peak_joint_vel_rad_s": 0.5,
    }
    score = _score_trace_metrics(
        metrics,
        {
            "min_limit_margin_rad": 4.0,
            "limit_violation_count": -12.0,
            "peak_joint_vel_rad_s": -0.35,
        },
    )
    expected = 4.0 * 0.2 + (-12.0 * 1.0) + (-0.35 * 0.5)
    assert score == pytest.approx(expected, rel=1e-9, abs=1e-9)
