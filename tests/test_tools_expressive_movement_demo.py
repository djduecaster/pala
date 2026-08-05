from __future__ import annotations

from types import SimpleNamespace

import pytest

from pala.control.primitives import MoveToCommand, PrimitiveKind
from tools.expressive_movement_demo import _excited_targets, build_demo_segments


def _cfg_stub() -> SimpleNamespace:
    return SimpleNamespace(
        joint_names=["yaw", "pitch1", "pitch2", "roll", "pitch3"],
        joint_limits_rad=[[-1.57, 1.57] for _ in range(5)],
    )


def _args_stub(**overrides) -> SimpleNamespace:
    base = {
        "breath_s": 2.6,
        "curious_pause_s": 0.45,
        "breath_amp_rad": 0.08,
        "breath_period_s": 6.0,
        "curious_glance_amp_rad": 0.2,
        "curious_glance_duration_s": 0.55,
        "curious_glance_rate_rad_s": 1.45,
        "curious_orient_amp_rad": 0.24,
        "curious_orient_rate_rad_s": 1.0,
        "excite_cycles": 3,
        "base_amp_rad": 0.18,
        "pitch1_back_rad": 0.3,
        "pitch1_back_sign": -1.0,
        "excite_rate_rad_s": 2.8,
        "excite_step_s": 0.22,
        "excited_hold_s": 0.65,
        "settle_s": 0.8,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_excited_targets_lean_back_and_shake_base():
    cfg = _cfg_stub()
    args = _args_stub(excite_cycles=2, base_amp_rad=0.2, pitch1_back_rad=0.35)

    targets = _excited_targets(cfg, args, [0.0] * 5)

    assert len(targets) == 6
    assert all(target[1] == pytest.approx(-0.35) for target in targets)
    assert targets[1][0] == pytest.approx(0.2)
    assert targets[2][0] == pytest.approx(-0.2)
    assert targets[-1][0] == pytest.approx(0.0)


def test_build_demo_segments_includes_curious_and_excited_sections():
    cfg = _cfg_stub()
    args = _args_stub(excite_cycles=2)

    segments = build_demo_segments(cfg, args)
    names = [segment.name for segment in segments]

    assert names[:9] == [
        "opening_breath",
        "curious_orient_left",
        "curious_glance_up_left",
        "curious_pause_left",
        "curious_orient_right",
        "curious_glance_up_right",
        "curious_pause_right",
        "curious_recenter",
        "excited_pose_hold_in",
    ]
    shake_segments = [segment for segment in segments if segment.name.startswith("excited_shake_")]
    assert len(shake_segments) == 6
    assert all(segment.action.primitive == PrimitiveKind.MOVE_TO for segment in shake_segments)
    assert all(isinstance(segment.action.command, MoveToCommand) for segment in shake_segments)
    assert names[-1] == "excited_pose_hold_out"


def test_excited_targets_support_pitch_direction_override():
    cfg = _cfg_stub()
    args = _args_stub(excite_cycles=1, pitch1_back_rad=0.25, pitch1_back_sign=1.0)

    targets = _excited_targets(cfg, args, [0.0] * 5)

    assert all(target[1] == pytest.approx(0.25) for target in targets)
