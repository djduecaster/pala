from __future__ import annotations

import pytest

from pala.control.executor import ExecutionStatus, TrajectoryExecutor, _normalize_style_profiles
from pala.control.primitives import (
    GazeToCommand,
    GlanceCommand,
    HoldCommand,
    HomeCommand,
    MoveToCommand,
    NodCommand,
    ScanSweepCommand,
    PrimitiveKind,
)
from pala.types import ActionPlan


def _limits(n: int = 5):
    return [[-1.0, 1.0] for _ in range(n)]


def test_move_to_target_length_mismatch_is_rejected():
    executor = TrajectoryExecutor(_limits(5))
    action = ActionPlan(
        primitive=PrimitiveKind.MOVE_TO,
        command=MoveToCommand(target_rad=[0.1], relative=False, rate_rad_s=1.0, timeout_s=1.0),
        confidence=1.0,
    )
    executor.step(action, dt=0.01)
    assert executor.control_state.status == ExecutionStatus.REJECTED
    assert executor.control_state.reason == "move_to target length mismatch"


def test_move_to_timeout_marks_timed_out(monkeypatch):
    now = {"t": 100.0}
    monkeypatch.setattr("pala.control.executor.time.monotonic", lambda: now["t"])
    executor = TrajectoryExecutor(_limits(5))
    action = ActionPlan(
        primitive=PrimitiveKind.MOVE_TO,
        command=MoveToCommand(target_rad=[1.0, 0.0, 0.0, 0.0, 0.0], rate_rad_s=0.01, timeout_s=0.1),
        confidence=1.0,
    )

    executor.step(action, dt=0.01)
    now["t"] = 100.2
    executor.step(action, dt=0.01)
    assert executor.control_state.status == ExecutionStatus.TIMED_OUT
    assert executor.control_state.reason == "timeout"


def test_gaze_to_requires_dwell_before_done(monkeypatch):
    now = {"t": 200.0}
    monkeypatch.setattr("pala.control.executor.time.monotonic", lambda: now["t"])
    executor = TrajectoryExecutor(_limits(5))
    action = ActionPlan(
        primitive=PrimitiveKind.GAZE_TO,
        command=GazeToCommand(yaw_rad=0.0, pitch_rad=0.0, rate_rad_s=10.0, dwell_s=0.2, timeout_s=2.0),
        confidence=1.0,
    )

    executor.step(action, dt=0.01)
    assert executor.control_state.status == ExecutionStatus.RUNNING
    now["t"] = 200.1
    executor.step(action, dt=0.01)
    assert executor.control_state.status == ExecutionStatus.RUNNING
    now["t"] = 200.25
    executor.step(action, dt=0.01)
    assert executor.control_state.status == ExecutionStatus.DONE


def test_same_action_id_not_reactivated_after_terminal(monkeypatch):
    now = {"t": 300.0}
    monkeypatch.setattr("pala.control.executor.time.monotonic", lambda: now["t"])
    executor = TrajectoryExecutor(_limits(5))
    action = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=1.0,
    )
    executor.step(action, dt=0.01)
    assert executor.control_state.status == ExecutionStatus.RUNNING

    executor._finish_active(status=ExecutionStatus.DONE, reason=None)  # noqa: SLF001 - deliberate branch test
    assert executor.control_state.status == ExecutionStatus.DONE
    executor.step(action, dt=0.01)
    assert executor.control_state.status == ExecutionStatus.DONE


def test_normalize_style_profiles_tolerates_bad_entries():
    out = _normalize_style_profiles(
        {
            "curious": {"rate_scale": 2.0, "amp_scale": "not-a-number"},
            " ": {"rate_scale": 5.0},
            "bad": "shape",
        }
    )
    assert out["curious"]["rate_scale"] == 2.0
    assert "bad" not in out or isinstance(out["bad"], dict)
    assert out["calm"]["rate_scale"] > 0


def test_unknown_style_name_falls_back_to_calm_profile():
    executor = TrajectoryExecutor(_limits(5))
    calm = executor._style_profile("calm")  # noqa: SLF001 - explicit fallback check
    unknown = executor._style_profile("does-not-exist")  # noqa: SLF001 - explicit fallback check
    assert unknown == calm


def test_apply_rate_limit_with_negative_dt_clamps_to_zero():
    executor = TrajectoryExecutor(_limits(2))
    action = ActionPlan(
        primitive=PrimitiveKind.MOVE_TO,
        command=MoveToCommand(target_rad=[0.5, -0.5], rate_rad_s=1.0, timeout_s=2.0),
        confidence=1.0,
    )
    cmd = executor.step(action, dt=-1.0)
    assert cmd.joint_angles_rad == [0.0, 0.0]


def test_home_action_executes_home_branch_and_finishes_when_already_at_origin():
    executor = TrajectoryExecutor(_limits(5))
    action = ActionPlan(
        primitive=PrimitiveKind.HOME,
        command=HomeCommand(rate_rad_s=2.0),
        confidence=1.0,
    )
    executor.step(action, dt=0.05)
    assert executor.control_state.status == ExecutionStatus.DONE


def test_relative_move_to_targets_current_offset():
    executor = TrajectoryExecutor(_limits(5))
    executor._current = [0.2, 0.0, 0.0, 0.0, 0.0]  # noqa: SLF001 - targeted branch setup
    action = ActionPlan(
        primitive=PrimitiveKind.MOVE_TO,
        command=MoveToCommand(target_rad=[0.1, 0.0, 0.0, 0.0, 0.0], relative=True, rate_rad_s=10.0, timeout_s=2.0),
        confidence=1.0,
    )
    cmd = executor.step(action, dt=0.05)
    assert cmd.joint_angles_rad[0] == pytest.approx(0.3, abs=1e-6)


def test_nod_primitive_path_runs_and_finishes_by_elapsed_duration(monkeypatch):
    now = {"t": 400.0}
    monkeypatch.setattr("pala.control.executor.time.monotonic", lambda: now["t"])
    executor = TrajectoryExecutor(_limits(5))
    action = ActionPlan(
        primitive=PrimitiveKind.NOD,
        command=NodCommand(amp_rad=0.2, duration_s=0.1, cycles=2, rate_rad_s=10.0),
        confidence=1.0,
    )

    executor.step(action, dt=0.01)
    assert executor.control_state.status == ExecutionStatus.RUNNING
    now["t"] = 400.12
    executor.step(action, dt=0.02)
    assert executor.control_state.status == ExecutionStatus.DONE


def test_command_kind_mismatch_marks_active_action_rejected():
    executor = TrajectoryExecutor(_limits(5))
    bad = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=1.0,
    )
    bad.command = MoveToCommand(target_rad=[0.1, 0.0, 0.0, 0.0, 0.0], relative=False, rate_rad_s=1.0, timeout_s=1.0)
    executor.step(bad, dt=0.01)
    assert executor.control_state.status == ExecutionStatus.REJECTED
    assert executor.control_state.reason == "command-kind mismatch"


def test_gaze_done_resets_reached_time_when_target_is_lost():
    executor = TrajectoryExecutor(_limits(5))
    command = GazeToCommand(yaw_rad=0.0, pitch_rad=0.0, rate_rad_s=1.0, dwell_s=0.2, timeout_s=1.0)
    target = [0.8, 0.0, 0.0, 0.0, 0.0]

    executor._active_reached_s = 10.0  # noqa: SLF001 - branch setup
    executor._current = [0.0, 0.0, 0.0, 0.0, 0.0]  # noqa: SLF001 - branch setup
    assert executor._gaze_done(command, now=11.0, target=target) is False  # noqa: SLF001 - explicit helper test
    assert executor._active_reached_s is None  # noqa: SLF001 - explicit helper test


def test_glance_target_up_and_down_update_pitch_axis():
    executor = TrajectoryExecutor(_limits(5))
    executor._active_base = [0.0, 0.0, 0.0, 0.0, 0.0]  # noqa: SLF001 - direct helper coverage
    style = {"amp_scale": 1.0, "rate_scale": 1.0, "duration_scale": 1.0, "settle_scale": 1.0}
    up = executor._glance_target(GlanceCommand(direction="up", amp_rad=0.3, duration_s=1.0, rate_rad_s=1.0), 0.5, style)  # noqa: SLF001
    down = executor._glance_target(  # noqa: SLF001
        GlanceCommand(direction="down", amp_rad=0.3, duration_s=1.0, rate_rad_s=1.0),
        0.5,
        style,
    )
    assert up[4] < 0.0
    assert down[4] > 0.0


def test_scan_sweep_completes_with_auto_waypoints(monkeypatch):
    now = {"t": 500.0}
    monkeypatch.setattr("pala.control.executor.time.monotonic", lambda: now["t"])

    limits = [
        [-1.57, 1.57],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]
    executor = TrajectoryExecutor(limits)
    action = ActionPlan(
        primitive=PrimitiveKind.SCAN_SWEEP,
        command=ScanSweepCommand(
            positions=0,
            camera_hfov_deg=70.42,
            overlap=0.2,
            dwell_s=0.0,
            rate_rad_s=20.0,
            edge_margin_rad=0.0,
            return_to_center=False,
            timeout_s=4.0,
        ),
        confidence=1.0,
    )

    for _ in range(200):
        executor.step(action, dt=0.02)
        if executor.control_state.status == ExecutionStatus.DONE:
            break
        now["t"] += 0.02

    assert executor.control_state.status == ExecutionStatus.DONE
