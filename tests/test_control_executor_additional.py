from __future__ import annotations

import pytest

from pala.control.executor import ExecutionStatus, TrajectoryExecutor, _normalize_style_profiles
from pala.control.primitives import GazeToCommand, HoldCommand, MoveToCommand, PrimitiveKind
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
