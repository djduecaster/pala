import math

from pala.control.executor import TrajectoryExecutor
from pala.types import ActionPlan
from pala.control.primitives import (
    PrimitiveKind,
    HoldCommand,
    MoveToCommand,
    BreathCommand,
    GlanceCommand,
    OrientToZoneCommand,
)


def test_control_clamps_limits():
    limits = [
        [-0.1, 0.1],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]
    executor = TrajectoryExecutor(limits)
    action = ActionPlan(
        primitive=PrimitiveKind.MOVE_TO,
        command=MoveToCommand(target_rad=[-1.0, 0.0, 0.0, 0.0, 0.0], rate_rad_s=10.0),
        confidence=1.0,
    )
    cmd = executor.step(action, dt=1.0)
    assert math.isclose(cmd.joint_angles_rad[0], -0.1, abs_tol=1e-6)


def test_control_preempts_on_new_intent():
    limits = [[-1.0, 1.0] for _ in range(5)]
    executor = TrajectoryExecutor(limits)

    breath = ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command=BreathCommand(amp_rad=0.08, period_s=3.0, rate_rad_s=1.0),
        confidence=1.0,
    )
    move = ActionPlan(
        primitive=PrimitiveKind.MOVE_TO,
        command=MoveToCommand(target_rad=[0.4, 0.0, 0.0, 0.0, 0.0], rate_rad_s=3.0, timeout_s=1.0),
        confidence=1.0,
    )

    executor.step(breath, dt=0.02)
    executor.step(move, dt=0.02)

    assert executor.control_state.active_kind == PrimitiveKind.MOVE_TO


def test_control_same_intent_is_not_reactivated():
    limits = [[-1.0, 1.0] for _ in range(5)]
    executor = TrajectoryExecutor(limits)

    breath = ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command=BreathCommand(amp_rad=0.08, period_s=3.0, rate_rad_s=1.0),
        confidence=1.0,
    )
    breath_same_intent = ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command=BreathCommand(amp_rad=0.08, period_s=3.0, rate_rad_s=1.0),
        confidence=0.4,
    )

    executor.step(breath, dt=0.02)
    started = executor.control_state.started_monotonic_s
    executor.step(breath_same_intent, dt=0.02)

    assert executor.control_state.active_kind == PrimitiveKind.BREATH
    assert executor.control_state.started_monotonic_s == started


def test_control_replaces_active_hold_without_cancel():
    limits = [[-1.0, 1.0] for _ in range(5)]
    executor = TrajectoryExecutor(limits)

    hold = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=1.0,
        cancel_current=False,
    )
    move = ActionPlan(
        primitive=PrimitiveKind.MOVE_TO,
        command=MoveToCommand(target_rad=[0.4, 0.0, 0.0, 0.0, 0.0], rate_rad_s=3.0, timeout_s=1.0),
        confidence=1.0,
        cancel_current=False,
    )

    executor.step(hold, dt=0.02)
    executor.step(move, dt=0.02)

    assert executor.control_state.active_kind == PrimitiveKind.MOVE_TO


def test_control_glance_finishes(monkeypatch):
    fake_time = {"t": 1_000.0}
    monkeypatch.setattr("pala.control.executor.time.monotonic", lambda: fake_time["t"])

    limits = [[-1.0, 1.0] for _ in range(5)]
    executor = TrajectoryExecutor(limits)
    glance = ActionPlan(
        primitive=PrimitiveKind.GLANCE,
        command=GlanceCommand(direction="left", amp_rad=0.3, duration_s=0.05, rate_rad_s=10.0),
        confidence=1.0,
    )

    executor.step(glance, dt=0.02)
    fake_time["t"] += 0.1
    executor.step(glance, dt=0.08)

    assert executor.control_state.status.value == "done"


def test_orient_to_zone_finishes_with_clamped_target():
    limits = [
        [-0.2, 0.2],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]
    executor = TrajectoryExecutor(limits)
    orient = ActionPlan(
        primitive=PrimitiveKind.ORIENT_TO_ZONE,
        command=OrientToZoneCommand(zone="right", amp_rad=1.0, rate_rad_s=2.0),
        confidence=1.0,
    )

    for _ in range(100):
        executor.step(orient, dt=0.02)
        if executor.control_state.status.value == "done":
            break

    assert executor.control_state.status.value == "done"
