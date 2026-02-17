import math
import time

from pala.control.executor import TrajectoryExecutor
from pala.types import ActionPlan
from pala.control.primitives import (
    PrimitiveKind,
    MoveToCommand,
    BreathCommand,
    GlanceCommand,
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


def test_control_breath_preempted_by_move_to():
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


def test_control_glance_finishes():
    limits = [[-1.0, 1.0] for _ in range(5)]
    executor = TrajectoryExecutor(limits)
    glance = ActionPlan(
        primitive=PrimitiveKind.GLANCE,
        command=GlanceCommand(direction="left", amp_rad=0.3, duration_s=0.05, rate_rad_s=10.0),
        confidence=1.0,
    )

    executor.step(glance, dt=0.02)
    time.sleep(0.06)
    executor.step(glance, dt=0.08)

    assert executor.control_state.status.value == "done"
