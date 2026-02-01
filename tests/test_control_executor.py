import math

from pala.control.executor import TrajectoryExecutor
from pala.types import ActionPlan
from pala.control.primitives import PRIMITIVE_GLANCE_LEFT


def test_control_clamps_limits():
    limits = [
        [-0.1, 0.1],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]
    executor = TrajectoryExecutor(limits)
    action = ActionPlan(primitive=PRIMITIVE_GLANCE_LEFT, params={"rate_rad_s": 10.0}, confidence=1.0)
    cmd = executor.step(action, dt=1.0)
    assert math.isclose(cmd.joint_angles_rad[0], -0.1, abs_tol=1e-6)
