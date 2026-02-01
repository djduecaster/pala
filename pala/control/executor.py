from __future__ import annotations

import math
import time
from typing import List, Optional

from ..types import ActionPlan, HardwareCommand
from .primitives import (
    PRIMITIVE_HOLD,
    PRIMITIVE_GLANCE_LEFT,
    PRIMITIVE_GLANCE_RIGHT,
    PRIMITIVE_ACKNOWLEDGE,
    PRIMITIVE_BREATH,
)


class TrajectoryExecutor:
    """Minimal trajectory executor.

    TODO: Port kinematics primitives from ../pala_old/pala_project/src/kinematics/*
    into pala/control/* and replace this executor.
    """

    def __init__(self, joint_limits_rad: List[List[float]]):
        self._limits = joint_limits_rad
        self._current = [0.0 for _ in joint_limits_rad]
        self._target = list(self._current)
        self._action_sig: Optional[str] = None
        self._action_start = time.monotonic()

    def set_action(self, action: ActionPlan) -> None:
        sig = f"{action.primitive}:{sorted(action.params.items())}"
        if sig != self._action_sig:
            self._action_sig = sig
            self._action_start = time.monotonic()

    def step(self, action: ActionPlan, dt: float) -> HardwareCommand:
        self.set_action(action)
        now = time.monotonic()

        # Base target is current pose unless primitive changes it
        target = list(self._current)

        if action.primitive == PRIMITIVE_GLANCE_LEFT:
            target[0] += -0.35
        elif action.primitive == PRIMITIVE_GLANCE_RIGHT:
            target[0] += 0.35
        elif action.primitive == PRIMITIVE_ACKNOWLEDGE:
            amp = float(action.params.get("amp_rad", 0.2))
            dur = float(action.params.get("duration_s", 0.4))
            t = min(max(0.0, now - self._action_start), dur)
            if dur > 0:
                target[4] += amp * math.sin(math.pi * (t / dur))
        elif action.primitive == PRIMITIVE_BREATH:
            amp = float(action.params.get("amp_rad", 0.1))
            period = float(action.params.get("period_s", 6.0))
            phase = (now - self._action_start) * (2.0 * math.pi / max(0.1, period))
            target[1] += amp * math.sin(phase)
            target[4] += 0.6 * amp * math.sin(phase)
        elif action.primitive == PRIMITIVE_HOLD:
            pass

        # Apply rate limit (simple first-order)
        rate = float(action.params.get("rate_rad_s", 1.5))
        max_step = max(0.0, rate * max(0.0, dt))
        for i in range(len(target)):
            delta = target[i] - self._current[i]
            if delta > max_step:
                delta = max_step
            elif delta < -max_step:
                delta = -max_step
            self._current[i] += delta

        self._current = _clamp(self._current, self._limits)
        self._target = list(target)

        return HardwareCommand(
            timestamp_monotonic_s=now,
            joint_angles_rad=list(self._current),
            enable=True,
        )


def _clamp(vals: List[float], limits: List[List[float]]) -> List[float]:
    out = []
    for v, lim in zip(vals, limits):
        lo, hi = float(lim[0]), float(lim[1])
        out.append(max(lo, min(hi, v)))
    return out
