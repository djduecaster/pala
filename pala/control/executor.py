from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import List, Optional

from ..types import (
    ActionPlan,
    HardwareCommand,
    PrimitiveKind,
    PrimitiveCommand,
    HoldCommand,
    HomeCommand,
    MoveToCommand,
    GazeToCommand,
    GlanceCommand,
    NodCommand,
    BreathCommand,
    OrientToZoneCommand,
)


class ExecutionStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    CANCELED = "canceled"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"


@dataclass
class ControlState:
    active_kind: Optional[PrimitiveKind]
    started_monotonic_s: Optional[float]
    status: ExecutionStatus
    reason: Optional[str] = None


class TrajectoryExecutor:
    """Typed primitive executor with preemption, completion, and safety clamping."""

    _PRIORITY = {
        PrimitiveKind.HOLD: 100,
        PrimitiveKind.MOVE_TO: 90,
        PrimitiveKind.GAZE_TO: 90,
        PrimitiveKind.HOME: 90,
        PrimitiveKind.NOD: 80,
        PrimitiveKind.GLANCE: 80,
        PrimitiveKind.ORIENT_TO_ZONE: 80,
        PrimitiveKind.BREATH: 10,
    }

    def __init__(self, joint_limits_rad: List[List[float]]):
        self._limits = joint_limits_rad
        self._current = [0.0 for _ in joint_limits_rad]
        self._target = list(self._current)

        self._active_action: Optional[ActionPlan] = None
        self._active_start_s: Optional[float] = None
        self._active_base: List[float] = list(self._current)
        self._active_target: Optional[List[float]] = None
        self._active_timeout_s: Optional[float] = None
        self._active_reached_s: Optional[float] = None
        self._state = ControlState(
            active_kind=None,
            started_monotonic_s=None,
            status=ExecutionStatus.DONE,
        )

    @property
    def control_state(self) -> ControlState:
        return self._state

    def step(self, action: ActionPlan, dt: float) -> HardwareCommand:
        now = time.monotonic()
        self._maybe_activate(action, now)
        self._apply_timeout(now)

        target = list(self._current)
        rate = 1.5
        done = False

        if self._active_action is None:
            self._activate(ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=1.0), now)

        assert self._active_action is not None
        kind = self._active_action.primitive
        command = self._active_action.command
        elapsed = max(0.0, now - (self._active_start_s or now))

        if kind == PrimitiveKind.HOLD and isinstance(command, HoldCommand):
            target = list(self._current)
        elif kind == PrimitiveKind.HOME and isinstance(command, HomeCommand):
            target = list(self._active_target or self._current)
            rate = command.rate_rad_s
            done = self._within_tol(target, self._current)
        elif kind == PrimitiveKind.MOVE_TO and isinstance(command, MoveToCommand):
            target = list(self._active_target or self._current)
            rate = command.rate_rad_s
            done = self._within_tol(target, self._current)
        elif kind == PrimitiveKind.GAZE_TO and isinstance(command, GazeToCommand):
            target = list(self._active_target or self._current)
            rate = command.rate_rad_s
            done = self._gaze_done(command, now, target)
        elif kind == PrimitiveKind.ORIENT_TO_ZONE and isinstance(command, OrientToZoneCommand):
            target = list(self._active_target or self._current)
            rate = command.rate_rad_s
            done = self._within_tol(target, self._current)
        elif kind == PrimitiveKind.GLANCE and isinstance(command, GlanceCommand):
            target = self._glance_target(command, elapsed)
            rate = command.rate_rad_s
            done = elapsed >= max(0.01, command.duration_s)
        elif kind == PrimitiveKind.NOD and isinstance(command, NodCommand):
            target = self._nod_target(command, elapsed)
            rate = command.rate_rad_s
            done = elapsed >= max(0.01, command.duration_s)
        elif kind == PrimitiveKind.BREATH and isinstance(command, BreathCommand):
            target = self._breath_target(command, elapsed)
            rate = command.rate_rad_s
        else:
            self._state.status = ExecutionStatus.REJECTED
            self._state.reason = "command-kind mismatch"
            self._activate(ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=0.0), now)
            target = list(self._current)

        self._target = _clamp(target, self._limits)
        self._apply_rate_limit(self._target, rate, dt)
        self._current = _clamp(self._current, self._limits)

        if done:
            self._state.status = ExecutionStatus.DONE
            self._state.reason = None
            self._active_action = None
            self._active_start_s = None
            self._active_target = None
            self._active_timeout_s = None
            self._active_reached_s = None

        return HardwareCommand(
            timestamp_monotonic_s=now,
            joint_angles_rad=list(self._current),
            enable=True,
        )

    def _maybe_activate(self, request: ActionPlan, now: float) -> None:
        if self._active_action is None:
            self._activate(request, now)
            return
        if request == self._active_action:
            return

        current_priority = self._PRIORITY[self._active_action.primitive]
        incoming_priority = self._PRIORITY[request.primitive]
        if incoming_priority >= current_priority:
            self._state.status = ExecutionStatus.CANCELED
            self._state.reason = "preempted"
            self._activate(request, now)

    def _activate(self, action: ActionPlan, now: float) -> None:
        self._active_action = action
        self._active_start_s = now
        self._active_base = list(self._current)
        self._active_target = None
        self._active_timeout_s = None
        self._active_reached_s = None
        self._state = ControlState(
            active_kind=action.primitive,
            started_monotonic_s=now,
            status=ExecutionStatus.RUNNING,
        )

        command = action.command
        if action.primitive == PrimitiveKind.HOME and isinstance(command, HomeCommand):
            self._active_target = [0.0 for _ in self._current]
        elif action.primitive == PrimitiveKind.MOVE_TO and isinstance(command, MoveToCommand):
            if len(command.target_rad) != len(self._current):
                self._state.status = ExecutionStatus.REJECTED
                self._state.reason = "move_to target length mismatch"
                self._active_action = ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=0.0)
                self._active_start_s = now
                self._active_target = None
                self._active_timeout_s = None
                return
            if command.relative:
                self._active_target = [c + d for c, d in zip(self._current, command.target_rad)]
            else:
                self._active_target = list(command.target_rad)
            self._active_timeout_s = max(0.1, command.timeout_s)
        elif action.primitive == PrimitiveKind.GAZE_TO and isinstance(command, GazeToCommand):
            target = list(self._current)
            if len(target) > 0:
                target[0] = command.yaw_rad
            if len(target) > 4:
                target[4] = command.pitch_rad
            self._active_target = target
            self._active_timeout_s = max(0.1, command.timeout_s)
        elif action.primitive == PrimitiveKind.ORIENT_TO_ZONE and isinstance(command, OrientToZoneCommand):
            zone_to_yaw = {"left": -command.amp_rad, "center": 0.0, "right": command.amp_rad}
            target = list(self._current)
            if len(target) > 0:
                target[0] = zone_to_yaw[command.zone]
            self._active_target = target

    def _apply_timeout(self, now: float) -> None:
        if self._active_action is None or self._active_timeout_s is None or self._active_start_s is None:
            return
        if (now - self._active_start_s) <= self._active_timeout_s:
            return
        self._state.status = ExecutionStatus.TIMED_OUT
        self._state.reason = "timeout"
        self._activate(ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=0.0), now)

    def _gaze_done(self, command: GazeToCommand, now: float, target: List[float]) -> bool:
        reached = self._within_tol(target, self._current)
        if not reached:
            self._active_reached_s = None
            return False
        if self._active_reached_s is None:
            self._active_reached_s = now
        return (now - self._active_reached_s) >= max(0.0, command.dwell_s)

    def _glance_target(self, command: GlanceCommand, elapsed: float) -> List[float]:
        dur = max(0.01, command.duration_s)
        phase = math.sin(math.pi * min(elapsed, dur) / dur)
        target = list(self._active_base)
        if command.direction == "left" and len(target) > 0:
            target[0] -= command.amp_rad * phase
        elif command.direction == "right" and len(target) > 0:
            target[0] += command.amp_rad * phase
        elif command.direction == "up" and len(target) > 4:
            target[4] -= command.amp_rad * phase
        elif command.direction == "down" and len(target) > 4:
            target[4] += command.amp_rad * phase
        return target

    def _nod_target(self, command: NodCommand, elapsed: float) -> List[float]:
        dur = max(0.01, command.duration_s)
        t = min(elapsed, dur)
        envelope = math.sin(math.pi * (t / dur))
        phase = 2.0 * math.pi * max(1, command.cycles) * (t / dur)
        target = list(self._active_base)
        if len(target) > 4:
            target[4] += command.amp_rad * envelope * math.sin(phase)
        return target

    def _breath_target(self, command: BreathCommand, elapsed: float) -> List[float]:
        period = max(0.1, command.period_s)
        phase = elapsed * (2.0 * math.pi / period)
        target = list(self._active_base)
        if len(target) > 1:
            target[1] += command.amp_rad * math.sin(phase)
        if len(target) > 4:
            target[4] += 0.6 * command.amp_rad * math.sin(phase)
        return target

    def _apply_rate_limit(self, target: List[float], rate_rad_s: float, dt: float) -> None:
        max_step = max(0.0, float(rate_rad_s) * max(0.0, dt))
        for i in range(len(target)):
            delta = target[i] - self._current[i]
            if delta > max_step:
                delta = max_step
            elif delta < -max_step:
                delta = -max_step
            self._current[i] += delta

    @staticmethod
    def _within_tol(target: List[float], current: List[float], tol: float = 0.02) -> bool:
        return all(abs(t - c) <= tol for t, c in zip(target, current))


def _clamp(vals: List[float], limits: List[List[float]]) -> List[float]:
    out = []
    for v, lim in zip(vals, limits):
        lo, hi = float(lim[0]), float(lim[1])
        out.append(max(lo, min(hi, v)))
    return out

