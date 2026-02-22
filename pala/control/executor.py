from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Dict, List, Optional

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
from ..types.style_profiles import default_style_profiles


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
    """Typed primitive executor where actions run to completion unless canceled."""

    def __init__(
        self,
        joint_limits_rad: List[List[float]],
        *,
        style_profiles: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self._limits = joint_limits_rad
        self._current = [0.0 for _ in joint_limits_rad]
        self._target = list(self._current)
        self._style_profiles = _normalize_style_profiles(style_profiles)

        self._active_action: Optional[ActionPlan] = None
        self._active_start_s: Optional[float] = None
        self._active_base: List[float] = list(self._current)
        self._active_target: Optional[List[float]] = None
        self._active_timeout_s: Optional[float] = None
        self._active_reached_s: Optional[float] = None
        self._last_terminal_action_id: Optional[str] = None
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

        if self._active_action is None and action.action_id != self._last_terminal_action_id:
            self._activate(action, now)

        target = list(self._current)
        rate = 1.5
        done = False

        if self._active_action is None:
            self._target = list(self._current)
            return HardwareCommand(
                timestamp_monotonic_s=now,
                joint_angles_rad=list(self._current),
                enable=True,
            )

        kind = self._active_action.primitive
        command = self._active_action.command
        style = self._style_profile(self._active_action.style)
        elapsed = max(0.0, now - (self._active_start_s or now))

        if kind == PrimitiveKind.HOLD and isinstance(command, HoldCommand):
            target = list(self._current)
        elif kind == PrimitiveKind.HOME and isinstance(command, HomeCommand):
            target = list(self._active_target or self._current)
            rate = command.rate_rad_s * style["rate_scale"]
        elif kind == PrimitiveKind.MOVE_TO and isinstance(command, MoveToCommand):
            target = list(self._active_target or self._current)
            rate = command.rate_rad_s * style["rate_scale"]
        elif kind == PrimitiveKind.GAZE_TO and isinstance(command, GazeToCommand):
            target = list(self._active_target or self._current)
            rate = command.rate_rad_s * style["rate_scale"]
        elif kind == PrimitiveKind.ORIENT_TO_ZONE and isinstance(command, OrientToZoneCommand):
            target = list(self._active_target or self._current)
            rate = command.rate_rad_s * style["rate_scale"]
        elif kind == PrimitiveKind.GLANCE and isinstance(command, GlanceCommand):
            target = self._glance_target(command, elapsed, style)
            rate = command.rate_rad_s * style["rate_scale"]
            done = elapsed >= max(0.01, command.duration_s * style["duration_scale"])
        elif kind == PrimitiveKind.NOD and isinstance(command, NodCommand):
            target = self._nod_target(command, elapsed, style)
            rate = command.rate_rad_s * style["rate_scale"]
            done = elapsed >= max(0.01, command.duration_s * style["duration_scale"])
        elif kind == PrimitiveKind.BREATH and isinstance(command, BreathCommand):
            target = self._breath_target(command, elapsed, style)
            rate = command.rate_rad_s * style["rate_scale"]
        else:
            self._finish_active(
                status=ExecutionStatus.REJECTED,
                reason="command-kind mismatch",
            )
            target = list(self._current)

        self._target = _clamp(target, self._limits)
        # Completion checks must use feasible (clamped) targets so out-of-limit
        # requests do not leave primitives running indefinitely.
        if kind in {PrimitiveKind.HOME, PrimitiveKind.MOVE_TO, PrimitiveKind.ORIENT_TO_ZONE}:
            done = self._within_tol(self._target, self._current)
        elif kind == PrimitiveKind.GAZE_TO and isinstance(command, GazeToCommand):
            done = self._gaze_done(command, now, self._target)
        self._apply_rate_limit(self._target, rate, dt)
        self._current = _clamp(self._current, self._limits)

        if done:
            self._finish_active(status=ExecutionStatus.DONE, reason=None)

        return HardwareCommand(
            timestamp_monotonic_s=now,
            joint_angles_rad=list(self._current),
            enable=True,
        )

    def _maybe_activate(self, request: ActionPlan, now: float) -> None:
        if self._active_action is None:
            return
        if request.action_id == self._active_action.action_id:
            return
        # Latest-intent arbitration: any new, different action intent replaces
        # the active primitive. Equivalent intents are ignored to avoid
        # unnecessary re-activation churn.
        if self._same_intent(self._active_action, request):
            return
        self._finish_active(
            status=ExecutionStatus.CANCELED,
            reason=f"replaced_by:{request.action_id}",
        )
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
                self._finish_active(
                    status=ExecutionStatus.REJECTED,
                    reason="move_to target length mismatch",
                )
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
        self._finish_active(status=ExecutionStatus.TIMED_OUT, reason="timeout")

    def _gaze_done(self, command: GazeToCommand, now: float, target: List[float]) -> bool:
        reached = self._within_tol(target, self._current)
        if not reached:
            self._active_reached_s = None
            return False
        if self._active_reached_s is None:
            self._active_reached_s = now
        return (now - self._active_reached_s) >= max(0.0, command.dwell_s)

    def _glance_target(self, command: GlanceCommand, elapsed: float, style: Dict[str, float]) -> List[float]:
        dur = max(0.01, command.duration_s * style["duration_scale"])
        t = min(max(0.0, elapsed), dur)
        x = t / dur
        # Three-phase trajectory: anticipation -> accent -> settle.
        if x < 0.2:
            phase = -0.25 * math.sin(math.pi * (x / 0.2))
        elif x < 0.7:
            phase = math.sin(math.pi * ((x - 0.2) / 0.5))
        else:
            phase = 0.35 * math.sin(math.pi * ((1.0 - x) / 0.3))
        amp = command.amp_rad * style["amp_scale"]
        target = list(self._active_base)
        if command.direction == "left" and len(target) > 0:
            target[0] -= amp * phase
        elif command.direction == "right" and len(target) > 0:
            target[0] += amp * phase
        elif command.direction == "up" and len(target) > 4:
            target[4] -= amp * phase
        elif command.direction == "down" and len(target) > 4:
            target[4] += amp * phase
        return target

    def _nod_target(self, command: NodCommand, elapsed: float, style: Dict[str, float]) -> List[float]:
        dur = max(0.01, command.duration_s * style["duration_scale"])
        t = min(elapsed, dur)
        envelope = math.sin(math.pi * (t / dur)) ** max(1.0, style["settle_scale"])
        phase = 2.0 * math.pi * max(1, command.cycles) * (t / dur)
        target = list(self._active_base)
        if len(target) > 4:
            target[4] += command.amp_rad * style["amp_scale"] * envelope * math.sin(phase)
        return target

    def _breath_target(self, command: BreathCommand, elapsed: float, style: Dict[str, float]) -> List[float]:
        period = max(0.1, command.period_s * style["duration_scale"])
        phase = elapsed * (2.0 * math.pi / period)
        target = list(self._active_base)
        amp = command.amp_rad * style["amp_scale"]
        if len(target) > 1:
            target[1] += amp * math.sin(phase)
        if len(target) > 2:
            target[2] += 0.35 * amp * math.sin(phase + 0.35)
        if len(target) > 4:
            target[4] += 0.6 * amp * math.sin(phase)
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

    def _finish_active(self, *, status: ExecutionStatus, reason: Optional[str]) -> None:
        terminal_id = None if self._active_action is None else self._active_action.action_id
        self._state.status = status
        self._state.reason = reason
        self._state.active_kind = None
        self._active_action = None
        self._active_start_s = None
        self._active_target = None
        self._active_timeout_s = None
        self._active_reached_s = None
        if terminal_id is not None:
            self._last_terminal_action_id = terminal_id

    def _style_profile(self, style_name: str) -> Dict[str, float]:
        key = str(style_name).strip().lower()
        return self._style_profiles.get(key, self._style_profiles["calm"])

    @staticmethod
    def _same_intent(active: ActionPlan, request: ActionPlan) -> bool:
        return (
            active.primitive == request.primitive
            and active.command == request.command
            and active.style == request.style
        )


def _clamp(vals: List[float], limits: List[List[float]]) -> List[float]:
    out = []
    for v, lim in zip(vals, limits):
        lo, hi = float(lim[0]), float(lim[1])
        out.append(max(lo, min(hi, v)))
    return out


def _normalize_style_profiles(raw: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    defaults = default_style_profiles()
    if not isinstance(raw, dict):
        return defaults

    out = {name: dict(vals) for name, vals in defaults.items()}
    for name, vals in raw.items():
        if not isinstance(vals, dict):
            continue
        key = str(name).strip().lower()
        if not key:
            continue
        merged = dict(out.get(key, defaults["calm"]))
        for param in ("amp_scale", "rate_scale", "duration_scale", "settle_scale"):
            try:
                if param in vals:
                    merged[param] = float(vals[param])
            except Exception:
                continue
        out[key] = merged
    return out
