"""Validated pose performances and control-rate sequencing of typed actions."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from pala.config.load import RobotConfig
from typing import Callable

from pala.control.executor import ExecutionStatus, TrajectoryExecutor
from pala.types import ActionPlan, HardwareCommand, HoldCommand, MoveToCommand, PrimitiveKind


@dataclass(frozen=True)
class PerformanceStep:
    name: str
    target_rad: tuple[float, ...]
    rate_deg_s: float
    hold_s: float


@dataclass(frozen=True)
class PerformancePlan:
    plan_id: str
    name: str
    steps: tuple[PerformanceStep, ...]
    end_state: str


@dataclass(frozen=True)
class PerformanceStatus:
    plan_id: str
    name: str
    step: int
    step_name: str
    phase: str
    status: str
    angles_rad: tuple[float, ...]
    timestamp_monotonic_s: float
    reason: str | None = None


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"Invalid {label}")
    return result


def validate_pose(cfg: RobotConfig, pose: tuple[float, ...]) -> None:
    if len(pose) != len(cfg.joint_names):
        raise ValueError("Pose length does not match joint names")
    for name, angle, (lo, hi) in zip(cfg.joint_names, pose, cfg.joint_limits_rad):
        angle = _number(angle, name)
        if not lo - 1e-6 <= angle <= hi + 1e-6:
            raise ValueError(f"{name} target exceeds configured joint limits")
        cal = cfg.servo_calibration["per_joint"][name]
        mapped = math.degrees(angle) * _number(cal["angle_scale"], "scale") + _number(cal["angle_offset"], "offset")
        if not -1e-4 <= mapped <= 180 + 1e-4:
            raise ValueError(f"{name} target would saturate servo mapping")


class PerformanceLibrary:
    def __init__(self, path: str | Path, cfg: RobotConfig) -> None:
        self.path = Path(path)
        self.raw = json.loads(self.path.read_text())
        if self.raw.get("version") != 1 or self.raw.get("joint_names") != cfg.joint_names:
            raise ValueError("Unsupported performance version or joint order")
        self.poses = {}
        for name, values in self.raw["poses_deg"].items():
            if len(values) != len(cfg.joint_names):
                raise ValueError(f"Invalid pose: {name}")
            pose = tuple(math.radians(_number(v, name)) for v in values)
            validate_pose(cfg, pose)
            self.poses[name] = pose
        for required in ("zero", "rest", "attention"):
            if required not in self.poses:
                raise ValueError(f"Missing pose: {required}")
        if any(self.poses["zero"]):
            raise ValueError("Zero pose must be all zeros")
        self.performances = {}
        for name, entry in self.raw["performances"].items():
            steps = []
            for item in entry["steps"]:
                if ("pose" in item) == ("target_deg" in item):
                    raise ValueError("Step requires exactly one pose or target_deg")
                if "pose" in item and item["pose"] not in self.poses:
                    raise ValueError(f"Unknown pose: {item['pose']}")
                if "pose" in item:
                    target = self.poses[item["pose"]]
                else:
                    target = tuple(math.radians(_number(v, "target")) for v in item["target_deg"])
                validate_pose(cfg, target)
                rate = _number(item["rate_deg_s"], "rate", positive=True)
                hold = _number(item.get("hold_s", 0), "hold")
                if hold < 0 or not item["name"]:
                    raise ValueError("Invalid performance step")
                steps.append(PerformanceStep(item["name"], target, rate, hold))
            if not 1 <= len(steps) <= 64 or entry["end_state"] not in self.poses:
                raise ValueError(f"Invalid performance: {name}")
            if steps[-1].target_rad != self.poses[entry["end_state"]]:
                raise ValueError(f"Performance {name} does not finish at its declared pose")
            self.performances[name] = (tuple(steps), entry["end_state"])
        for required in ("startup", "greet", "attend", "settle", "shutdown"):
            if required not in self.performances:
                raise ValueError(f"Missing performance: {required}")

    def plan(self, name: str) -> PerformancePlan:
        if name == "demo":
            steps = self.performances["greet"][0] + self.performances["attend"][0] + self.performances["settle"][0]
            state = "rest"
        else:
            steps, state = self.performances[name]
        return PerformancePlan(uuid4().hex, name, steps, state)


class PerformanceSequencer:
    """Owned only by the control loop; no hardware access, sleeping, or shared globals."""
    def __init__(self, cfg: RobotConfig, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg, self.clock = cfg, clock
        unit_style = {name: 1.0 for name in ("amp_scale", "rate_scale", "duration_scale", "settle_scale")}
        self.executor = TrajectoryExecutor(cfg.joint_limits_rad, style_profiles={"calm": unit_style},
                                           clock=clock, position_tolerance_rad=1e-6)
        self.current = tuple(0.0 for _ in cfg.joint_names)
        self.plan: PerformancePlan | None = None
        self.index = 0
        self.phase = "idle"
        self.phase_start = 0.0
        self.deadline = 0.0
        self.action = ActionPlan(PrimitiveKind.HOLD, HoldCommand(), 1.0)
        self.done = False
        self.failure: str | None = None

    def _move(self, now: float) -> None:
        step = self.plan.steps[self.index]
        seconds = max(abs(a-b) for a, b in zip(step.target_rad, self.current)) / math.radians(step.rate_deg_s) + 2
        self.phase, self.phase_start, self.deadline = "move", now, now + seconds + 0.1
        self.action = ActionPlan(PrimitiveKind.MOVE_TO,
                                MoveToCommand(list(step.target_rad), rate_rad_s=math.radians(step.rate_deg_s), timeout_s=seconds),
                                1.0, explanation=f"{self.plan.name}:{step.name}")

    def _advance(self, now: float) -> None:
        self.index += 1
        if self.index == len(self.plan.steps):
            self.index -= 1
            self.done, self.phase = True, "complete"
            self.action = ActionPlan(PrimitiveKind.HOLD, HoldCommand(), 1.0, explanation=f"{self.plan.name}:complete")
        else:
            self._move(now)

    def tick(self, plan: PerformancePlan | None, dt: float) -> tuple[HardwareCommand, PerformanceStatus | None]:
        now = self.clock()
        if plan is not None and (self.plan is None or plan.plan_id != self.plan.plan_id):
            if self.plan is not None and not self.done and plan.name != "shutdown":
                raise RuntimeError("Cannot interrupt an active performance")
            for step in plan.steps:
                validate_pose(self.cfg, step.target_rad)
            self.plan, self.index, self.done, self.failure = plan, 0, False, None
            self._move(now)
        if self.failure:
            return HardwareCommand(now, list(self.current), False), self.report(now)
        if self.plan is not None and not self.done:
            if self.phase == "hold" and now - self.phase_start >= self.plan.steps[self.index].hold_s:
                self._advance(now)
            elif self.phase == "move" and now >= self.deadline:
                self.failure = "movement deadline exceeded"
        if not self.failure:
            command = self.executor.step(self.action, min(max(dt, 0), 2 / self.cfg.loop_rates.control_hz))
            self.current = tuple(command.joint_angles_rad)
            state = self.executor.control_state
            if state.status in (ExecutionStatus.TIMED_OUT, ExecutionStatus.REJECTED, ExecutionStatus.CANCELED):
                self.failure = f"{state.status.value}: {state.reason}"
            elif self.plan is not None and not self.done and self.phase == "move" and state.status == ExecutionStatus.DONE:
                step = self.plan.steps[self.index]
                if step.hold_s:
                    self.phase, self.phase_start = "hold", now
                    self.action = ActionPlan(PrimitiveKind.HOLD, HoldCommand(), 1.0, explanation=f"{self.plan.name}:{step.name}:hold")
                else:
                    self._advance(now)
        return HardwareCommand(now, list(self.current), not bool(self.failure)), self.report(now)

    def report(self, now: float) -> PerformanceStatus | None:
        if self.plan is None:
            return None
        return PerformanceStatus(self.plan.plan_id, self.plan.name, self.index + 1,
                                 self.plan.steps[self.index].name, self.phase,
                                 "failed" if self.failure else "complete" if self.done else "running",
                                 self.current, now, self.failure)
