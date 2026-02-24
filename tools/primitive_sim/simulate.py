from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Optional, Sequence

import pala.control.executor as executor_module
from pala.control import TrajectoryExecutor
from pala.control.executor import ExecutionStatus
from pala.types import (
    ActionPlan,
    BreathCommand,
    GazeToCommand,
    GlanceCommand,
    HoldCommand,
    HomeCommand,
    MoveToCommand,
    NodCommand,
    OrientToZoneCommand,
    PrimitiveKind,
    action_plan_from_dict,
    to_json_dict,
)

TERMINAL_STATUSES = {
    ExecutionStatus.DONE,
    ExecutionStatus.CANCELED,
    ExecutionStatus.TIMED_OUT,
    ExecutionStatus.REJECTED,
}
CONTINUOUS_PRIMITIVES = {PrimitiveKind.HOLD, PrimitiveKind.BREATH}


@dataclass(frozen=True)
class SimSegment:
    name: str
    action: ActionPlan
    max_s: float
    stop_on_done: bool


class _SimClock:
    def __init__(self, start_s: float = 0.0) -> None:
        self._t = float(start_s)

    def now(self) -> float:
        return self._t

    def advance(self, dt_s: float) -> None:
        self._t += max(0.0, float(dt_s))


@contextmanager
def _patched_executor_clock(clock: _SimClock) -> Iterator[None]:
    original = executor_module.time.monotonic
    executor_module.time.monotonic = clock.now
    try:
        yield
    finally:
        executor_module.time.monotonic = original


def simulate_segments(
    *,
    joint_names: Sequence[str],
    joint_limits_rad: Sequence[Sequence[float]],
    segments: Sequence[SimSegment],
    hz: float,
    style_profiles: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, Any]:
    rate_hz = max(1.0, float(hz))
    dt_s = 1.0 / rate_hz

    limits = [[float(l[0]), float(l[1])] for l in joint_limits_rad]
    names = [str(n) for n in joint_names]
    if len(names) != len(limits):
        raise ValueError("joint_names and joint_limits_rad must have matching lengths")

    summary_status_counts: dict[str, int] = {}
    min_by_joint = [float("inf") for _ in names]
    max_by_joint = [float("-inf") for _ in names]
    peak_vel_by_joint = [0.0 for _ in names]

    samples: list[dict[str, Any]] = []
    segment_summaries: list[dict[str, Any]] = []

    prev_t: Optional[float] = None
    prev_angles: Optional[list[float]] = None

    # Executor elapsed math uses a truthy check on start time, so avoid 0.0.
    clock = _SimClock(start_s=1000.0)
    with _patched_executor_clock(clock):
        executor = TrajectoryExecutor(limits, style_profiles=style_profiles)

        for seg in segments:
            start_t = clock.now()
            first_tick = True
            ticks = 0

            while (clock.now() - start_t) < max(0.01, float(seg.max_s)):
                request = seg.action if not first_tick else replace(seg.action, cancel_current=True)
                cmd = executor.step(request, dt_s)
                state = executor.control_state
                now_t = clock.now()
                angles = [float(v) for v in cmd.joint_angles_rad]

                for i, a in enumerate(angles):
                    min_by_joint[i] = min(min_by_joint[i], a)
                    max_by_joint[i] = max(max_by_joint[i], a)

                if prev_t is not None and prev_angles is not None:
                    actual_dt = max(1e-6, now_t - prev_t)
                    for i, a in enumerate(angles):
                        vel = abs((a - prev_angles[i]) / actual_dt)
                        peak_vel_by_joint[i] = max(peak_vel_by_joint[i], vel)

                prev_t = now_t
                prev_angles = list(angles)

                key = state.status.value
                summary_status_counts[key] = summary_status_counts.get(key, 0) + 1

                samples.append(
                    {
                        "t_s": now_t,
                        "segment": seg.name,
                        "request_primitive": request.primitive.value,
                        "active_primitive": None if state.active_kind is None else state.active_kind.value,
                        "status": key,
                        "reason": state.reason,
                        "action_id": request.action_id,
                        "joint_angles_rad": angles,
                        "enable": bool(cmd.enable),
                    }
                )
                ticks += 1
                first_tick = False

                should_stop = bool(seg.stop_on_done) and state.status in TERMINAL_STATUSES
                clock.advance(dt_s)
                if should_stop:
                    break

            end_t = clock.now()
            segment_summaries.append(
                {
                    "name": seg.name,
                    "ticks": ticks,
                    "elapsed_s": max(0.0, end_t - start_t),
                    "final_status": executor.control_state.status.value,
                    "final_reason": executor.control_state.reason,
                }
            )

    joint_stats: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        lo = 0.0 if min_by_joint[i] == float("inf") else min_by_joint[i]
        hi = 0.0 if max_by_joint[i] == float("-inf") else max_by_joint[i]
        joint_stats.append(
            {
                "joint": name,
                "min_rad": lo,
                "max_rad": hi,
                "peak_vel_rad_s": peak_vel_by_joint[i],
            }
        )

    return {
        "metadata": {
            "generated_wall_s": time.time(),
            "hz": rate_hz,
            "dt_s": dt_s,
            "joint_names": names,
            "joint_limits_rad": limits,
            "segment_count": len(segments),
        },
        "segments": [
            {
                "name": seg.name,
                "max_s": float(seg.max_s),
                "stop_on_done": bool(seg.stop_on_done),
                "action": to_json_dict(seg.action),
            }
            for seg in segments
        ],
        "samples": samples,
        "summary": {
            "status_counts": summary_status_counts,
            "joint_stats": joint_stats,
        },
    }


def write_trace_json(path: Path, trace: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def build_suite_segments(*, suite_breath_s: float = 6.0, style: str = "calm") -> list[SimSegment]:
    return [
        SimSegment(
            "home",
            ActionPlan(PrimitiveKind.HOME, HomeCommand(rate_rad_s=1.2), 1.0, "sim_suite", style=style),
            max_s=2.0,
            stop_on_done=True,
        ),
        SimSegment(
            "breath",
            ActionPlan(
                PrimitiveKind.BREATH,
                BreathCommand(amp_rad=0.08, period_s=6.5, rate_rad_s=1.0),
                1.0,
                "sim_suite",
                style=style,
            ),
            max_s=max(1.0, float(suite_breath_s)),
            stop_on_done=False,
        ),
        SimSegment(
            "glance_left",
            ActionPlan(
                PrimitiveKind.GLANCE,
                GlanceCommand(direction="left", amp_rad=0.26, duration_s=0.7, rate_rad_s=1.6),
                1.0,
                "sim_suite",
                style=style,
            ),
            max_s=1.3,
            stop_on_done=True,
        ),
        SimSegment(
            "glance_right",
            ActionPlan(
                PrimitiveKind.GLANCE,
                GlanceCommand(direction="right", amp_rad=0.26, duration_s=0.7, rate_rad_s=1.6),
                1.0,
                "sim_suite",
                style=style,
            ),
            max_s=1.3,
            stop_on_done=True,
        ),
        SimSegment(
            "nod",
            ActionPlan(
                PrimitiveKind.NOD,
                NodCommand(amp_rad=0.2, duration_s=0.9, cycles=1, rate_rad_s=1.8),
                1.0,
                "sim_suite",
                style=style,
            ),
            max_s=1.5,
            stop_on_done=True,
        ),
        SimSegment(
            "orient_left",
            ActionPlan(
                PrimitiveKind.ORIENT_TO_ZONE,
                OrientToZoneCommand(zone="left", amp_rad=0.22, rate_rad_s=1.3),
                1.0,
                "sim_suite",
                style=style,
            ),
            max_s=1.7,
            stop_on_done=True,
        ),
        SimSegment(
            "orient_center",
            ActionPlan(
                PrimitiveKind.ORIENT_TO_ZONE,
                OrientToZoneCommand(zone="center", amp_rad=0.22, rate_rad_s=1.3),
                1.0,
                "sim_suite",
                style=style,
            ),
            max_s=1.7,
            stop_on_done=True,
        ),
        SimSegment(
            "gaze_to",
            ActionPlan(
                PrimitiveKind.GAZE_TO,
                GazeToCommand(yaw_rad=0.12, pitch_rad=-0.08, rate_rad_s=1.4, dwell_s=0.2, timeout_s=1.6),
                1.0,
                "sim_suite",
                style=style,
            ),
            max_s=2.0,
            stop_on_done=True,
        ),
    ]


def load_segments_from_json(path: Path) -> list[SimSegment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload: Any = raw
    if isinstance(payload, Mapping):
        payload = payload.get("segments")
    if not isinstance(payload, list):
        raise ValueError("script JSON must be a list or an object with a 'segments' list")

    out: list[SimSegment] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"segment[{idx}] must be an object")

        action_payload: Any = item.get("action", item)
        if not isinstance(action_payload, Mapping):
            raise ValueError(f"segment[{idx}] action must be an object")

        action = action_plan_from_dict(action_payload)
        if action is None:
            raise ValueError(f"segment[{idx}] contains an invalid action payload")

        name = str(item.get("name") or f"segment_{idx+1:02d}_{action.primitive.value}").strip()
        if not name:
            name = f"segment_{idx+1:02d}_{action.primitive.value}"

        max_s = _as_float(item.get("max_s"), _default_segment_duration(action))
        stop_default = action.primitive not in CONTINUOUS_PRIMITIVES
        stop_on_done = _as_bool(item.get("stop_on_done"), stop_default)

        out.append(
            SimSegment(
                name=name,
                action=action,
                max_s=max(0.01, max_s),
                stop_on_done=stop_on_done,
            )
        )
    return out


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on", "y"}:
            return True
        if token in {"0", "false", "no", "off", "n", ""}:
            return False
        return bool(default)
    return bool(value)


def _default_segment_duration(action: ActionPlan) -> float:
    command = action.command
    if action.primitive in CONTINUOUS_PRIMITIVES:
        return 4.0
    if isinstance(command, (GlanceCommand, NodCommand)):
        return max(0.5, float(command.duration_s) * 1.5)
    if isinstance(command, (GazeToCommand, MoveToCommand)):
        return max(0.5, float(command.timeout_s) * 1.2)
    if isinstance(command, HomeCommand):
        return 2.0
    if isinstance(command, OrientToZoneCommand):
        return 1.8
    if isinstance(command, HoldCommand):
        return 2.0
    return 2.0
