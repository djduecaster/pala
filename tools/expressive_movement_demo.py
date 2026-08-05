#!/usr/bin/env python3
"""Run a short choreographed expressive movement demo on dummy or real servo hardware."""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
from dataclasses import dataclass, replace
from typing import Optional, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pala.config import load_config
from pala.control import TrajectoryExecutor
from pala.control.executor import ExecutionStatus
from pala.control.primitives import (
    BreathCommand,
    GlanceCommand,
    HoldCommand,
    HomeCommand,
    MoveToCommand,
    OrientToZoneCommand,
    PrimitiveKind,
)
from pala.main import _build_servo
from pala.types import ActionPlan

logger = logging.getLogger(__name__)

_DONE_STATES = {
    ExecutionStatus.DONE,
    ExecutionStatus.TIMED_OUT,
    ExecutionStatus.REJECTED,
}


@dataclass(frozen=True)
class Segment:
    name: str
    action: ActionPlan
    max_s: float
    stop_on_done: bool


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short expressive motion demo")
    parser.add_argument("--config", default="config/robot.yaml", help="Config path")
    parser.add_argument("--runtime-mode", default=None, help="Optional mode override (for example jetson_full)")
    parser.add_argument("--enable", action="store_true", help="Required to actuate hardware in jetson_full mode")
    parser.add_argument("--dry-run", action="store_true", help="Run the control executor without sending servo commands")
    parser.add_argument("--keep-enabled", action="store_true", help="Do not disable servos on exit")
    parser.add_argument("--hz", type=float, default=0.0, help="Control update rate (0 uses config control_hz)")
    parser.add_argument("--status-every-s", type=float, default=0.5, help="Status logging cadence")
    parser.add_argument("--neutral-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neutral-end", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--breath-s", type=float, default=4.0, help="Opening breath duration")
    parser.add_argument("--breath-amp-rad", type=float, default=0.11, help="Opening breath amplitude")
    parser.add_argument("--breath-period-s", type=float, default=5.2, help="Opening breath period")
    parser.add_argument("--curious-glance-amp-rad", type=float, default=0.24, help="Curious glance amplitude")
    parser.add_argument("--curious-glance-duration-s", type=float, default=0.85, help="Curious glance duration")
    parser.add_argument("--curious-glance-rate-rad-s", type=float, default=1.45, help="Curious glance rate")
    parser.add_argument("--curious-orient-amp-rad", type=float, default=0.34, help="Curious orient amplitude")
    parser.add_argument("--curious-orient-rate-rad-s", type=float, default=1.0, help="Curious yaw/orient rate")
    parser.add_argument("--curious-pause-s", type=float, default=0.6, help="Small hold after each curious glance")
    parser.add_argument("--excite-cycles", type=int, default=5, help="Base shake cycles during the excited gesture")
    parser.add_argument("--base-amp-rad", type=float, default=0.26, help="Base yaw amplitude for excited shake")
    parser.add_argument("--pitch1-back-rad", type=float, default=0.36, help="Backward pitch1 lean magnitude")
    parser.add_argument(
        "--pitch1-back-sign",
        type=float,
        default=-1.0,
        help="Sign applied to the backward pitch1 lean (flip to 1.0 if hardware pitch direction is reversed)",
    )
    parser.add_argument("--excite-rate-rad-s", type=float, default=1.9, help="Rate limit for excited move_to setpoints")
    parser.add_argument("--excite-step-s", type=float, default=0.42, help="Max dwell per excited shake setpoint")
    parser.add_argument("--excited-hold-s", type=float, default=0.9, help="Hold the leaned-back excited pose before and after the shake")
    parser.add_argument("--settle-s", type=float, default=1.2, help="Final compose hold after returning home")
    return parser.parse_args(argv)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _joint_index(joint_names: Sequence[str], aliases: Sequence[str], *, required: bool = True) -> Optional[int]:
    normalized = {str(name).strip().lower(): idx for idx, name in enumerate(joint_names)}
    for alias in aliases:
        idx = normalized.get(str(alias).strip().lower())
        if idx is not None:
            return idx
    if required:
        raise ValueError(f"Missing required joint; tried aliases={list(aliases)} available={list(joint_names)}")
    return None


def _neutral_pose(cfg) -> list[float]:
    pose: list[float] = []
    for limits in cfg.joint_limits_rad:
        lo = float(limits[0])
        hi = float(limits[1])
        pose.append(min(hi, max(lo, 0.0)))
    return pose


def _clamp_joint(cfg, joint_idx: int, value: float) -> float:
    lo = float(cfg.joint_limits_rad[joint_idx][0])
    hi = float(cfg.joint_limits_rad[joint_idx][1])
    return min(hi, max(lo, float(value)))


def _move_target(
    cfg,
    base_pose: Sequence[float],
    *,
    yaw: Optional[float] = None,
    pitch1: Optional[float] = None,
) -> list[float]:
    target = [float(v) for v in base_pose]
    yaw_idx = _joint_index(cfg.joint_names, ("yaw", "base", "base_yaw"))
    pitch1_idx = _joint_index(cfg.joint_names, ("pitch1", "shoulder", "neck_pitch"))
    if yaw is not None:
        target[yaw_idx] = _clamp_joint(cfg, yaw_idx, float(yaw))
    if pitch1 is not None:
        target[pitch1_idx] = _clamp_joint(cfg, pitch1_idx, float(pitch1))
    return target


def _excited_targets(cfg, args: argparse.Namespace, base_pose: Sequence[float]) -> list[list[float]]:
    cycles = max(1, int(args.excite_cycles))
    yaw_idx = _joint_index(cfg.joint_names, ("yaw", "base", "base_yaw"))
    pitch1_idx = _joint_index(cfg.joint_names, ("pitch1", "shoulder", "neck_pitch"))

    center_yaw = float(base_pose[yaw_idx])
    leaned_pitch1 = _clamp_joint(
        cfg,
        pitch1_idx,
        float(base_pose[pitch1_idx]) + (float(args.pitch1_back_sign) * abs(float(args.pitch1_back_rad))),
    )

    targets = [_move_target(cfg, base_pose, yaw=center_yaw, pitch1=leaned_pitch1)]
    base_amp = abs(float(args.base_amp_rad))
    for cycle_idx in range(cycles):
        decay = 1.0 - (0.18 * cycle_idx)
        amp = max(0.04, base_amp * decay)
        targets.append(_move_target(cfg, base_pose, yaw=center_yaw + amp, pitch1=leaned_pitch1))
        targets.append(_move_target(cfg, base_pose, yaw=center_yaw - amp, pitch1=leaned_pitch1))
    targets.append(_move_target(cfg, base_pose, yaw=center_yaw, pitch1=leaned_pitch1))
    return targets


def build_demo_segments(cfg, args: argparse.Namespace) -> list[Segment]:
    base_pose = _neutral_pose(cfg)
    excited_targets = _excited_targets(cfg, args, base_pose)
    excited_pose = excited_targets[0]

    segments: list[Segment] = [
        Segment(
            name="opening_breath",
            action=ActionPlan(
                primitive=PrimitiveKind.BREATH,
                command=BreathCommand(
                    amp_rad=float(args.breath_amp_rad),
                    period_s=float(args.breath_period_s),
                    rate_rad_s=1.0,
                ),
                confidence=1.0,
                explanation="expressive_demo:opening_breath",
                style="calm",
            ),
            max_s=max(1.0, float(args.breath_s)),
            stop_on_done=False,
        ),
        Segment(
            name="curious_orient_left",
            action=ActionPlan(
                primitive=PrimitiveKind.ORIENT_TO_ZONE,
                command=OrientToZoneCommand(
                    zone="left",
                    amp_rad=float(args.curious_orient_amp_rad),
                    rate_rad_s=float(args.curious_orient_rate_rad_s),
                ),
                confidence=1.0,
                explanation="expressive_demo:curious_left",
                style="curious",
            ),
            max_s=2.0,
            stop_on_done=True,
        ),
        Segment(
            name="curious_glance_up_left",
            action=ActionPlan(
                primitive=PrimitiveKind.GLANCE,
                command=GlanceCommand(
                    direction="up",
                    amp_rad=float(args.curious_glance_amp_rad),
                    duration_s=float(args.curious_glance_duration_s),
                    rate_rad_s=float(args.curious_glance_rate_rad_s),
                ),
                confidence=1.0,
                explanation="expressive_demo:curious_up_left",
                style="curious",
            ),
            max_s=max(1.2, float(args.curious_glance_duration_s) + float(args.curious_pause_s)),
            stop_on_done=True,
        ),
        Segment(
            name="curious_pause_left",
            action=ActionPlan(
                primitive=PrimitiveKind.HOLD,
                command=HoldCommand(),
                confidence=1.0,
                explanation="expressive_demo:curious_pause_left",
                style="curious",
            ),
            max_s=max(0.1, float(args.curious_pause_s)),
            stop_on_done=False,
        ),
        Segment(
            name="curious_orient_right",
            action=ActionPlan(
                primitive=PrimitiveKind.ORIENT_TO_ZONE,
                command=OrientToZoneCommand(
                    zone="right",
                    amp_rad=float(args.curious_orient_amp_rad),
                    rate_rad_s=float(args.curious_orient_rate_rad_s),
                ),
                confidence=1.0,
                explanation="expressive_demo:curious_right",
                style="curious",
            ),
            max_s=2.0,
            stop_on_done=True,
        ),
        Segment(
            name="curious_glance_up_right",
            action=ActionPlan(
                primitive=PrimitiveKind.GLANCE,
                command=GlanceCommand(
                    direction="up",
                    amp_rad=float(args.curious_glance_amp_rad),
                    duration_s=float(args.curious_glance_duration_s),
                    rate_rad_s=float(args.curious_glance_rate_rad_s),
                ),
                confidence=1.0,
                explanation="expressive_demo:curious_up_right",
                style="curious",
            ),
            max_s=max(1.2, float(args.curious_glance_duration_s) + float(args.curious_pause_s)),
            stop_on_done=True,
        ),
        Segment(
            name="curious_pause_right",
            action=ActionPlan(
                primitive=PrimitiveKind.HOLD,
                command=HoldCommand(),
                confidence=1.0,
                explanation="expressive_demo:curious_pause_right",
                style="curious",
            ),
            max_s=max(0.1, float(args.curious_pause_s)),
            stop_on_done=False,
        ),
        Segment(
            name="curious_recenter",
            action=ActionPlan(
                primitive=PrimitiveKind.ORIENT_TO_ZONE,
                command=OrientToZoneCommand(
                    zone="center",
                    amp_rad=float(args.curious_orient_amp_rad),
                    rate_rad_s=float(args.curious_orient_rate_rad_s),
                ),
                confidence=1.0,
                explanation="expressive_demo:curious_center",
                style="curious",
            ),
            max_s=2.0,
            stop_on_done=True,
        ),
        Segment(
            name="excited_pose_hold_in",
            action=ActionPlan(
                primitive=PrimitiveKind.MOVE_TO,
                command=MoveToCommand(
                    target_rad=excited_pose,
                    relative=False,
                    rate_rad_s=float(args.excite_rate_rad_s),
                    timeout_s=max(0.3, float(args.excited_hold_s)),
                ),
                confidence=1.0,
                explanation="expressive_demo:excited_pose_hold_in",
                style="focused",
            ),
            max_s=max(0.3, float(args.excited_hold_s)),
            stop_on_done=True,
        ),
    ]

    for idx, target in enumerate(excited_targets):
        segments.append(
            Segment(
                name=f"excited_shake_{idx:02d}",
                action=ActionPlan(
                    primitive=PrimitiveKind.MOVE_TO,
                    command=MoveToCommand(
                        target_rad=target,
                        relative=False,
                        rate_rad_s=float(args.excite_rate_rad_s),
                        timeout_s=max(0.2, float(args.excite_step_s)),
                    ),
                    confidence=1.0,
                    explanation="expressive_demo:excited_shake",
                    style="focused",
                ),
                max_s=max(0.2, float(args.excite_step_s)),
                stop_on_done=True,
            )
        )

    segments.append(
        Segment(
            name="excited_pose_hold_out",
            action=ActionPlan(
                primitive=PrimitiveKind.MOVE_TO,
                command=MoveToCommand(
                    target_rad=excited_pose,
                    relative=False,
                    rate_rad_s=float(args.excite_rate_rad_s),
                    timeout_s=max(0.3, float(args.excited_hold_s)),
                ),
                confidence=1.0,
                explanation="expressive_demo:excited_pose_hold_out",
                style="focused",
            ),
            max_s=max(0.3, float(args.excited_hold_s)),
            stop_on_done=True,
        )
    )

    return segments


def _run_segment(
    *,
    segment: Segment,
    executor: TrajectoryExecutor,
    servo,
    actuation_on: bool,
    hz: float,
    status_every_s: float,
    joint_names: Sequence[str],
) -> None:
    start = time.monotonic()
    last = start
    next_tick = start
    last_status = start
    first_tick = True

    logger.info("segment start: %s primitive=%s style=%s", segment.name, segment.action.primitive.value, segment.action.style)

    while True:
        now = time.monotonic()
        if now >= start + max(0.01, segment.max_s):
            break
        if now < next_tick:
            time.sleep(next_tick - now)
            continue

        dt = max(0.0, now - last)
        last = now
        request = segment.action if not first_tick else replace(segment.action, cancel_current=True)
        cmd = executor.step(request, dt)
        first_tick = False

        if actuation_on:
            servo.set_angles(cmd.joint_angles_rad)

        if now - last_status >= max(0.1, status_every_s):
            preview = ", ".join(
                f"{joint_names[i]}={cmd.joint_angles_rad[i]:+.3f}"
                for i in range(min(5, len(joint_names)))
            )
            logger.info(
                "[%s] t=%.2fs status=%s%s %s",
                segment.name,
                now - start,
                executor.control_state.status.value,
                f" reason={executor.control_state.reason}" if executor.control_state.reason else "",
                preview,
            )
            last_status = now

        if segment.stop_on_done and executor.control_state.status in _DONE_STATES:
            break

        next_tick += 1.0 / hz

    logger.info(
        "segment end: %s status=%s reason=%s",
        segment.name,
        executor.control_state.status.value,
        executor.control_state.reason,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_logging()
    args = _parse_args(argv)
    cfg = load_config(args.config)
    if args.runtime_mode:
        cfg.mode = args.runtime_mode

    actuation_on = bool(args.enable) and not bool(args.dry_run)
    if cfg.mode == "jetson_full" and not actuation_on and not args.dry_run:
        logger.error("Refusing to actuate jetson_full hardware without --enable")
        return 2

    hz = float(args.hz) if args.hz > 0 else float(cfg.loop_rates.control_hz)
    hz = max(1.0, hz)

    servo = _build_servo(cfg)
    executor = TrajectoryExecutor(cfg.joint_limits_rad, style_profiles=getattr(cfg, "style_profiles", None))
    segments = build_demo_segments(cfg, args)

    logger.info(
        "expressive_demo: mode=%s actuation_on=%s dry_run=%s hz=%.1f segments=%d",
        cfg.mode,
        actuation_on,
        bool(args.dry_run),
        hz,
        len(segments),
    )

    try:
        if actuation_on:
            servo.enable(True)

        if args.neutral_start:
            _run_segment(
                segment=Segment(
                    name="neutral_start",
                    action=ActionPlan(
                        primitive=PrimitiveKind.HOME,
                        command=HomeCommand(rate_rad_s=1.0),
                        confidence=1.0,
                        explanation="expressive_demo:neutral_start",
                        style="calm",
                    ),
                    max_s=2.0,
                    stop_on_done=True,
                ),
                executor=executor,
                servo=servo,
                actuation_on=actuation_on,
                hz=hz,
                status_every_s=float(args.status_every_s),
                joint_names=cfg.joint_names,
            )

        for segment in segments:
            _run_segment(
                segment=segment,
                executor=executor,
                servo=servo,
                actuation_on=actuation_on,
                hz=hz,
                status_every_s=float(args.status_every_s),
                joint_names=cfg.joint_names,
            )

        if args.neutral_end:
            _run_segment(
                segment=Segment(
                    name="neutral_end",
                    action=ActionPlan(
                        primitive=PrimitiveKind.HOME,
                        command=HomeCommand(rate_rad_s=1.1),
                        confidence=1.0,
                        explanation="expressive_demo:neutral_end",
                        style="calm",
                    ),
                    max_s=max(0.5, float(args.settle_s)) + 1.0,
                    stop_on_done=True,
                ),
                executor=executor,
                servo=servo,
                actuation_on=actuation_on,
                hz=hz,
                status_every_s=float(args.status_every_s),
                joint_names=cfg.joint_names,
            )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    finally:
        if actuation_on and not args.keep_enabled:
            servo.enable(False)
        servo.shutdown()

    logger.info("expressive_demo complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
