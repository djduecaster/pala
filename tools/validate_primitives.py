#!/usr/bin/env python3
"""Primitive validation runner for hardware or dry-run testing."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass, replace
from typing import Any, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pala.config import load_config
from pala.control import TrajectoryExecutor
from pala.control.executor import ExecutionStatus
from pala.control.primitives import (
    PrimitiveKind,
    HoldCommand,
    HomeCommand,
    MoveToCommand,
    GazeToCommand,
    GlanceCommand,
    NodCommand,
    BreathCommand,
    OrientToZoneCommand,
)
from pala.main import _build_servo
from pala.hardware import DummyServo
from pala.types import ActionPlan, to_json_dict


@dataclass(frozen=True)
class Segment:
    name: str
    action: ActionPlan
    max_s: float
    stop_on_done: bool


@dataclass
class SegmentResult:
    name: str
    elapsed_s: float
    final_status: str
    final_reason: Optional[str]
    samples: int


class JsonlWriter:
    def __init__(self, path: Optional[str]) -> None:
        self._fh = None
        if path:
            p = pathlib.Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = p.open("a", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate typed control primitives on servo hardware")
    parser.add_argument("--config", default="config/robot.yaml", help="Config path")
    parser.add_argument("--runtime-mode", default=None, help="Optional mode override (e.g. jetson_full)")
    parser.add_argument("--scenario", choices=["suite", "single", "sweep"], default="suite")
    parser.add_argument("--enable", action="store_true", help="Required to actuate hardware in jetson_full")
    parser.add_argument("--dry-run", action="store_true", help="Run executor only (no servo writes)")
    parser.add_argument("--hz", type=float, default=0.0, help="Control update rate (0 uses config control_hz)")
    parser.add_argument("--status-every-s", type=float, default=0.5, help="Status print interval")
    parser.add_argument("--neutral-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neutral-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-jsonl", default="logs/primitive_validation.jsonl", help="Optional JSONL output path")
    parser.add_argument("--no-log", action="store_true", help="Disable JSONL logging")

    parser.add_argument("--primitive", default="breath", help="Primitive for single/sweep scenarios")
    parser.add_argument("--duration-s", type=float, default=4.0, help="Segment duration (single/sweep)")
    parser.add_argument("--rate-rad-s", type=float, default=1.2, help="Rate limit parameter")
    parser.add_argument("--amp-rad", type=float, default=0.1, help="Amplitude parameter")
    parser.add_argument("--period-s", type=float, default=7.0, help="Period parameter")
    parser.add_argument("--direction", choices=["left", "right", "up", "down"], default="left")
    parser.add_argument("--zone", choices=["left", "center", "right"], default="center")
    parser.add_argument("--yaw-rad", type=float, default=0.0, help="Yaw target for gaze_to")
    parser.add_argument("--pitch-rad", type=float, default=0.0, help="Pitch target for gaze_to")
    parser.add_argument("--target-rad", default=None, help="Comma-separated target list for move_to")
    parser.add_argument("--relative", action="store_true", help="Use relative move_to target")
    parser.add_argument("--cycles", type=int, default=1, help="Nod cycles")
    parser.add_argument("--dwell-s", type=float, default=0.0, help="Gaze dwell before done")
    parser.add_argument("--timeout-s", type=float, default=1.5, help="Timeout for move_to/gaze_to")

    parser.add_argument("--suite-breath-s", type=float, default=8.0, help="Breath duration in suite")
    parser.add_argument("--sweep-param", choices=["amp_rad", "duration_s", "rate_rad_s", "period_s"], default="amp_rad")
    parser.add_argument("--sweep-values", default="0.05,0.08,0.12", help="Comma-separated values for sweep")
    parser.add_argument("--neutral-between", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _build_action(args: argparse.Namespace, cfg, *, primitive_name: Optional[str] = None) -> ActionPlan:
    token = (primitive_name or args.primitive).strip().lower()
    primitive = PrimitiveKind(token)

    if primitive == PrimitiveKind.HOLD:
        command = HoldCommand()
    elif primitive == PrimitiveKind.HOME:
        command = HomeCommand(rate_rad_s=float(args.rate_rad_s))
    elif primitive == PrimitiveKind.BREATH:
        command = BreathCommand(
            amp_rad=float(args.amp_rad),
            period_s=float(args.period_s),
            rate_rad_s=float(args.rate_rad_s),
        )
    elif primitive == PrimitiveKind.GLANCE:
        command = GlanceCommand(
            direction=str(args.direction),
            amp_rad=float(args.amp_rad),
            duration_s=float(args.duration_s),
            rate_rad_s=float(args.rate_rad_s),
        )
    elif primitive == PrimitiveKind.NOD:
        command = NodCommand(
            amp_rad=float(args.amp_rad),
            duration_s=float(args.duration_s),
            cycles=max(1, int(args.cycles)),
            rate_rad_s=float(args.rate_rad_s),
        )
    elif primitive == PrimitiveKind.ORIENT_TO_ZONE:
        command = OrientToZoneCommand(
            zone=str(args.zone),
            amp_rad=float(args.amp_rad),
            rate_rad_s=float(args.rate_rad_s),
        )
    elif primitive == PrimitiveKind.GAZE_TO:
        command = GazeToCommand(
            yaw_rad=float(args.yaw_rad),
            pitch_rad=float(args.pitch_rad),
            rate_rad_s=float(args.rate_rad_s),
            dwell_s=float(args.dwell_s),
            timeout_s=float(args.timeout_s),
        )
    elif primitive == PrimitiveKind.MOVE_TO:
        target = _parse_move_target(args, cfg)
        command = MoveToCommand(
            target_rad=target,
            relative=bool(args.relative),
            rate_rad_s=float(args.rate_rad_s),
            timeout_s=float(args.timeout_s),
        )
    else:
        raise ValueError(f"Unsupported primitive for validator: {primitive.value}")

    return ActionPlan(
        primitive=primitive,
        command=command,
        confidence=1.0,
        explanation="primitive_validation",
    )


def _parse_move_target(args: argparse.Namespace, cfg) -> list[float]:
    if args.target_rad:
        pieces = [p.strip() for p in str(args.target_rad).split(",") if p.strip()]
        vals = [float(p) for p in pieces]
        if len(vals) != len(cfg.joint_names):
            raise ValueError(
                f"--target-rad expects {len(cfg.joint_names)} values; got {len(vals)}"
            )
        return vals

    target = [0.0 for _ in cfg.joint_names]
    for i, lim in enumerate(cfg.joint_limits_rad):
        lo, hi = float(lim[0]), float(lim[1])
        span = hi - lo
        if span <= 1e-6:
            target[i] = lo
            continue
        frac = 0.15 if i % 2 == 0 else -0.15
        v = frac * span
        target[i] = max(lo, min(hi, v))
    return target


def _is_continuous(kind: PrimitiveKind) -> bool:
    return kind in {PrimitiveKind.HOLD, PrimitiveKind.BREATH}


def _run_segment(
    *,
    segment: Segment,
    executor: TrajectoryExecutor,
    servo,
    actuation_on: bool,
    hz: float,
    status_every_s: float,
    joint_names: list[str],
    joint_limits_rad: list[list[float]],
    summary: dict[str, Any],
    logger: JsonlWriter,
) -> SegmentResult:
    start = time.monotonic()
    last = start
    next_tick = start
    last_status = start
    first_tick = True
    samples = 0

    logger.write(
        {
            "type": "segment_start",
            "segment": segment.name,
            "action": to_json_dict(segment.action),
            "monotonic_s": start,
        }
    )

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

        samples += 1
        status = executor.control_state.status
        reason = executor.control_state.reason
        summary["status_counts"][status.value] = summary["status_counts"].get(status.value, 0) + 1

        for i, angle in enumerate(cmd.joint_angles_rad):
            summary["max_abs_by_joint"][i] = max(summary["max_abs_by_joint"][i], abs(float(angle)))
            lo, hi = float(joint_limits_rad[i][0]), float(joint_limits_rad[i][1])
            if angle <= lo + 1e-6 or angle >= hi - 1e-6:
                summary["limit_hits_by_joint"][i] += 1

        logger.write(
            {
                "type": "tick",
                "segment": segment.name,
                "monotonic_s": now,
                "elapsed_s": now - start,
                "status": status.value,
                "reason": reason,
                "angles_rad": [float(v) for v in cmd.joint_angles_rad],
                "enable": bool(cmd.enable),
            }
        )

        if now - last_status >= max(0.1, status_every_s):
            status_line = (
                f"[{segment.name}] t={now - start:5.2f}s status={status.value}"
                + (f" reason={reason}" if reason else "")
            )
            angle_preview = ", ".join(f"{joint_names[i]}={cmd.joint_angles_rad[i]:+.3f}" for i in range(min(5, len(joint_names))))
            print(f"{status_line} {angle_preview}")
            last_status = now

        if segment.stop_on_done and status in {
            ExecutionStatus.DONE,
            ExecutionStatus.TIMED_OUT,
            ExecutionStatus.REJECTED,
        }:
            break

        next_tick += 1.0 / hz

    end = time.monotonic()
    final = executor.control_state
    logger.write(
        {
            "type": "segment_end",
            "segment": segment.name,
            "monotonic_s": end,
            "elapsed_s": end - start,
            "final_status": final.status.value,
            "final_reason": final.reason,
            "samples": samples,
        }
    )
    return SegmentResult(
        name=segment.name,
        elapsed_s=end - start,
        final_status=final.status.value,
        final_reason=final.reason,
        samples=samples,
    )


def _suite_segments(args: argparse.Namespace, cfg) -> list[Segment]:
    names = cfg.joint_names
    target = [0.0 for _ in names]
    if len(target) > 0:
        target[0] = 0.2
    if len(target) > 4:
        target[4] = -0.15

    return [
        Segment("hold", ActionPlan(PrimitiveKind.HOLD, HoldCommand(), 1.0, "suite"), max_s=1.5, stop_on_done=False),
        Segment("home", ActionPlan(PrimitiveKind.HOME, HomeCommand(rate_rad_s=1.2), 1.0, "suite"), max_s=2.0, stop_on_done=True),
        Segment(
            "breath",
            ActionPlan(PrimitiveKind.BREATH, BreathCommand(amp_rad=0.08, period_s=7.0, rate_rad_s=1.0), 1.0, "suite"),
            max_s=max(1.0, float(args.suite_breath_s)),
            stop_on_done=False,
        ),
        Segment(
            "glance_left",
            ActionPlan(PrimitiveKind.GLANCE, GlanceCommand(direction="left", amp_rad=0.25, duration_s=0.7, rate_rad_s=1.6), 1.0, "suite"),
            max_s=1.2,
            stop_on_done=True,
        ),
        Segment(
            "glance_right",
            ActionPlan(PrimitiveKind.GLANCE, GlanceCommand(direction="right", amp_rad=0.25, duration_s=0.7, rate_rad_s=1.6), 1.0, "suite"),
            max_s=1.2,
            stop_on_done=True,
        ),
        Segment(
            "nod",
            ActionPlan(PrimitiveKind.NOD, NodCommand(amp_rad=0.18, duration_s=0.8, cycles=1, rate_rad_s=1.8), 1.0, "suite"),
            max_s=1.4,
            stop_on_done=True,
        ),
        Segment(
            "orient_left",
            ActionPlan(PrimitiveKind.ORIENT_TO_ZONE, OrientToZoneCommand(zone="left", amp_rad=0.2, rate_rad_s=1.4), 1.0, "suite"),
            max_s=1.6,
            stop_on_done=True,
        ),
        Segment(
            "orient_center",
            ActionPlan(PrimitiveKind.ORIENT_TO_ZONE, OrientToZoneCommand(zone="center", amp_rad=0.2, rate_rad_s=1.4), 1.0, "suite"),
            max_s=1.6,
            stop_on_done=True,
        ),
        Segment(
            "orient_right",
            ActionPlan(PrimitiveKind.ORIENT_TO_ZONE, OrientToZoneCommand(zone="right", amp_rad=0.2, rate_rad_s=1.4), 1.0, "suite"),
            max_s=1.6,
            stop_on_done=True,
        ),
        Segment(
            "gaze_to",
            ActionPlan(PrimitiveKind.GAZE_TO, GazeToCommand(yaw_rad=0.1, pitch_rad=-0.1, rate_rad_s=1.4, dwell_s=0.2, timeout_s=1.8), 1.0, "suite"),
            max_s=2.0,
            stop_on_done=True,
        ),
        Segment(
            "move_to",
            ActionPlan(PrimitiveKind.MOVE_TO, MoveToCommand(target_rad=target, relative=False, rate_rad_s=1.2, timeout_s=2.0), 1.0, "suite"),
            max_s=2.2,
            stop_on_done=True,
        ),
    ]


def _single_segments(args: argparse.Namespace, cfg) -> list[Segment]:
    action = _build_action(args, cfg)
    return [
        Segment(
            name=f"single_{action.primitive.value}",
            action=action,
            max_s=max(0.1, float(args.duration_s)),
            stop_on_done=not _is_continuous(action.primitive),
        )
    ]


def _sweep_segments(args: argparse.Namespace, cfg) -> list[Segment]:
    values = [float(v.strip()) for v in str(args.sweep_values).split(",") if v.strip()]
    if not values:
        raise ValueError("No sweep values provided")
    segments: list[Segment] = []
    for value in values:
        if args.sweep_param == "amp_rad":
            args.amp_rad = value
        elif args.sweep_param == "duration_s":
            args.duration_s = value
        elif args.sweep_param == "rate_rad_s":
            args.rate_rad_s = value
        elif args.sweep_param == "period_s":
            args.period_s = value
        action = _build_action(args, cfg)
        segments.append(
            Segment(
                name=f"sweep_{action.primitive.value}_{args.sweep_param}={value:.4g}",
                action=action,
                max_s=max(0.1, float(args.duration_s)),
                stop_on_done=not _is_continuous(action.primitive),
            )
        )
    return segments


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.runtime_mode:
        cfg.mode = args.runtime_mode

    actuation_on = bool(args.enable) and not bool(args.dry_run)
    if cfg.mode == "jetson_full" and not actuation_on and not args.dry_run:
        print("Refusing to actuate jetson_full hardware without --enable")
        return 2

    hz = float(args.hz) if args.hz > 0 else float(cfg.loop_rates.control_hz)
    hz = max(1.0, hz)

    logger = JsonlWriter(None if args.no_log else args.log_jsonl)
    # Dry runs must remain hardware-free even when the config mode is jetson_full.
    servo = DummyServo() if args.dry_run else _build_servo(cfg)
    executor = TrajectoryExecutor(cfg.joint_limits_rad)

    summary: dict[str, Any] = {
        "status_counts": {},
        "max_abs_by_joint": [0.0 for _ in cfg.joint_names],
        "limit_hits_by_joint": [0 for _ in cfg.joint_names],
    }
    results: list[SegmentResult] = []

    if args.scenario == "suite":
        segments = _suite_segments(args, cfg)
    elif args.scenario == "single":
        segments = _single_segments(args, cfg)
    else:
        segments = _sweep_segments(args, cfg)

    print(
        f"primitive_validate: scenario={args.scenario} mode={cfg.mode} "
        f"actuation_on={actuation_on} dry_run={args.dry_run} hz={hz:.1f} "
        f"segments={len(segments)}"
    )

    try:
        if actuation_on:
            servo.enable(True)

        if args.neutral_start:
            home = Segment(
                name="neutral_start",
                action=ActionPlan(PrimitiveKind.HOME, HomeCommand(rate_rad_s=1.0), 1.0, "validator"),
                max_s=2.0,
                stop_on_done=True,
            )
            results.append(
                _run_segment(
                    segment=home,
                    executor=executor,
                    servo=servo,
                    actuation_on=actuation_on,
                    hz=hz,
                    status_every_s=args.status_every_s,
                    joint_names=cfg.joint_names,
                    joint_limits_rad=cfg.joint_limits_rad,
                    summary=summary,
                    logger=logger,
                )
            )

        for i, segment in enumerate(segments):
            results.append(
                _run_segment(
                    segment=segment,
                    executor=executor,
                    servo=servo,
                    actuation_on=actuation_on,
                    hz=hz,
                    status_every_s=args.status_every_s,
                    joint_names=cfg.joint_names,
                    joint_limits_rad=cfg.joint_limits_rad,
                    summary=summary,
                    logger=logger,
                )
            )
            if args.scenario == "sweep" and args.neutral_between and i + 1 < len(segments):
                settle = Segment(
                    name=f"neutral_between_{i+1}",
                    action=ActionPlan(PrimitiveKind.HOME, HomeCommand(rate_rad_s=1.0), 1.0, "validator"),
                    max_s=1.6,
                    stop_on_done=True,
                )
                results.append(
                    _run_segment(
                        segment=settle,
                        executor=executor,
                        servo=servo,
                        actuation_on=actuation_on,
                        hz=hz,
                        status_every_s=args.status_every_s,
                        joint_names=cfg.joint_names,
                        joint_limits_rad=cfg.joint_limits_rad,
                        summary=summary,
                        logger=logger,
                    )
                )

        if args.neutral_end:
            home = Segment(
                name="neutral_end",
                action=ActionPlan(PrimitiveKind.HOME, HomeCommand(rate_rad_s=1.0), 1.0, "validator"),
                max_s=2.0,
                stop_on_done=True,
            )
            results.append(
                _run_segment(
                    segment=home,
                    executor=executor,
                    servo=servo,
                    actuation_on=actuation_on,
                    hz=hz,
                    status_every_s=args.status_every_s,
                    joint_names=cfg.joint_names,
                    joint_limits_rad=cfg.joint_limits_rad,
                    summary=summary,
                    logger=logger,
                )
            )
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if actuation_on:
            servo.enable(False)
        servo.shutdown()
        logger.close()

    print("\nsummary:")
    print(f"  segments_run={len(results)}")
    print(f"  status_counts={summary['status_counts']}")
    for i, name in enumerate(cfg.joint_names):
        print(
            f"  joint={name} max_abs_rad={summary['max_abs_by_joint'][i]:.4f} "
            f"limit_hits={summary['limit_hits_by_joint'][i]}"
        )
    if not args.no_log:
        print(f"  jsonl={args.log_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
