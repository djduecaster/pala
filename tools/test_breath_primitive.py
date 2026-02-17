#!/usr/bin/env python3
"""Run the typed BREATH primitive through the control executor."""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pala.config import load_config
from pala.control import TrajectoryExecutor
from pala.control.primitives import PrimitiveKind, BreathCommand
from pala.main import _build_servo
from pala.types import ActionPlan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test BREATH primitive on hardware or dummy servo")
    parser.add_argument("--config", default="config/robot.yaml", help="Config path")
    parser.add_argument("--mode", default=None, help="Optional mode override (e.g. jetson_full)")
    parser.add_argument("--enable", action="store_true", help="Required to send commands to real hardware")
    parser.add_argument("--duration-s", type=float, default=20.0, help="Test duration")
    parser.add_argument("--hz", type=float, default=0.0, help="Control update rate (0 uses config control_hz)")
    parser.add_argument("--amp-rad", type=float, default=0.08, help="Breath amplitude")
    parser.add_argument("--period-s", type=float, default=7.0, help="Breath period")
    parser.add_argument("--rate-rad-s", type=float, default=1.0, help="Executor rate limit")
    parser.add_argument(
        "--pitch2-scale",
        type=float,
        default=0.6,
        help="Extra pitch2 breathe scale relative to --amp-rad",
    )
    parser.add_argument("--status-every-s", type=float, default=1.0, help="Status print cadence")
    parser.add_argument("--keep-enabled", action="store_true", help="Do not disable servos on exit")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    cfg = load_config(args.config)
    if args.mode:
        cfg.mode = args.mode

    if cfg.mode == "jetson_full" and not args.enable:
        print("Refusing to move hardware without --enable in jetson_full mode.")
        return 2

    servo = _build_servo(cfg)
    executor = TrajectoryExecutor(cfg.joint_limits_rad)
    action = ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command=BreathCommand(
            amp_rad=float(args.amp_rad),
            period_s=float(args.period_s),
            rate_rad_s=float(args.rate_rad_s),
        ),
        confidence=1.0,
        explanation="manual_breath_test",
    )

    hz = float(args.hz) if args.hz > 0 else float(cfg.loop_rates.control_hz)
    hz = max(1.0, hz)
    dt_target = 1.0 / hz

    print(
        f"breath_test: mode={cfg.mode} enable={args.enable} duration_s={args.duration_s:.1f} "
        f"hz={hz:.1f} amp={args.amp_rad:.3f} period={args.period_s:.2f} "
        f"rate={args.rate_rad_s:.2f} pitch2_scale={args.pitch2_scale:.2f}"
    )
    pitch2_idx = cfg.joint_names.index("pitch2") if "pitch2" in cfg.joint_names else None
    if pitch2_idx is None:
        print("warning: pitch2 joint not found; running default breath only")

    start = time.monotonic()
    last = start
    next_tick = start
    last_status = start

    try:
        if args.enable:
            servo.enable(True)

        while True:
            now = time.monotonic()
            if now >= start + max(0.1, float(args.duration_s)):
                break
            if now < next_tick:
                time.sleep(next_tick - now)
                continue

            dt = max(0.0, now - last)
            last = now
            cmd = executor.step(action, dt)
            cmd_angles = list(cmd.joint_angles_rad)
            if pitch2_idx is not None and 0 <= pitch2_idx < len(cmd_angles):
                phase = (now - start) * (2.0 * math.pi / max(0.1, float(args.period_s)))
                cmd_angles[pitch2_idx] += float(args.amp_rad) * float(args.pitch2_scale) * math.sin(phase)
            if args.enable:
                servo.set_angles(cmd_angles)

            if now - last_status >= max(0.1, float(args.status_every_s)):
                joint_preview = ", ".join(f"{v:+.3f}" for v in cmd_angles)
                print(f"t={now - start:6.2f}s joints_rad=[{joint_preview}]")
                last_status = now

            next_tick += dt_target
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if args.enable and not args.keep_enabled:
            servo.enable(False)
        servo.shutdown()

    print("breath_test complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
