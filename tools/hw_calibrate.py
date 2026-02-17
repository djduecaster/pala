#!/usr/bin/env python3
"""Manual servo calibration tool (Jetson hardware only)."""
from __future__ import annotations

import argparse
import math
import time
from typing import Iterable, List

from pala.config import load_config
from pala.hardware import PCA9685Servo, ServoCalibration


def _build_servo(cfg) -> tuple[PCA9685Servo, List[str], dict]:
    servo_cal = cfg.servo_calibration
    if not servo_cal:
        raise ValueError("servo_calibration is required in config")

    joint_names = list(cfg.joint_names)
    channels = list(servo_cal.get("channels", []))
    if len(channels) != len(joint_names):
        raise ValueError("servo_calibration.channels must match joint_names length")

    per_joint = servo_cal.get("per_joint", {})
    if not isinstance(per_joint, dict):
        raise ValueError("servo_calibration.per_joint must be a mapping")

    min_pulse = []
    max_pulse = []
    scales = []
    offsets = []
    reverses = []
    for name in joint_names:
        if name not in per_joint:
            raise ValueError(f"servo_calibration.per_joint missing '{name}'")
        jc = per_joint[name] or {}
        min_pulse.append(float(jc["min_pulse"]))
        max_pulse.append(float(jc["max_pulse"]))
        scales.append(float(jc["angle_scale"]))
        offsets.append(float(jc["angle_offset"]))
        reverses.append(bool(jc["reverse"]))

    cal = ServoCalibration(
        bus_number=int(servo_cal.get("bus_number", 7)),
        address=int(servo_cal.get("address", 0x40)),
        frequency=int(servo_cal.get("frequency", 50)),
        channels=channels,
        min_pulse_us=min_pulse,
        max_pulse_us=max_pulse,
        angle_scales=scales,
        angle_offsets_deg=offsets,
        reverses=reverses,
    )
    return PCA9685Servo(cal), joint_names, per_joint


def _map_servo_deg(joint_deg: float, jc: dict) -> tuple[float, float]:
    scale = float(jc["angle_scale"])
    offset = float(jc["angle_offset"])
    reverse = bool(jc["reverse"])
    raw = joint_deg * scale + offset
    if reverse:
        raw = 180.0 - raw
    clamped = max(0.0, min(180.0, raw))
    return raw, clamped


def _print_joint_map(joint: str, joint_deg: float, jc: dict) -> None:
    scale = float(jc["angle_scale"])
    offset = float(jc["angle_offset"])
    reverse = bool(jc["reverse"])
    _raw, clamped = _map_servo_deg(joint_deg, jc)
    print(
        f"map joint={joint} control_deg={joint_deg:.2f} "
        f"scale={scale:.4f} offset={offset:.2f} reverse={reverse} "
        f"servo_deg={clamped:.2f}"
    )


def _send_single_joint(
    servo: PCA9685Servo,
    joint_names: List[str],
    current_cmd_rad: List[float],
    *,
    smoothing: bool,
    slew_rate_deg_s: float,
    interp_dt_s: float,
    joint: str,
    joint_rad: float,
) -> None:
    target = list(current_cmd_rad)
    idx = joint_names.index(joint)
    target[idx] = joint_rad
    _apply_command(
        servo,
        current_cmd_rad,
        target,
        smoothing=smoothing,
        slew_rate_deg_s=slew_rate_deg_s,
        interp_dt_s=interp_dt_s,
    )


def _cmd_neutral(
    servo: PCA9685Servo,
    current_cmd_rad: List[float],
    *,
    smoothing: bool,
    slew_rate_deg_s: float,
    interp_dt_s: float,
) -> None:
    target = [0.0 for _ in current_cmd_rad]
    _apply_command(
        servo,
        current_cmd_rad,
        target,
        smoothing=smoothing,
        slew_rate_deg_s=slew_rate_deg_s,
        interp_dt_s=interp_dt_s,
    )


def _apply_command(
    servo: PCA9685Servo,
    current_cmd_rad: List[float],
    target_cmd_rad: List[float],
    *,
    smoothing: bool,
    slew_rate_deg_s: float,
    interp_dt_s: float,
) -> None:
    if not smoothing:
        for i in range(len(current_cmd_rad)):
            current_cmd_rad[i] = float(target_cmd_rad[i])
        servo.set_angles(current_cmd_rad)
        return

    dt_s = max(0.005, float(interp_dt_s))
    max_step_rad = math.radians(max(1e-3, float(slew_rate_deg_s))) * dt_s

    while True:
        max_abs_delta = 0.0
        for i in range(len(current_cmd_rad)):
            delta = float(target_cmd_rad[i]) - float(current_cmd_rad[i])
            if abs(delta) > max_abs_delta:
                max_abs_delta = abs(delta)
            if delta > max_step_rad:
                delta = max_step_rad
            elif delta < -max_step_rad:
                delta = -max_step_rad
            current_cmd_rad[i] += delta

        servo.set_angles(current_cmd_rad)
        if max_abs_delta <= max_step_rad:
            break
        time.sleep(dt_s)

    # Snap to exact target after interpolation to avoid accumulated float drift.
    for i in range(len(current_cmd_rad)):
        current_cmd_rad[i] = float(target_cmd_rad[i])
    servo.set_angles(current_cmd_rad)


def _print_joint_config_table(cfg, joint_names: List[str], per_joint: dict) -> None:
    print("joint config + mapping samples (control_deg -> servo_deg)")
    for idx, name in enumerate(joint_names):
        jc = per_joint[name]
        lo_rad, hi_rad = cfg.joint_limits_rad[idx]
        lo_deg = math.degrees(float(lo_rad))
        hi_deg = math.degrees(float(hi_rad))
        _raw_lo, map_lo = _map_servo_deg(lo_deg, jc)
        _raw_0, map_0 = _map_servo_deg(0.0, jc)
        _raw_hi, map_hi = _map_servo_deg(hi_deg, jc)
        print(
            f"{name}: min_pulse={jc['min_pulse']} max_pulse={jc['max_pulse']} "
            f"scale={jc['angle_scale']} offset={jc['angle_offset']} reverse={jc['reverse']}"
        )
        print(
            f"  control_deg=[{lo_deg:.2f}, 0.00, {hi_deg:.2f}] "
            f"-> servo_deg=[{map_lo:.2f}, {map_0:.2f}, {map_hi:.2f}]"
        )


def _run_repl(
    servo: PCA9685Servo,
    joint_names: List[str],
    per_joint: dict,
    current_cmd_rad: List[float],
    *,
    smoothing: bool,
    slew_rate_deg_s: float,
    interp_dt_s: float,
) -> int:
    print("REPL commands:")
    print("  neutral")
    print("  <joint> <deg>   (example: yaw 5)")
    print("  on | off")
    print("  q")
    print("joints:", ", ".join(joint_names))
    while True:
        line = input("hw-cal> ").strip()
        if not line:
            continue
        if line.lower() in {"q", "quit", "exit"}:
            return 0
        if line.lower() == "neutral":
            _cmd_neutral(
                servo,
                current_cmd_rad,
                smoothing=smoothing,
                slew_rate_deg_s=slew_rate_deg_s,
                interp_dt_s=interp_dt_s,
            )
            print("set neutral")
            continue
        if line.lower() == "off":
            servo.enable(False)
            print("servos disabled")
            continue
        if line.lower() == "on":
            servo.enable(True)
            print("servos enabled")
            continue
        parts = line.split()
        if len(parts) != 2:
            print("invalid command")
            continue
        joint, deg_raw = parts
        if joint not in joint_names:
            print(f"unknown joint: {joint}")
            continue
        try:
            joint_deg = float(deg_raw)
        except ValueError:
            print(f"invalid degree value: {deg_raw}")
            continue
        _print_joint_map(joint, joint_deg, per_joint[joint])
        _send_single_joint(
            servo,
            joint_names,
            current_cmd_rad,
            smoothing=smoothing,
            slew_rate_deg_s=slew_rate_deg_s,
            interp_dt_s=interp_dt_s,
            joint=joint,
            joint_rad=math.radians(joint_deg),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual servo calibration tool")
    parser.add_argument("--config", type=str, default="config/robot.yaml", help="Config path")
    parser.add_argument("--enable", action="store_true", help="Required for any command that moves hardware")
    parser.add_argument("--list-joints", action="store_true", help="Print joint names and mapping values")
    parser.add_argument("--neutral", action="store_true", help="Command all joints to control 0 rad")
    parser.add_argument("--joint", type=str, default=None, help="Joint name to command")
    parser.add_argument("--deg", type=float, default=None, help="Joint command in control degrees")
    parser.add_argument("--rad", type=float, default=None, help="Joint command in control radians")
    parser.add_argument("--hold-s", type=float, default=0.5, help="Hold time for one-shot command")
    parser.add_argument(
        "--slew-rate-deg-s",
        type=float,
        default=35.0,
        help="Smoothing slew rate in control degrees/second (default: 35)",
    )
    parser.add_argument(
        "--interp-dt-s",
        type=float,
        default=0.02,
        help="Interpolation timestep in seconds for smoothing (default: 0.02)",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable interpolation and send target immediately",
    )
    parser.add_argument("--repl", action="store_true", help="Interactive calibration REPL")
    args = parser.parse_args()

    cfg = load_config(args.config)
    servo, joint_names, per_joint = _build_servo(cfg)
    current_cmd_rad = [0.0 for _ in joint_names]
    smoothing = not args.no_smoothing
    needs_motion = args.neutral or args.repl or (args.joint is not None)
    try:
        if args.list_joints:
            _print_joint_config_table(cfg, joint_names, per_joint)

        if needs_motion and not args.enable:
            print("Refusing to move hardware without --enable")
            return 2

        if args.repl:
            print(
                f"smoothing={'on' if smoothing else 'off'} "
                f"slew_rate_deg_s={args.slew_rate_deg_s:.2f} interp_dt_s={args.interp_dt_s:.3f}"
            )
            return _run_repl(
                servo,
                joint_names,
                per_joint,
                current_cmd_rad,
                smoothing=smoothing,
                slew_rate_deg_s=float(args.slew_rate_deg_s),
                interp_dt_s=float(args.interp_dt_s),
            )

        if args.neutral:
            _cmd_neutral(
                servo,
                current_cmd_rad,
                smoothing=smoothing,
                slew_rate_deg_s=float(args.slew_rate_deg_s),
                interp_dt_s=float(args.interp_dt_s),
            )
            print("set neutral control pose (all joints = 0 rad)")
            time.sleep(max(0.0, float(args.hold_s)))

        if args.joint is not None:
            if args.joint not in joint_names:
                raise ValueError(f"Unknown joint '{args.joint}'. Valid: {joint_names}")
            if (args.deg is None) == (args.rad is None):
                raise ValueError("Provide exactly one of --deg or --rad with --joint")
            joint_deg = float(args.deg) if args.deg is not None else math.degrees(float(args.rad))
            _print_joint_map(args.joint, joint_deg, per_joint[args.joint])
            _send_single_joint(
                servo,
                joint_names,
                current_cmd_rad,
                smoothing=smoothing,
                slew_rate_deg_s=float(args.slew_rate_deg_s),
                interp_dt_s=float(args.interp_dt_s),
                joint=args.joint,
                joint_rad=math.radians(joint_deg),
            )
            print(f"set joint={args.joint} control_deg={joint_deg:.2f}")
            time.sleep(max(0.0, float(args.hold_s)))

        if not args.list_joints and not args.neutral and args.joint is None and not args.repl:
            parser.print_help()
            return 1
        return 0
    finally:
        servo.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
