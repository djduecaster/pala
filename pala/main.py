from __future__ import annotations

import signal
import threading
import time
import os
from typing import Optional

from .config import load_config
from .types import PerceptionState, ActionPlan, HardwareCommand
from .perception import PerceptionNode
from .perception.frame_source import DummyFrameSource, CameraFrameSource
from .planner import HeuristicPlanner
from .behavior import BehaviorPolicy
from .control import TrajectoryExecutor
from .hardware import DummyServo, PCA9685Servo, ServoCalibration
from .utils import RateLimiter, LatestValue, maybe_logger


def main() -> int:
    cfg = load_config("config/robot.yaml")
    max_runtime_s = _parse_max_runtime_s()

    stop = threading.Event()

    # Shared state
    latest_perception = LatestValue[PerceptionState]()
    latest_action = LatestValue[ActionPlan]()
    latest_command = LatestValue[HardwareCommand]()

    # Nodes
    perception = PerceptionNode(source=_build_frame_source(cfg))
    planner = HeuristicPlanner()
    behavior = BehaviorPolicy(planner=planner, dwell_s=2.0, cooldown_s=1.0)
    executor = TrajectoryExecutor(cfg.joint_limits_rad)
    servo = _build_servo(cfg)

    # Optional logging
    perception_log = maybe_logger(cfg.logging.perception_jsonl if cfg.logging.enabled else None)
    action_log = maybe_logger(cfg.logging.actions_jsonl if cfg.logging.enabled else None)

    def _handle_sig(_sig, _frame):
        stop.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    # --- Perception loop ---
    def perception_loop() -> None:
        rl = RateLimiter(cfg.loop_rates.perception_hz)
        last_print = time.monotonic()
        while not stop.is_set():
            st = perception.step()
            latest_perception.set(st, st.timestamp_monotonic_s)

            now = time.monotonic()
            if now - last_print >= 2.0:
                zone = st.debug.get("zone_hint") if st.debug else None
                print(f"[perception] fps={st.fps or 0:.1f} zone={zone}")
                last_print = now

            rl.sleep()

    # --- Behavior loop ---
    def behavior_loop() -> None:
        rl = RateLimiter(cfg.loop_rates.behavior_hz)
        while not stop.is_set():
            st, _ = latest_perception.get()
            action = behavior.step(st)
            ts = time.monotonic()
            latest_action.set(action, ts)

            if perception_log and st is not None:
                perception_log.write(st)
            if action_log:
                action_log.write(action)

            rl.sleep()

    # --- Control loop ---
    def control_loop() -> None:
        rl = RateLimiter(cfg.loop_rates.control_hz)
        last_ts = time.monotonic()
        while not stop.is_set():
            action, _ = latest_action.get()
            if action is None:
                action = ActionPlan(primitive="hold", params={}, confidence=0.1)

            now = time.monotonic()
            dt = now - last_ts
            last_ts = now

            cmd = executor.step(action, dt)
            latest_command.set(cmd, cmd.timestamp_monotonic_s)

            rl.sleep()

    # --- Hardware loop ---
    def hardware_loop() -> None:
        rl = RateLimiter(cfg.loop_rates.hardware_hz)
        deadman_s = cfg.deadman_timeout_ms / 1000.0
        enabled = True
        last_print = time.monotonic()
        while not stop.is_set():
            cmd, ts = latest_command.get()
            now = time.monotonic()
            if cmd is None or ts is None or (now - ts) > deadman_s:
                if enabled:
                    servo.enable(False)
                    enabled = False
            else:
                if not enabled:
                    servo.enable(True)
                    enabled = True
                servo.set_angles(cmd.joint_angles_rad)

            if now - last_print >= 2.0:
                state = "enabled" if enabled else "deadman"
                print(f"[hardware] state={state}")
                last_print = now

            rl.sleep()

    threads = [
        threading.Thread(target=perception_loop, daemon=True),
        threading.Thread(target=behavior_loop, daemon=True),
        threading.Thread(target=control_loop, daemon=True),
        threading.Thread(target=hardware_loop, daemon=True),
    ]

    for t in threads:
        t.start()

    start_time = time.monotonic()
    try:
        while not stop.is_set():
            if max_runtime_s is not None and (time.monotonic() - start_time) >= max_runtime_s:
                stop.set()
                break
            time.sleep(0.1)
    finally:
        stop.set()
        perception.shutdown()
        servo.shutdown()
        if perception_log:
            perception_log.close()
        if action_log:
            action_log.close()

    for t in threads:
        t.join(timeout=1.0)

    print("[main] clean shutdown")
    return 0


def _build_servo(cfg) -> DummyServo:
    if cfg.mode != "jetson_full":
        return DummyServo(log_every=20)

    servo_cal = cfg.servo_calibration
    if not servo_cal:
        raise ValueError("servo_calibration is required for jetson_full mode")

    channels = list(servo_cal.get("channels", []))
    if len(channels) != len(cfg.joint_names):
        raise ValueError("servo_calibration.channels must match joint_names length")

    per_joint = servo_cal.get("per_joint", {})
    if not isinstance(per_joint, dict):
        raise ValueError("servo_calibration.per_joint must be a mapping")

    min_pulse = []
    max_pulse = []
    scales = []
    offsets = []
    reverses = []
    for name in cfg.joint_names:
        if name not in per_joint:
            raise ValueError(f"servo_calibration.per_joint missing '{name}'")
        joint_cal = per_joint[name] or {}
        min_pulse.append(float(joint_cal["min_pulse"]))
        max_pulse.append(float(joint_cal["max_pulse"]))
        scales.append(float(joint_cal["angle_scale"]))
        offsets.append(float(joint_cal["angle_offset"]))
        reverses.append(bool(joint_cal["reverse"]))

    calibration = ServoCalibration(
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
    return PCA9685Servo(calibration)


def _build_frame_source(cfg):
    if cfg.mode != "jetson_full":
        return DummyFrameSource()

    from .hardware.camera_gst import GStreamerCamera

    camera = GStreamerCamera(
        device=cfg.camera.device,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fps=cfg.camera.fps,
        pipeline=cfg.camera.pipeline,
    )
    return CameraFrameSource(camera)


def _parse_max_runtime_s() -> Optional[float]:
    raw = os.getenv("PALA_MAX_RUNTIME_S")
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError("PALA_MAX_RUNTIME_S must be a number") from None


if __name__ == "__main__":
    raise SystemExit(main())
