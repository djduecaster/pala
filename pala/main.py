from __future__ import annotations

import argparse
import signal
import threading
import time
import os
import logging
import sys
from typing import Dict, Optional

from .config import load_config
from .types import PerceptionState, ActionPlan, HardwareCommand
from .perception import PerceptionNode, LatestFrameCache
from .perception.preview_tap import PreviewTapWriter
from .perception.frame_source import DummyFrameSource, CameraFrameSource
from .behavior import HoldBehaviorPolicy
from .behavior.manual import ManualBehaviorPolicy
from .control.performances import PerformanceLibrary, PerformanceSequencer
from .utils.manual_console import ManualConsole
from .control import TrajectoryExecutor
from .control.primitives import PrimitiveKind, HoldCommand
from .hardware import DummyServo, PCA9685Servo, ServoCalibration
from .utils import RateLimiter, LatestValue, maybe_logger

logger = logging.getLogger(__name__)

def main(argv: Optional[list[str]] = None) -> int:
    _configure_logging()
    args = _parse_cli_args([] if argv is None else argv)
    cfg = load_config(args.config)
    _apply_mode_override(cfg, args.mode)
    library = PerformanceLibrary(args.performances, cfg) if args.manual else None
    probe = None
    if sum((args.attention_probe, args.live_greeting, args.live_interaction)) > 1:
        raise ValueError("Choose only one model test mode")
    live_mode = args.live_greeting or args.live_interaction
    if args.attention_probe or live_mode:
        if not args.manual:
            raise ValueError("Model test modes require --manual; use the corresponding tools entry point")
        if args.probe_mock and cfg.mode != "dev":
            raise ValueError("--probe-mock is limited to dev mode")
        if args.live_interaction:
            from tools.live_interaction import LiveInteraction
            if not {"notice", "excite", "point_left", "point_right", "breathe", "breathe_sway"} <= library.performances.keys():
                raise ValueError("Live interaction requires the desk performance library")
            probe = LiveInteraction.from_args(args)
            probe.breathing = True
            probe.auto_arm = True
            probe.auto_arm_pending = True
        elif args.live_greeting:
            from tools.live_greeting import LiveGreeting
            probe = LiveGreeting.from_args(args)
        else:
            from tools.attention_probe import AttentionProbe
            probe = AttentionProbe.from_args(args)
    if args.manual and cfg.mode == "jetson_full":
        if not args.enable:
            raise ValueError("Manual hardware mode requires --enable")
        logger.info("Stop other servo owners. Establish physical zero before continuing.")
        if input("Type ZERO to confirm the known starting posture: ").strip() != "ZERO":
            logger.info("Startup canceled before hardware initialization")
            return 0
    max_runtime_s = _parse_max_runtime_s()
    run_log_dir = _init_run_log_dir(cfg)
    if run_log_dir:
        logger.info("run log scope=%s", run_log_dir)

    stop = threading.Event()
    graceful_requested = threading.Event()
    thread_failure_lock = threading.Lock()
    thread_failure: Dict[str, str] = {}

    # Shared state
    latest_perception = LatestValue[PerceptionState]()
    latest_action = LatestValue[ActionPlan]()
    latest_command = LatestValue[HardwareCommand]()
    latest_frame = LatestFrameCache()
    latest_performance = LatestValue()
    latest_execution = LatestValue()
    manual = ManualBehaviorPolicy(library) if args.manual else None
    sequencer = PerformanceSequencer(cfg) if args.manual else None
    console = ManualConsole(sys.stdin) if args.manual else None

    # Nodes
    perception = PerceptionNode(source=_build_frame_source(cfg))
    behavior = HoldBehaviorPolicy()
    executor = TrajectoryExecutor(cfg.joint_limits_rad, style_profiles=getattr(cfg, "style_profiles", None))
    preview_tap = _build_preview_tap(cfg)

    # Optional logging
    perception_log = maybe_logger(
        _scope_log_path(cfg.logging.perception_jsonl, run_log_dir) if cfg.logging.enabled else None
    )
    action_log = maybe_logger(
        _scope_log_path(cfg.logging.actions_jsonl, run_log_dir) if cfg.logging.enabled else None
    )

    interaction_log = maybe_logger(
        os.path.join(run_log_dir or "logs", "interaction.jsonl")
        if args.manual and cfg.logging.enabled else None
    )
    if args.manual and run_log_dir:
        import json
        with open(os.path.join(run_log_dir, "performances.json"), "w") as output:
            json.dump(library.raw, output, indent=2)
        with open(args.config) as source, open(os.path.join(run_log_dir, "robot.yaml"), "w") as output:
            output.write(source.read())

    def _handle_sig(_sig, _frame):
        if args.manual and _sig == signal.SIGTERM and not graceful_requested.is_set():
            graceful_requested.set()
        else:
            stop.set()

    # Initialize outputs only after validating input and preparing logging.
    servo = _build_servo(cfg)

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    def _record_thread_failure(name: str, exc: Exception) -> None:
        with thread_failure_lock:
            if not thread_failure:
                thread_failure["name"] = name
                thread_failure["error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("runtime thread crashed: %s", name)
        stop.set()

    def _thread_guard(name: str, fn):
        def _wrapped() -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - fail-fast loop supervision
                _record_thread_failure(name, exc)

        return _wrapped

    # --- Perception loop ---
    def perception_loop() -> None:
        rl = RateLimiter(cfg.loop_rates.perception_hz)
        last_print = time.monotonic()
        while not stop.is_set():
            st = perception.step()
            latest_perception.set(st, st.timestamp_monotonic_s)
            packet = perception.latest_packet()
            if packet is not None:
                latest_frame.set(packet.frame, mono_ns=packet.mono_ns, pts_ns=packet.pts_ns)
                cmd, _ = latest_command.get()
                preview_extra = _build_preview_extra(cfg, cmd)
                preview_tap.write_with_extra(
                    packet.frame,
                    mono_ns=packet.mono_ns,
                    pts_ns=packet.pts_ns,
                    extra=preview_extra,
                )

            if perception_log:
                perception_log.write(st)

            now = time.monotonic()
            if now - last_print >= 2.0:
                logger.info(
                    "perception fps=%.1f frame_id=%s age_ms=%.1f source_alive=%s",
                    st.fps or 0.0,
                    st.frame_id,
                    st.frame_age_ms or 0.0,
                    st.source_alive,
                )
                last_print = now

            rl.sleep()

    # --- Behavior loop ---
    def behavior_loop() -> None:
        rl = RateLimiter(cfg.loop_rates.behavior_hz)
        last_action_id: Optional[str] = None
        while not stop.is_set():
            if manual is not None:
                report, _ = latest_execution.get()
                plan = manual.step(report)
                latest_performance.set(plan, time.monotonic())
                if manual.status()["failure"]:
                    raise RuntimeError(manual.status()["failure"])
                rl.sleep()
                continue
            st, _ = latest_perception.get()
            action = behavior.step(st)
            ts = time.monotonic()
            latest_action.set(action, ts)

            if action_log:
                action_log.write(
                    {
                        "ts_wall_s": time.time(),
                        "ts_monotonic_s": ts,
                        "action": action,
                    }
                )

            if action.action_id != last_action_id:
                logger.info(
                    "decision primitive=%s style=%s conf=%.2f reason=%s",
                    action.primitive.value,
                    action.style,
                    action.confidence,
                    _short_text(action.explanation, max_chars=80),
                )
                last_action_id = action.action_id

            rl.sleep()

    # --- Control loop ---
    def control_loop() -> None:
        rl = RateLimiter(cfg.loop_rates.control_hz)
        last_ts = time.monotonic()
        last_performance_phase = None
        startup_hold_action = ActionPlan(
            primitive=PrimitiveKind.HOLD,
            command=HoldCommand(),
            confidence=0.1,
        )
        while not stop.is_set():
            action, _ = latest_action.get()
            if action is None:
                action = startup_hold_action

            now = time.monotonic()
            dt = now - last_ts
            last_ts = now

            if sequencer is not None:
                plan, _ = latest_performance.get()
                cmd, report = sequencer.tick(plan, dt)
                if report is not None:
                    latest_execution.set(report, now)
                    phase_key = (report.plan_id, report.step, report.phase, report.status)
                    if phase_key != last_performance_phase:
                        logger.info("performance=%s step=%d %s phase=%s status=%s",
                                    report.name, report.step, report.step_name, report.phase, report.status)
                        if action_log:
                            action_log.write({"ts_wall_s": time.time(), "action": sequencer.action, "execution": report})
                        last_performance_phase = phase_key
                    if interaction_log:
                        interaction_log.write({"type": "execution", "report": report, "command": cmd})
                    if report.status == "failed":
                        latest_command.set(cmd, cmd.timestamp_monotonic_s)
                        raise RuntimeError(report.reason)
            else:
                cmd = executor.step(action, dt)
            latest_command.set(cmd, cmd.timestamp_monotonic_s)

            rl.sleep()

    # --- Hardware loop ---
    def hardware_loop() -> None:
        rl = RateLimiter(cfg.loop_rates.hardware_hz)
        deadman_s = cfg.deadman_timeout_ms / 1000.0
        enabled = True
        last_state: Optional[str] = None
        received_command = False
        try:
            while not stop.is_set():
                cmd, ts = latest_command.get()
                now = time.monotonic()
                state = "enabled"
                if cmd is None or ts is None or (now - ts) > deadman_s:
                    if enabled:
                        servo.enable(False)
                        enabled = False
                    state = "deadman"
                    if manual is not None and received_command:
                        raise RuntimeError("Manual control command expired; outputs disabled")
                else:
                    received_command = True
                    if cmd.enable is False:
                        if enabled:
                            servo.enable(False)
                            enabled = False
                        state = "commanded_disable"
                    else:
                        if not enabled:
                            servo.enable(True)
                            enabled = True
                        servo.set_angles(cmd.joint_angles_rad)
                        state = "enabled"

                if state != last_state:
                    logger.info("hardware state=%s", state)
                    last_state = state

                rl.sleep()
        finally:
            servo.enable(False)

    threads = [
        threading.Thread(target=_thread_guard("perception", perception_loop), daemon=True),
        threading.Thread(target=_thread_guard("behavior", behavior_loop), daemon=True),
        threading.Thread(target=_thread_guard("control", control_loop), daemon=True),
        threading.Thread(target=_thread_guard("hardware", hardware_loop), daemon=True),
    ]

    recorder = None
    if args.record_pov:
        from pathlib import Path
        from tools.pov_recorder import PovRecorder
        from uuid import uuid4
        recording_dir = Path(run_log_dir or 'logs/recordings') / ('pov_' + uuid4().hex[:8])
        recorder = PovRecorder(latest_frame, recording_dir, fragmented=True, max_lag_s=None)
        recorder.start()

    for t in threads:
        t.start()

    start_time = time.monotonic()
    shutdown_started = None
    if manual is not None:
        if probe is not None:
            logger.info("Live model test: wait for rest, then arm; help lists commands" if live_mode else
                        "Attention probe: wait for PROBE READY, then type 1–4 + Enter; help lists commands")
        else:
            logger.info("Manual interaction: greet | attend | settle | demo | status | help | shutdown/q | stop")
        logger.info("Startup enters rest. Busy requests are rejected. Ctrl-C/stop disables; shutdown/q returns to zero first.")
    try:
        while not stop.is_set():
            with thread_failure_lock:
                if thread_failure:
                    break
            if max_runtime_s is not None and (time.monotonic() - start_time) >= max_runtime_s:
                if manual is None:
                    stop.set()
                    break
                graceful_requested.set()
            if manual is not None:
                commands = console.poll()
                for text in commands:
                    token = text.strip().lower()
                    if not token:
                        continue
                    if token in {"stop", "abort"}:
                        stop.set()
                        break
                    if token in {"help", "?"}:
                        if probe is not None:
                            if args.live_interaction:
                                from tools.live_interaction import HELP
                            elif args.live_greeting:
                                from tools.live_greeting import HELP
                            else:
                                from tools.attention_probe import HELP
                            logger.info(HELP)
                            continue
                        logger.info("greet: greet once and stay attentive; attend: attention pose; settle: return to rest; "
                                    "demo: greet, attend, settle; status: current progress; shutdown/q: zero then disable; stop/Ctrl-C: disable immediately")
                        continue
                    if token == "status":
                        logger.info("manual status=%s", manual.status())
                        if live_mode:
                            logger.info("live armed=%s request_inflight=%s", probe.armed, bool(probe.inflight))
                        continue
                    if token in {"shutdown", "q", "quit"}:
                        graceful_requested.set()
                        continue
                    if probe is not None:
                        probe.command(token, manual.status())
                        continue
                    accepted, message = manual.request(token)
                    logger.info("manual request=%s accepted=%s: %s", token, accepted, message)
                    if interaction_log:
                        interaction_log.write({"type": "request", "command": token, "accepted": accepted, "message": message})
                if stop.is_set():
                    break
                if graceful_requested.is_set() and shutdown_started is None:
                    accepted, message = manual.request("shutdown")
                    logger.info("manual shutdown: %s", message)
                    shutdown_started = time.monotonic()
                state = manual.status()
                if probe is not None and not graceful_requested.is_set():
                    probe.tick(latest_frame.get(max_age_ms=500), state)
                    if live_mode:
                        request = probe.take_request()
                        if request is not None:
                            accepted, message = manual.request(request)
                            if args.live_interaction:
                                probe.motion_result(request, accepted)
                            probe.record({"type": "motion_request", "command": request, "accepted": accepted, "message": message})
                            logger.info("live request=%s accepted=%s: %s", request, accepted, message)
                if state["finished"]:
                    stop.set()
                elif shutdown_started is not None and time.monotonic() - shutdown_started > 30:
                    raise RuntimeError("Graceful shutdown exceeded 30 seconds; disabling outputs")
            time.sleep(0.05 if manual is not None else 0.1)
    finally:
        stop.set()
        if probe is not None:
            try:
                probe.close()
            except Exception:
                logger.exception("probe log close failed")
        for t in threads:
            t.join(timeout=1.0)

        alive = [t.name or f"thread_{idx}" for idx, t in enumerate(threads) if t.is_alive()]
        if alive:
            logger.warning("shutdown proceeding with live threads=%s", ",".join(alive))

        try:
            perception.shutdown()
        except Exception:  # noqa: BLE001 - shutdown should not crash process exit
            logger.exception("perception shutdown failed")
        if hasattr(behavior, "shutdown"):
            try:
                behavior.shutdown()
            except Exception:  # noqa: BLE001 - shutdown should not crash process exit
                logger.exception("behavior shutdown failed")
        try:
            servo.shutdown()
        except Exception:  # noqa: BLE001 - shutdown should not crash process exit
            logger.exception("servo shutdown failed")
        try:
            preview_tap.close()
        except Exception:  # noqa: BLE001 - shutdown should not crash process exit
            logger.exception("preview tap close failed")
        if recorder is not None:
            recorder.close()
        if perception_log:
            try:
                perception_log.close()
            except Exception:  # noqa: BLE001 - shutdown should not crash process exit
                logger.exception("perception log close failed")
        if interaction_log:
            interaction_log.close()
        if action_log:
            try:
                action_log.close()
            except Exception:  # noqa: BLE001 - shutdown should not crash process exit
                logger.exception("action log close failed")

    with thread_failure_lock:
        failed = dict(thread_failure)
    if failed:
        logger.error("runtime exiting due to thread failure thread=%s error=%s", failed["name"], failed["error"])
        return 1

    logger.info("clean shutdown")
    return 0


def _build_servo(cfg) -> DummyServo:
    # TODO: Add a jetson_perception mode that uses DummyServo but real camera.
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
    if cfg.mode not in {"jetson_perception", "jetson_full"}:
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


def _build_preview_tap(cfg) -> PreviewTapWriter:
    tap_cfg = getattr(cfg, "telemetry_preview", None)
    if tap_cfg is None:
        return PreviewTapWriter(
            enabled=False,
            jpeg_path="logs/telemetry/preview/latest.jpg",
            meta_path="logs/telemetry/preview/latest.json",
            max_hz=4.0,
            max_width=640,
            max_height=360,
            jpeg_quality=65,
        )

    return PreviewTapWriter(
        enabled=bool(tap_cfg.enabled),
        jpeg_path=str(tap_cfg.jpeg_path),
        meta_path=str(tap_cfg.meta_path),
        max_hz=float(tap_cfg.max_hz),
        max_width=int(tap_cfg.max_width),
        max_height=int(tap_cfg.max_height),
        jpeg_quality=int(tap_cfg.jpeg_quality),
    )


def _build_preview_extra(cfg, cmd: Optional[HardwareCommand]) -> Optional[Dict[str, object]]:
    if cmd is None:
        return None
    angles = [float(v) for v in cmd.joint_angles_rad]
    names = [str(v) for v in getattr(cfg, "joint_names", [])]
    if len(names) != len(angles):
        names = [f"joint_{i}" for i in range(len(angles))]

    return {
        "command": {
            "joint_names": names,
            "joint_angles_rad": angles,
            "enable": bool(cmd.enable),
            "timestamp_monotonic_s": float(cmd.timestamp_monotonic_s),
        }
    }


def _short_text(value: Optional[str], *, max_chars: int) -> str:
    token = " ".join(str(value or "").split()).strip()
    if not token:
        return "-"
    if len(token) <= max_chars:
        return token
    return token[: max_chars - 3] + "..."


def _init_run_log_dir(cfg) -> Optional[str]:
    scope_enabled = str(os.getenv("PALA_RUN_SCOPED_LOGS", "1")).strip().lower()
    if scope_enabled in {"0", "false", "no", "off"}:
        return None
    has_log_targets = bool(
        getattr(cfg.logging, "enabled", False)
    )
    if not has_log_targets:
        return None
    run_id = os.getenv("PALA_RUN_ID")
    if not run_id:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    root = os.getenv("PALA_RUN_LOG_ROOT", "logs/runs")
    path = os.path.join(root, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def _scope_log_path(path: Optional[str], run_log_dir: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if not run_log_dir:
        return path
    filename = os.path.basename(path)
    if not filename:
        return path
    return os.path.join(run_log_dir, filename)


def _parse_max_runtime_s() -> Optional[float]:
    raw = os.getenv("PALA_MAX_RUNTIME_S")
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError("PALA_MAX_RUNTIME_S must be a number") from None


def _configure_logging() -> None:
    level_name = os.getenv("PALA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_cli_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PALA runtime")
    parser.add_argument(
        "--config",
        default="config/robot.yaml",
        help="Path to runtime config YAML",
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "jetson_perception", "jetson_full"],
        help="Override mode from config",
    )
    parser.add_argument("--record-pov", action="store_true", help="Record up to ten minutes of silent 720p/20fps POV video; requires ffmpeg")
    parser.add_argument("--manual", action="store_true", help="Enable manually triggered deterministic gestures")
    parser.add_argument("--enable", action="store_true", help="Required for --manual with jetson_full")
    parser.add_argument("--performances", default="config/performances.json", help="Validated gesture library for --manual")
    parser.add_argument("--live-interaction", action="store_true", help="Supervised notice, greet, excite, settle interaction")
    parser.add_argument("--live-greeting", action="store_true", help="Explicitly armed Gemini-triggered greeting test")
    parser.add_argument("--attention-probe", action="store_true", help="Four timed Gemini observations at rest; no model actuation")
    parser.add_argument("--probe-mock", action="store_true", help="Dev-only fake model for offline testing")
    parser.add_argument("--gemini-model", default=os.getenv("PALA_GEMINI_MODEL"), help="Gemini model ID (or PALA_GEMINI_MODEL)")
    parser.add_argument("--gemini-key-file", help="File containing only API key; prefer a path outside the repository")
    parser.add_argument("--probe-output", default="logs/attention_probe", help="Parent directory for unique probe sessions")
    return parser.parse_args(argv)


def _apply_mode_override(cfg, mode_override: Optional[str]) -> None:
    if mode_override:
        cfg.mode = mode_override


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
