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
from .perception.detector import DummyDetector, JetsonDetector, DeepStreamDetector
from .perception.frame_source import DummyFrameSource, CameraFrameSource
from .behavior import BehaviorPolicy, BehaviorPolicyConfig
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
    max_runtime_s = _parse_max_runtime_s()
    run_log_dir = _init_run_log_dir(cfg)
    if run_log_dir:
        logger.info("run log scope=%s", run_log_dir)

    stop = threading.Event()
    thread_failure_lock = threading.Lock()
    thread_failure: Dict[str, str] = {}

    # Shared state
    latest_perception = LatestValue[PerceptionState]()
    latest_action = LatestValue[ActionPlan]()
    latest_command = LatestValue[HardwareCommand]()
    latest_control_state = LatestValue[object]()
    latest_frame = LatestFrameCache()

    # Nodes
    perception = PerceptionNode(source=_build_frame_source(cfg), detector=_build_detector(cfg))
    behavior = BehaviorPolicy(
        config=_build_behavior_config(cfg, run_log_dir=run_log_dir),
        frame_cache=latest_frame,
        dwell_s=2.0,
        cooldown_s=1.0,
    )
    executor = TrajectoryExecutor(cfg.joint_limits_rad, style_profiles=getattr(cfg, "style_profiles", None))
    servo = _build_servo(cfg)
    preview_tap = _build_preview_tap(cfg)

    # Optional logging
    perception_log = maybe_logger(
        _scope_log_path(cfg.logging.perception_jsonl, run_log_dir) if cfg.logging.enabled else None
    )
    action_log = maybe_logger(
        _scope_log_path(cfg.logging.actions_jsonl, run_log_dir) if cfg.logging.enabled else None
    )

    def _handle_sig(_sig, _frame):
        stop.set()

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
                zone = st.debug.get("zone_hint") if st.debug else None
                logger.info("perception fps=%.1f zone=%s", st.fps or 0.0, zone)
                last_print = now

            rl.sleep()

    # --- Behavior loop ---
    def behavior_loop() -> None:
        rl = RateLimiter(cfg.loop_rates.behavior_hz)
        last_action_id: Optional[str] = None
        last_env_summary: Optional[str] = None
        while not stop.is_set():
            st, _ = latest_perception.get()
            control_state, control_ts = latest_control_state.get()
            if control_state is not None and control_ts is not None and hasattr(behavior, "set_control_state"):
                behavior.set_control_state(control_state)
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
                summary = _latest_env_summary(behavior)
                if summary:
                    logger.info(
                        "decision primitive=%s style=%s conf=%.2f reason=%s env=%s",
                        action.primitive.value,
                        action.style,
                        action.confidence,
                        _short_text(action.explanation, max_chars=80),
                        _short_text(summary, max_chars=110),
                    )
                else:
                    logger.info(
                        "decision primitive=%s style=%s conf=%.2f reason=%s",
                        action.primitive.value,
                        action.style,
                        action.confidence,
                        _short_text(action.explanation, max_chars=80),
                    )
                last_action_id = action.action_id

            summary = _latest_env_summary(behavior)
            if summary and summary != last_env_summary:
                logger.info("env summary=%s", _short_text(summary, max_chars=140))
                last_env_summary = summary

            rl.sleep()

    # --- Control loop ---
    def control_loop() -> None:
        rl = RateLimiter(cfg.loop_rates.control_hz)
        last_ts = time.monotonic()
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

            cmd = executor.step(action, dt)
            latest_command.set(cmd, cmd.timestamp_monotonic_s)
            latest_control_state.set(executor.control_state, now)

            rl.sleep()

    # --- Hardware loop ---
    def hardware_loop() -> None:
        rl = RateLimiter(cfg.loop_rates.hardware_hz)
        deadman_s = cfg.deadman_timeout_ms / 1000.0
        enabled = True
        last_state: Optional[str] = None
        while not stop.is_set():
            cmd, ts = latest_command.get()
            now = time.monotonic()
            state = "enabled"
            if cmd is None or ts is None or (now - ts) > deadman_s:
                if enabled:
                    servo.enable(False)
                    enabled = False
                state = "deadman"
            else:
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

    threads = [
        threading.Thread(target=_thread_guard("perception", perception_loop), daemon=True),
        threading.Thread(target=_thread_guard("behavior", behavior_loop), daemon=True),
        threading.Thread(target=_thread_guard("control", control_loop), daemon=True),
        threading.Thread(target=_thread_guard("hardware", hardware_loop), daemon=True),
    ]

    for t in threads:
        t.start()

    start_time = time.monotonic()
    try:
        while not stop.is_set():
            with thread_failure_lock:
                if thread_failure:
                    break
            if max_runtime_s is not None and (time.monotonic() - start_time) >= max_runtime_s:
                stop.set()
                break
            time.sleep(0.1)
    finally:
        stop.set()
        perception.shutdown()
        if hasattr(behavior, "shutdown"):
            behavior.shutdown()
        servo.shutdown()
        preview_tap.close()
        if perception_log:
            perception_log.close()
        if action_log:
            action_log.close()

    for t in threads:
        t.join(timeout=1.0)

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
    # TODO: Allow jetson_perception to use GStreamer camera with dummy control/servo.
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


def _build_detector(cfg):
    if cfg.detector:
        if cfg.detector == "deepstream":
            if cfg.mode == "jetson_full":
                preflight_error = _deepstream_preflight_error()
                if preflight_error is not None:
                    logger.warning(
                        "DeepStream detector unavailable at startup (%s). "
                        "Using no-op detector (zero local detections).",
                        preflight_error,
                    )
                    return _NoopDetector(reason=preflight_error)
            return DeepStreamDetector(
                config_path=cfg.deepstream.config_path,
                person_class_id=cfg.deepstream.person_class_id,
                conf_threshold=cfg.deepstream.conf_threshold,
            )
        if cfg.detector == "jetson":
            return JetsonDetector()
        if cfg.detector == "dummy":
            return DummyDetector()
        raise ValueError(f"Unknown detector backend: {cfg.detector}")
    if cfg.mode == "jetson_full":
        return JetsonDetector()
    return DummyDetector()


class _NoopDetector:
    """Fallback detector used when DeepStream dependencies are missing at startup."""

    def __init__(self, *, reason: str):
        self._reason = reason

    def detect(self, _frame):
        return []

    def shutdown(self) -> None:
        return None


def _deepstream_preflight_error() -> Optional[str]:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - startup preflight diagnostics
        return f"missing_gstreamer_bindings:{type(exc).__name__}:{exc}"
    try:
        import pyds  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - startup preflight diagnostics
        return f"missing_pyds:{type(exc).__name__}:{exc}"
    return None


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


def _latest_env_summary(behavior: BehaviorPolicy) -> Optional[str]:
    world_state = getattr(behavior, "world_state", None)
    if world_state is None or not hasattr(world_state, "snapshot"):
        return None
    try:
        snap = world_state.snapshot()
    except Exception:  # noqa: BLE001 - best effort for logging only
        return None
    if not isinstance(snap, dict):
        return None
    env = snap.get("latest_env_snapshot")
    if not isinstance(env, dict):
        return None
    summary = env.get("summary")
    if summary is None:
        return None
    token = " ".join(str(summary).split()).strip()
    return token if token else None


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
        or (getattr(cfg, "cosmos", None) is not None and getattr(cfg.cosmos, "enabled", False))
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


def _build_behavior_config(cfg, *, run_log_dir: Optional[str] = None) -> BehaviorPolicyConfig:
    cosmos = getattr(cfg, "cosmos", None)
    if cosmos is None:
        return BehaviorPolicyConfig(remote_enabled=False)

    base_url = os.getenv("PALA_COSMOS_BASE_URL") or cosmos.base_url
    api_key = os.getenv("PALA_COSMOS_API_KEY")
    model = os.getenv("PALA_COSMOS_MODEL") or cosmos.model
    planner_prompt = os.getenv("PALA_COSMOS_PROMPT") or cosmos.planner_prompt

    env_log_path = _scope_log_path(
        str(getattr(cosmos, "behavior_env_log_path", "logs/behavior_env.jsonl")),
        run_log_dir,
    )
    planner_log_path = _scope_log_path(
        str(getattr(cosmos, "behavior_planner_log_path", "logs/behavior_planner.jsonl")),
        run_log_dir,
    )
    reasoning_log_path = _scope_log_path(
        str(getattr(cosmos, "behavior_reasoning_log_path", "logs/behavior_reasoning.jsonl")),
        run_log_dir,
    )
    trace_log_path = _scope_log_path(
        str(getattr(cosmos, "behavior_trace_log_path", "logs/behavior_trace.jsonl")),
        run_log_dir,
    )
    remote_enabled = bool(cosmos.enabled)
    idle_after_cfg = float(getattr(cosmos, "idle_after_s", 6.0))
    idle_glance_after_cfg = float(getattr(cosmos, "idle_glance_after_s", 10.0))
    arbiter_margin_cfg = float(getattr(cosmos, "arbiter_base_margin", 0.10))
    if not remote_enabled:
        idle_after_cfg = 0.0
        idle_glance_after_cfg = 0.8
        arbiter_margin_cfg = 0.05

    return BehaviorPolicyConfig(
        remote_enabled=remote_enabled,
        base_url=None if base_url in (None, "") else str(base_url),
        api_key=None if api_key in (None, "") else str(api_key),
        model=str(model),
        request_timeout_ms=int(getattr(cosmos, "request_timeout_ms", 6000)),
        error_backoff_s=float(getattr(cosmos, "behavior_error_backoff_s", 1.5)),
        client_error_backoff_s=float(getattr(cosmos, "behavior_client_error_backoff_s", 5.0)),
        env_hz=float(getattr(cosmos, "summarizer_hz", 0.25)),
        planner_hz=float(getattr(cosmos, "planner_hz", 0.5)),
        planner_event_delta_threshold=float(getattr(cosmos, "planner_event_delta_threshold", 0.65)),
        planner_event_cooldown_s=float(getattr(cosmos, "planner_event_cooldown_s", 0.7)),
        max_frame_age_ms=int(getattr(cosmos, "max_frame_age_ms", 500)),
        frame_window_s=float(getattr(cosmos, "summary_window_s", 6.0)),
        env_max_frames=int(getattr(cosmos, "summary_max_frames", 1)),
        planner_max_frames=int(getattr(cosmos, "planner_max_frames", 1)),
        frame_max_width=int(getattr(cosmos, "summary_max_width", 320)),
        frame_jpeg_quality=int(getattr(cosmos, "summary_jpeg_quality", 55)),
        request_min_fresh_frames=int(getattr(cosmos, "request_min_fresh_frames", 1)),
        planner_include_latest_frame=bool(getattr(cosmos, "planner_include_latest_frame", True)),
        env_max_tokens=int(getattr(cosmos, "env_max_tokens", 600)),
        planner_max_tokens=int(getattr(cosmos, "planner_max_tokens", 360)),
        proposer_max_age_s=max(0.2, float(getattr(cosmos, "response_ttl_ms", 2000)) / 1000.0),
        planner_max_proposals=int(getattr(cosmos, "planner_max_proposals", 1)),
        planner_use_env_context=bool(getattr(cosmos, "planner_use_env_context", True)),
        arbiter_min_dwell_s=float(getattr(cosmos, "arbiter_min_dwell_s", 1.2)),
        arbiter_base_margin=arbiter_margin_cfg,
        arbiter_takeover_no_signal_streak=int(getattr(cosmos, "arbiter_takeover_no_signal_streak", 2)),
        arbiter_takeover_no_commit_s=float(getattr(cosmos, "arbiter_takeover_no_commit_s", 2.0)),
        idle_after_s=idle_after_cfg,
        idle_glance_after_s=idle_glance_after_cfg,
        policy_identity=str(getattr(cosmos, "policy_identity", "You are PALA.")),
        policy_capabilities=str(getattr(cosmos, "policy_capabilities", "")),
        policy_safety=str(getattr(cosmos, "policy_safety", "")),
        policy_style=str(getattr(cosmos, "policy_style", "")),
        planner_prompt=str(planner_prompt),
        env_log_path=env_log_path,
        planner_log_path=planner_log_path,
        reasoning_log_path=reasoning_log_path,
        trace_log_path=trace_log_path,
    )


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
    return parser.parse_args(argv)


def _apply_mode_override(cfg, mode_override: Optional[str]) -> None:
    if mode_override:
        cfg.mode = mode_override

    mode = str(cfg.mode).strip().lower()
    if mode == "dev":
        cfg.detector = "dummy"
        if getattr(cfg, "cosmos", None) is not None:
            cfg.cosmos.enabled = False
        return

    if mode in {"jetson_perception", "jetson_full"} and str(cfg.detector).strip().lower() == "dummy":
        cfg.detector = "deepstream"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
