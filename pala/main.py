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
from .planner import (
    HeuristicPlanner,
    AsyncOrchestratorPlanner,
    AsyncSceneSummarizer,
    MemoryManager,
    MemoryManagerConfig,
    TimelineConfig,
    TimelineWriter,
)
from .behavior import BehaviorPolicy
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

    stop = threading.Event()
    thread_failure_lock = threading.Lock()
    thread_failure: Dict[str, str] = {}

    # Shared state
    latest_perception = LatestValue[PerceptionState]()
    latest_action = LatestValue[ActionPlan]()
    latest_command = LatestValue[HardwareCommand]()
    latest_frame = LatestFrameCache()

    # Nodes
    perception = PerceptionNode(source=_build_frame_source(cfg), detector=_build_detector(cfg))
    planner = _build_planner(cfg, latest_frame)
    behavior = BehaviorPolicy(planner=planner, dwell_s=2.0, cooldown_s=1.0)
    executor = TrajectoryExecutor(cfg.joint_limits_rad, style_profiles=getattr(cfg, "style_profiles", None))
    servo = _build_servo(cfg)
    preview_tap = _build_preview_tap(cfg)

    # Optional logging
    perception_log = maybe_logger(cfg.logging.perception_jsonl if cfg.logging.enabled else None)
    action_log = maybe_logger(cfg.logging.actions_jsonl if cfg.logging.enabled else None)

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
        while not stop.is_set():
            st, _ = latest_perception.get()
            action = behavior.step(st)
            ts = time.monotonic()
            latest_action.set(action, ts)

            if action_log:
                action_log.write(action)

            rl.sleep()

    # --- Control loop ---
    def control_loop() -> None:
        rl = RateLimiter(cfg.loop_rates.control_hz)
        last_ts = time.monotonic()
        startup_hold_action = ActionPlan(
            primitive=PrimitiveKind.HOLD,
            command=HoldCommand(),
            confidence=0.1,
            cancel_current=True,
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

            if now - last_print >= 2.0:
                logger.info("hardware state=%s", state)
                last_print = now

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
        if hasattr(planner, "shutdown"):
            planner.shutdown()
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


def _build_planner(cfg, latest_frame: LatestFrameCache):
    if getattr(cfg, "cosmos", None) and cfg.cosmos.enabled:
        base_url = os.getenv("PALA_COSMOS_BASE_URL") or cfg.cosmos.base_url
        api_key = os.getenv("PALA_COSMOS_API_KEY")
        model = os.getenv("PALA_COSMOS_MODEL") or cfg.cosmos.model
        planner_prompt = os.getenv("PALA_COSMOS_PROMPT") or cfg.cosmos.planner_prompt
        timeline = TimelineWriter(
            TimelineConfig(
                enabled=True,
                jsonl_path=cfg.cosmos.orchestrator_timeline_jsonl_path,
            )
        )
        memory_events = max(
            64,
            cfg.cosmos.memory_recent_decisions + cfg.cosmos.memory_recent_reasoning + (2 * cfg.cosmos.memory_recent_summaries),
        )
        memory_manager = MemoryManager(
            MemoryManagerConfig(
                enabled=cfg.cosmos.memory_enabled,
                jsonl_path=cfg.cosmos.memory_jsonl_path,
                recent_events=memory_events,
                digest_items=0,
                distill_every_n_events=0,
            )
        )
        summarizer = None
        if cfg.cosmos.summarizer_enabled:
            summarizer = AsyncSceneSummarizer(
                frame_cache=latest_frame,
                provider=cfg.cosmos.provider,
                base_url=base_url,
                api_key=api_key,
                model=model,
                policy_version=cfg.cosmos.policy_version,
                summarizer_hz=cfg.cosmos.summarizer_hz,
                max_frame_age_ms=cfg.cosmos.max_frame_age_ms,
                summary_window_s=cfg.cosmos.summary_window_s,
                summary_max_frames=cfg.cosmos.summary_max_frames,
                summary_max_width=cfg.cosmos.summary_max_width,
                summary_jpeg_quality=cfg.cosmos.summary_jpeg_quality,
                request_timeout_ms=cfg.cosmos.summarizer_timeout_ms,
                request_min_fresh_frames=cfg.cosmos.request_min_fresh_frames,
                timeline=timeline,
                memory_manager=memory_manager,
            )
        return AsyncOrchestratorPlanner(
            frame_cache=latest_frame,
            fallback=HeuristicPlanner(),
            summarizer=summarizer,
            memory_manager=memory_manager,
            timeline=timeline,
            provider=cfg.cosmos.provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            planner_prompt=planner_prompt,
            runtime_mode=cfg.mode,
            policy_version=cfg.cosmos.policy_version,
            policy_identity=cfg.cosmos.policy_identity,
            identity_file_path=cfg.cosmos.identity_file_path,
            policy_capabilities=cfg.cosmos.policy_capabilities,
            policy_safety=cfg.cosmos.policy_safety,
            policy_style=cfg.cosmos.policy_style,
            policy_output_contract=cfg.cosmos.policy_output_contract,
            orchestrator_hz=cfg.cosmos.planner_hz,
            max_frame_age_ms=cfg.cosmos.max_frame_age_ms,
            video_window_s=cfg.cosmos.video_window_s,
            video_max_frames=cfg.cosmos.video_max_frames,
            video_max_width=cfg.cosmos.video_max_width,
            video_jpeg_quality=cfg.cosmos.video_jpeg_quality,
            request_timeout_ms=cfg.cosmos.request_timeout_ms,
            response_ttl_ms=cfg.cosmos.response_ttl_ms,
            summary_ttl_ms=cfg.cosmos.summary_ttl_ms,
            planner_strict_schema=cfg.cosmos.planner_strict_schema,
            planner_allow_frame_fetch=cfg.cosmos.planner_allow_frame_fetch,
            planner_max_tool_calls_per_cycle=cfg.cosmos.planner_max_tool_calls_per_cycle,
            planner_include_images_first_pass=cfg.cosmos.planner_include_images_first_pass,
            memory_recent_decisions=cfg.cosmos.memory_recent_decisions,
            memory_recent_summaries=cfg.cosmos.memory_recent_summaries,
            memory_recent_reasoning=cfg.cosmos.memory_recent_reasoning,
            memory_enabled=cfg.cosmos.memory_enabled,
            memory_jsonl_path=cfg.cosmos.memory_jsonl_path,
            memory_recent_events=cfg.cosmos.memory_recent_events,
            memory_digest_items=cfg.cosmos.memory_digest_items,
            memory_distill_every_n_events=cfg.cosmos.memory_distill_every_n_events,
            context_max_transcript_items=cfg.cosmos.context_max_transcript_items,
            context_transcript_max_items=cfg.cosmos.context_transcript_max_items,
            context_transcript_per_type_max_items=cfg.cosmos.context_transcript_per_type_max_items,
            context_transcript_max_chars=cfg.cosmos.context_transcript_max_chars,
            context_memory_digest_max_items=cfg.cosmos.context_memory_digest_max_items,
            decision_repeat_detector_window=cfg.cosmos.decision_repeat_detector_window,
            inflight_guard_enabled=cfg.cosmos.inflight_guard_enabled,
            request_min_fresh_frames=cfg.cosmos.request_min_fresh_frames,
            reasoning_probe_enabled=cfg.cosmos.reasoning_probe_enabled,
            reasoning_probe_hz=cfg.cosmos.reasoning_probe_hz,
            reasoning_probe_timeout_ms=cfg.cosmos.reasoning_probe_timeout_ms,
            reasoning_probe_max_tokens=cfg.cosmos.reasoning_probe_max_tokens,
            commitment_ttl_ms=cfg.cosmos.commitment_ttl_ms,
            planner_self_critique_enabled=cfg.cosmos.planner_self_critique_enabled,
            planner_self_critique_confidence=cfg.cosmos.planner_self_critique_confidence,
            planner_self_critique_margin=cfg.cosmos.planner_self_critique_margin,
            planner_self_critique_max_calls_per_cycle=cfg.cosmos.planner_self_critique_max_calls_per_cycle,
        )
    return HeuristicPlanner()


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
