from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import yaml

from ..types.style_profiles import default_style_profiles


@dataclass
class LoopRates:
    perception_hz: float
    behavior_hz: float
    control_hz: float
    hardware_hz: float


@dataclass
class LoggingConfig:
    enabled: bool
    perception_jsonl: Optional[str]
    actions_jsonl: Optional[str]


@dataclass
class TelemetryPreviewConfig:
    enabled: bool = False
    jpeg_path: str = "logs/telemetry/preview/latest.jpg"
    meta_path: str = "logs/telemetry/preview/latest.json"
    max_hz: float = 4.0
    max_width: int = 640
    max_height: int = 360
    jpeg_quality: int = 65


@dataclass
class CameraConfig:
    device: str
    width: int
    height: int
    fps: int
    pipeline: Optional[str]


@dataclass
class DeepStreamConfig:
    config_path: Optional[str]
    person_class_id: int
    conf_threshold: Optional[float]


@dataclass
class CosmosConfig:
    enabled: bool = False
    base_url: Optional[str] = None
    provider: str = "auto"
    model: str = "nvidia/cosmos-reason2-2b"
    planner_prompt: str = (
        "Prioritize calm, safe desk-companion behavior. "
        "Always choose one concrete next action."
    )
    policy_identity: str = (
        "You are PALA, a social desk companion lamp that should feel alive, expressive, and safe."
    )
    policy_capabilities: str = (
        "You can move head/neck joints via primitives: hold, breath, glance, nod, orient_to_zone. "
        "You cannot manipulate external objects, move base position, or physically touch users."
    )
    policy_safety: str = (
        "Avoid sudden aggressive motion. Prefer stable conservative actions."
    )
    policy_style: str = (
        "Default style is calm; use curious for gentle tracking and focused for attentive task support."
    )
    # Remote behavior cadence.
    planner_hz: float = 0.5
    summarizer_hz: float = 0.10
    planner_event_delta_threshold: float = 0.65
    planner_event_cooldown_s: float = 0.7
    planner_max_proposals: int = 3
    planner_use_env_context: bool = True
    # Deterministic arbitration.
    arbiter_min_dwell_s: float = 1.2
    arbiter_base_margin: float = 0.10
    idle_after_s: float = 6.0
    idle_glance_after_s: float = 10.0
    startup_wake_enabled: bool = True
    startup_wake_left_s: float = 0.35
    startup_wake_right_s: float = 0.35
    startup_wake_loop_s: float = 0.45
    startup_wake_settle_s: float = 0.70
    startup_wake_rate_rad_s: float = 1.8
    startup_wake_yaw_rad: float = 0.16
    startup_wake_roll_rad: float = 0.10
    startup_wake_pitch2_rad: float = 0.12
    startup_observe_yaw_rad: float = 0.0
    startup_observe_pitch2_rad: float = -0.18
    startup_person_conf_fast_exit: float = 0.60
    mode_min_dwell_s: float = 1.0
    mode_return_home_settle_s: float = 1.2
    mode_recover_settle_s: float = 1.0
    mode_engage_person_conf: float = 0.45
    mode_disengage_person_conf: float = 0.20
    mode_boot_timeout_s: float = 6.7
    stale_penalty_per_s: float = 0.05
    stale_expire_s: float = 14.0
    action_guard_stale_after_s: float = 7.0
    action_guard_orient_cooldown_s: float = 1.2
    action_guard_glance_cooldown_s: float = 3.0
    action_guard_nod_cooldown_s: float = 4.0
    action_guard_home_cooldown_s: float = 4.0
    # Frame sampling and media packaging.
    planner_max_frames: int = 1
    planner_include_latest_frame: bool = True
    summary_window_s: float = 6.0
    summary_max_frames: int = 1
    summary_max_width: int = 320
    summary_jpeg_quality: int = 55
    max_frame_age_ms: int = 500
    request_min_fresh_frames: int = 1
    # Remote request and token budgets.
    request_timeout_ms: int = 20000
    behavior_error_backoff_s: float = 1.5
    behavior_client_error_backoff_s: float = 5.0
    env_max_tokens: int = 1000
    planner_max_tokens: int = 1000
    response_ttl_ms: int = 12000
    # Behavior V3 logs.
    behavior_env_log_path: str = "logs/behavior_env.jsonl"
    behavior_planner_log_path: str = "logs/behavior_planner.jsonl"
    behavior_reasoning_log_path: str = "logs/behavior_reasoning.jsonl"
    behavior_trace_log_path: str = "logs/behavior_trace.jsonl"


@dataclass
class RobotConfig:
    mode: str
    detector: str
    loop_rates: LoopRates
    deadman_timeout_ms: int
    joint_names: List[str]
    joint_limits_rad: List[List[float]]
    servo_calibration: Dict[str, Any]
    logging: LoggingConfig
    camera: CameraConfig
    deepstream: DeepStreamConfig
    telemetry_preview: TelemetryPreviewConfig = field(default_factory=TelemetryPreviewConfig)
    cosmos: CosmosConfig = field(default_factory=CosmosConfig)
    style_profiles: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _fail(path: str, msg: str) -> None:
    raise ValueError(f"Config error at '{path}': {msg}")


def _req(d: Dict[str, Any], key: str, path: str) -> Any:
    if key not in d:
        _fail(path, f"missing required key '{key}'")
    return d[key]


def _as_float(v: Any, path: str) -> float:
    try:
        return float(v)
    except Exception:
        _fail(path, f"expected number, got {type(v).__name__}")


def _as_int(v: Any, path: str) -> int:
    if isinstance(v, bool):
        _fail(path, "expected int, got bool")
    try:
        return int(v)
    except Exception:
        _fail(path, f"expected int, got {type(v).__name__}")


def _as_bool(v: Any, path: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        token = v.strip().lower()
        if token in {"true", "1", "yes", "y", "on"}:
            return True
        if token in {"false", "0", "no", "n", "off", ""}:
            return False
        _fail(path, f"expected bool string, got {v!r}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v in (0, 1):
            return bool(v)
        _fail(path, f"expected bool-compatible number (0/1), got {v!r}")
    _fail(path, f"expected bool, got {type(v).__name__}")


def _as_list(v: Any, path: str) -> List[Any]:
    if not isinstance(v, list):
        _fail(path, f"expected list, got {type(v).__name__}")
    return v


def load_config(path: str) -> RobotConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        _fail("root", "expected mapping at root")

    mode = str(data.get("mode", "dev")).strip().lower()
    allowed_modes = {"dev", "jetson_perception", "jetson_full"}
    if mode not in allowed_modes:
        _fail("mode", "expected one of dev|jetson_perception|jetson_full")

    detector = str(data.get("detector", "dummy")).strip().lower()
    if not detector:
        detector = "dummy"
    allowed_detectors = {"dummy", "deepstream", "jetson"}
    if detector not in allowed_detectors:
        _fail("detector", "expected one of dummy|deepstream|jetson")
    if detector == "jetson":
        _fail("detector", "detector 'jetson' is not implemented; use dummy or deepstream")

    loop_rates_raw = _req(data, "loop_rates", "root")
    if not isinstance(loop_rates_raw, dict):
        _fail("loop_rates", "expected mapping")

    loop_rates = LoopRates(
        perception_hz=_as_float(_req(loop_rates_raw, "perception_hz", "loop_rates"), "loop_rates.perception_hz"),
        behavior_hz=_as_float(_req(loop_rates_raw, "behavior_hz", "loop_rates"), "loop_rates.behavior_hz"),
        control_hz=_as_float(_req(loop_rates_raw, "control_hz", "loop_rates"), "loop_rates.control_hz"),
        hardware_hz=_as_float(_req(loop_rates_raw, "hardware_hz", "loop_rates"), "loop_rates.hardware_hz"),
    )

    deadman_timeout_ms = _as_int(_req(data, "deadman_timeout_ms", "root"), "deadman_timeout_ms")

    joint_names = _as_list(_req(data, "joint_names", "root"), "joint_names")
    if not all(isinstance(n, str) for n in joint_names):
        _fail("joint_names", "expected list of strings")

    joint_limits = _as_list(_req(data, "joint_limits_rad", "root"), "joint_limits_rad")
    if len(joint_limits) != len(joint_names):
        _fail("joint_limits_rad", "length must match joint_names")
    for i, lim in enumerate(joint_limits):
        if not isinstance(lim, list) or len(lim) != 2:
            _fail(f"joint_limits_rad[{i}]", "expected [min, max]")
        _as_float(lim[0], f"joint_limits_rad[{i}][0]")
        _as_float(lim[1], f"joint_limits_rad[{i}][1]")

    servo_cal = data.get("servo_calibration", {})
    if not isinstance(servo_cal, dict):
        _fail("servo_calibration", "expected mapping")

    logging_raw = data.get("logging", {})
    if not isinstance(logging_raw, dict):
        _fail("logging", "expected mapping")
    logging = LoggingConfig(
        enabled=_as_bool(logging_raw.get("enabled", False), "logging.enabled"),
        perception_jsonl=logging_raw.get("perception_jsonl"),
        actions_jsonl=logging_raw.get("actions_jsonl"),
    )

    telemetry_preview_raw = data.get("telemetry_preview", {})
    if not isinstance(telemetry_preview_raw, dict):
        _fail("telemetry_preview", "expected mapping")
    telemetry_preview = TelemetryPreviewConfig(
        enabled=_as_bool(telemetry_preview_raw.get("enabled", False), "telemetry_preview.enabled"),
        jpeg_path=str(telemetry_preview_raw.get("jpeg_path", "logs/telemetry/preview/latest.jpg")),
        meta_path=str(telemetry_preview_raw.get("meta_path", "logs/telemetry/preview/latest.json")),
        max_hz=_as_float(telemetry_preview_raw.get("max_hz", 4.0), "telemetry_preview.max_hz"),
        max_width=_as_int(telemetry_preview_raw.get("max_width", 640), "telemetry_preview.max_width"),
        max_height=_as_int(telemetry_preview_raw.get("max_height", 360), "telemetry_preview.max_height"),
        jpeg_quality=_as_int(telemetry_preview_raw.get("jpeg_quality", 65), "telemetry_preview.jpeg_quality"),
    )

    camera_raw = data.get("camera", {})
    if not isinstance(camera_raw, dict):
        _fail("camera", "expected mapping")
    camera = CameraConfig(
        device=str(camera_raw.get("device", "/dev/video0")),
        width=_as_int(camera_raw.get("width", 640), "camera.width"),
        height=_as_int(camera_raw.get("height", 480), "camera.height"),
        fps=_as_int(camera_raw.get("fps", 30), "camera.fps"),
        pipeline=camera_raw.get("pipeline"),
    )

    ds_raw = data.get("deepstream", {})
    if not isinstance(ds_raw, dict):
        _fail("deepstream", "expected mapping")
    ds_conf_threshold = ds_raw.get("conf_threshold")
    deepstream = DeepStreamConfig(
        config_path=ds_raw.get("config_path"),
        person_class_id=_as_int(ds_raw.get("person_class_id", 0), "deepstream.person_class_id"),
        conf_threshold=None if ds_conf_threshold is None else _as_float(ds_conf_threshold, "deepstream.conf_threshold"),
    )

    cosmos_raw = data.get("cosmos", {})
    if not isinstance(cosmos_raw, dict):
        _fail("cosmos", "expected mapping")
    cosmos = CosmosConfig(
        enabled=_as_bool(cosmos_raw.get("enabled", False), "cosmos.enabled"),
        base_url=None if cosmos_raw.get("base_url") in (None, "") else str(cosmos_raw.get("base_url")),
        provider=str(cosmos_raw.get("provider", "auto")),
        model=str(cosmos_raw.get("model", "nvidia/cosmos-reason2-2b")),
        planner_prompt=str(
            cosmos_raw.get(
                "planner_prompt",
                (
                    "Prioritize calm, safe desk-companion behavior. "
                    "Always choose one concrete next action."
                ),
            )
        ),
        policy_identity=str(
            cosmos_raw.get(
                "policy_identity",
                "You are PALA, a social desk companion lamp that should feel alive, expressive, and safe.",
            )
        ),
        policy_capabilities=str(
            cosmos_raw.get(
                "policy_capabilities",
                (
                    "You can move head/neck joints via primitives: hold, breath, glance, nod, orient_to_zone. "
                    "You cannot manipulate external objects, move base position, or physically touch users."
                ),
            )
        ),
        policy_safety=str(
            cosmos_raw.get(
                "policy_safety",
                "Avoid sudden aggressive motion. Prefer stable conservative actions.",
            )
        ),
        policy_style=str(
            cosmos_raw.get(
                "policy_style",
                "Default style is calm; use curious for gentle tracking and focused for attentive task support.",
            )
        ),
        planner_hz=_as_float(cosmos_raw.get("planner_hz", 0.5), "cosmos.planner_hz"),
        summarizer_hz=_as_float(cosmos_raw.get("summarizer_hz", 0.10), "cosmos.summarizer_hz"),
        planner_event_delta_threshold=_as_float(
            cosmos_raw.get("planner_event_delta_threshold", 0.65),
            "cosmos.planner_event_delta_threshold",
        ),
        planner_event_cooldown_s=_as_float(
            cosmos_raw.get("planner_event_cooldown_s", 0.7),
            "cosmos.planner_event_cooldown_s",
        ),
        planner_max_proposals=_as_int(cosmos_raw.get("planner_max_proposals", 3), "cosmos.planner_max_proposals"),
        planner_use_env_context=_as_bool(
            cosmos_raw.get("planner_use_env_context", True),
            "cosmos.planner_use_env_context",
        ),
        arbiter_min_dwell_s=_as_float(
            cosmos_raw.get("arbiter_min_dwell_s", 1.2),
            "cosmos.arbiter_min_dwell_s",
        ),
        arbiter_base_margin=_as_float(
            cosmos_raw.get("arbiter_base_margin", 0.10),
            "cosmos.arbiter_base_margin",
        ),
        idle_after_s=_as_float(cosmos_raw.get("idle_after_s", 6.0), "cosmos.idle_after_s"),
        idle_glance_after_s=_as_float(
            cosmos_raw.get("idle_glance_after_s", 10.0),
            "cosmos.idle_glance_after_s",
        ),
        startup_wake_enabled=_as_bool(
            cosmos_raw.get("startup_wake_enabled", True),
            "cosmos.startup_wake_enabled",
        ),
        startup_wake_left_s=_as_float(
            cosmos_raw.get("startup_wake_left_s", 0.35),
            "cosmos.startup_wake_left_s",
        ),
        startup_wake_right_s=_as_float(
            cosmos_raw.get("startup_wake_right_s", 0.35),
            "cosmos.startup_wake_right_s",
        ),
        startup_wake_loop_s=_as_float(
            cosmos_raw.get("startup_wake_loop_s", 0.45),
            "cosmos.startup_wake_loop_s",
        ),
        startup_wake_settle_s=_as_float(
            cosmos_raw.get("startup_wake_settle_s", 0.70),
            "cosmos.startup_wake_settle_s",
        ),
        startup_wake_rate_rad_s=_as_float(
            cosmos_raw.get("startup_wake_rate_rad_s", 1.8),
            "cosmos.startup_wake_rate_rad_s",
        ),
        startup_wake_yaw_rad=_as_float(
            cosmos_raw.get("startup_wake_yaw_rad", 0.16),
            "cosmos.startup_wake_yaw_rad",
        ),
        startup_wake_roll_rad=_as_float(
            cosmos_raw.get("startup_wake_roll_rad", 0.10),
            "cosmos.startup_wake_roll_rad",
        ),
        startup_wake_pitch2_rad=_as_float(
            cosmos_raw.get("startup_wake_pitch2_rad", 0.12),
            "cosmos.startup_wake_pitch2_rad",
        ),
        startup_observe_yaw_rad=_as_float(
            cosmos_raw.get("startup_observe_yaw_rad", 0.0),
            "cosmos.startup_observe_yaw_rad",
        ),
        startup_observe_pitch2_rad=_as_float(
            cosmos_raw.get("startup_observe_pitch2_rad", -0.18),
            "cosmos.startup_observe_pitch2_rad",
        ),
        startup_person_conf_fast_exit=_as_float(
            cosmos_raw.get("startup_person_conf_fast_exit", 0.60),
            "cosmos.startup_person_conf_fast_exit",
        ),
        mode_min_dwell_s=_as_float(cosmos_raw.get("mode_min_dwell_s", 1.0), "cosmos.mode_min_dwell_s"),
        mode_return_home_settle_s=_as_float(
            cosmos_raw.get("mode_return_home_settle_s", 1.2),
            "cosmos.mode_return_home_settle_s",
        ),
        mode_recover_settle_s=_as_float(
            cosmos_raw.get("mode_recover_settle_s", 1.0),
            "cosmos.mode_recover_settle_s",
        ),
        mode_engage_person_conf=_as_float(
            cosmos_raw.get("mode_engage_person_conf", 0.45),
            "cosmos.mode_engage_person_conf",
        ),
        mode_disengage_person_conf=_as_float(
            cosmos_raw.get("mode_disengage_person_conf", 0.20),
            "cosmos.mode_disengage_person_conf",
        ),
        mode_boot_timeout_s=_as_float(
            cosmos_raw.get("mode_boot_timeout_s", 6.7),
            "cosmos.mode_boot_timeout_s",
        ),
        stale_penalty_per_s=_as_float(cosmos_raw.get("stale_penalty_per_s", 0.05), "cosmos.stale_penalty_per_s"),
        stale_expire_s=_as_float(cosmos_raw.get("stale_expire_s", 14.0), "cosmos.stale_expire_s"),
        action_guard_stale_after_s=_as_float(
            cosmos_raw.get("action_guard_stale_after_s", cosmos_raw.get("stale_expire_s", 7.0)),
            "cosmos.action_guard_stale_after_s",
        ),
        action_guard_orient_cooldown_s=_as_float(
            cosmos_raw.get("action_guard_orient_cooldown_s", cosmos_raw.get("arbiter_orient_cooldown_s", 1.2)),
            "cosmos.action_guard_orient_cooldown_s",
        ),
        action_guard_glance_cooldown_s=_as_float(
            cosmos_raw.get("action_guard_glance_cooldown_s", cosmos_raw.get("idle_glance_after_s", 3.0)),
            "cosmos.action_guard_glance_cooldown_s",
        ),
        action_guard_nod_cooldown_s=_as_float(
            cosmos_raw.get("action_guard_nod_cooldown_s", 4.0),
            "cosmos.action_guard_nod_cooldown_s",
        ),
        action_guard_home_cooldown_s=_as_float(
            cosmos_raw.get("action_guard_home_cooldown_s", 4.0),
            "cosmos.action_guard_home_cooldown_s",
        ),
        planner_max_frames=_as_int(cosmos_raw.get("planner_max_frames", 1), "cosmos.planner_max_frames"),
        planner_include_latest_frame=_as_bool(
            cosmos_raw.get("planner_include_latest_frame", True),
            "cosmos.planner_include_latest_frame",
        ),
        summary_window_s=_as_float(
            cosmos_raw.get("summary_window_s", 6.0),
            "cosmos.summary_window_s",
        ),
        summary_max_frames=_as_int(
            cosmos_raw.get("summary_max_frames", 1),
            "cosmos.summary_max_frames",
        ),
        summary_max_width=_as_int(
            cosmos_raw.get("summary_max_width", 320),
            "cosmos.summary_max_width",
        ),
        summary_jpeg_quality=_as_int(
            cosmos_raw.get("summary_jpeg_quality", 55),
            "cosmos.summary_jpeg_quality",
        ),
        max_frame_age_ms=_as_int(cosmos_raw.get("max_frame_age_ms", 500), "cosmos.max_frame_age_ms"),
        request_min_fresh_frames=_as_int(
            cosmos_raw.get("request_min_fresh_frames", 1),
            "cosmos.request_min_fresh_frames",
        ),
        request_timeout_ms=_as_int(cosmos_raw.get("request_timeout_ms", 20000), "cosmos.request_timeout_ms"),
        behavior_error_backoff_s=_as_float(
            cosmos_raw.get("behavior_error_backoff_s", 1.5),
            "cosmos.behavior_error_backoff_s",
        ),
        behavior_client_error_backoff_s=_as_float(
            cosmos_raw.get("behavior_client_error_backoff_s", 5.0),
            "cosmos.behavior_client_error_backoff_s",
        ),
        env_max_tokens=_as_int(cosmos_raw.get("env_max_tokens", 1000), "cosmos.env_max_tokens"),
        planner_max_tokens=_as_int(cosmos_raw.get("planner_max_tokens", 1000), "cosmos.planner_max_tokens"),
        response_ttl_ms=_as_int(cosmos_raw.get("response_ttl_ms", 12000), "cosmos.response_ttl_ms"),
        behavior_env_log_path=str(cosmos_raw.get("behavior_env_log_path", "logs/behavior_env.jsonl")),
        behavior_planner_log_path=str(
            cosmos_raw.get("behavior_planner_log_path", "logs/behavior_planner.jsonl")
        ),
        behavior_reasoning_log_path=str(
            cosmos_raw.get("behavior_reasoning_log_path", "logs/behavior_reasoning.jsonl")
        ),
        behavior_trace_log_path=str(cosmos_raw.get("behavior_trace_log_path", "logs/behavior_trace.jsonl")),
    )

    style_profiles_raw = data.get("styles", {})
    if not isinstance(style_profiles_raw, dict):
        _fail("styles", "expected mapping")
    style_profiles = default_style_profiles()
    for name, raw in style_profiles_raw.items():
        if not isinstance(raw, dict):
            _fail(f"styles.{name}", "expected mapping")
        key = str(name).strip().lower()
        if not key:
            continue
        profile = dict(style_profiles.get(key, {}))
        for param in ("amp_scale", "rate_scale", "duration_scale", "settle_scale"):
            if param in raw:
                profile[param] = _as_float(raw[param], f"styles.{key}.{param}")
        style_profiles[key] = profile

    return RobotConfig(
        mode=mode,
        detector=detector,
        loop_rates=loop_rates,
        deadman_timeout_ms=deadman_timeout_ms,
        joint_names=joint_names,
        joint_limits_rad=joint_limits,
        servo_calibration=servo_cal,
        logging=logging,
        telemetry_preview=telemetry_preview,
        camera=camera,
        deepstream=deepstream,
        cosmos=cosmos,
        style_profiles=style_profiles,
    )
