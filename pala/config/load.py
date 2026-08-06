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
class CosmosConfig:
    enabled: bool = False
    base_url: Optional[str] = None
    provider: str = "auto"
    model: str = "nvidia/cosmos-reason2-2b"
    planner_prompt: str = (
        "Prioritize calm, safe desk-companion behavior. "
        "Always choose one concrete next action."
    )
    request_timeout_ms: int = 20000


@dataclass
class RobotConfig:
    mode: str
    loop_rates: LoopRates
    deadman_timeout_ms: int
    joint_names: List[str]
    joint_limits_rad: List[List[float]]
    servo_calibration: Dict[str, Any]
    logging: LoggingConfig
    camera: CameraConfig
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

    cosmos_raw = data.get("cosmos", {})
    if not isinstance(cosmos_raw, dict):
        _fail("cosmos", "expected mapping")
    cosmos = CosmosConfig(
        enabled=_as_bool(cosmos_raw.get("enabled", False), "cosmos.enabled"),
        base_url=None if cosmos_raw.get("base_url") in (None, "") else str(cosmos_raw.get("base_url")),
        provider=str(cosmos_raw.get("provider", "auto")),
        model=str(cosmos_raw.get("model", "nvidia/cosmos-reason2-2b")),
        planner_prompt=str(cosmos_raw.get("planner_prompt", "Describe the scene and choose one next behavior.")),
        request_timeout_ms=_as_int(cosmos_raw.get("request_timeout_ms", 20000), "cosmos.request_timeout_ms"),
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
        loop_rates=loop_rates,
        deadman_timeout_ms=deadman_timeout_ms,
        joint_names=joint_names,
        joint_limits_rad=joint_limits,
        servo_calibration=servo_cal,
        logging=logging,
        telemetry_preview=telemetry_preview,
        camera=camera,
        cosmos=cosmos,
        style_profiles=style_profiles,
    )
