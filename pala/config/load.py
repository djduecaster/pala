from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import yaml


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
class CameraConfig:
    device: str
    width: int
    height: int
    fps: int
    pipeline: Optional[str]


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

    mode = str(data.get("mode", "dev"))

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
        enabled=bool(logging_raw.get("enabled", False)),
        perception_jsonl=logging_raw.get("perception_jsonl"),
        actions_jsonl=logging_raw.get("actions_jsonl"),
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

    return RobotConfig(
        mode=mode,
        loop_rates=loop_rates,
        deadman_timeout_ms=deadman_timeout_ms,
        joint_names=joint_names,
        joint_limits_rad=joint_limits,
        servo_calibration=servo_cal,
        logging=logging,
        camera=camera,
    )
