from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Union
import json
from uuid import uuid4


@dataclass
class PerceptionState:
    """Capture health and timing for the latest camera frame."""

    timestamp_monotonic_s: float
    timestamp_wall_s: Optional[float] = None
    frame_id: Optional[int] = None
    fps: Optional[float] = None
    latency_ms: Optional[float] = None
    frame_age_ms: Optional[float] = None
    source_alive: bool = True
    is_new_frame: bool = False
    debug: Dict[str, Any] = field(default_factory=dict)


class PrimitiveKind(str, Enum):
    HOLD = "hold"
    HOME = "home"
    MOVE_TO = "move_to"
    GAZE_TO = "gaze_to"
    GLANCE = "glance"
    NOD = "nod"
    BREATH = "breath"
    ORIENT_TO_ZONE = "orient_to_zone"
    SCAN_SWEEP = "scan_sweep"


@dataclass(frozen=True)
class HoldCommand:
    pass


@dataclass(frozen=True)
class HomeCommand:
    rate_rad_s: float = 1.5


@dataclass(frozen=True)
class MoveToCommand:
    target_rad: List[float]
    relative: bool = False
    rate_rad_s: float = 1.5
    timeout_s: float = 2.0


@dataclass(frozen=True)
class GazeToCommand:
    yaw_rad: float
    pitch_rad: float
    rate_rad_s: float = 1.5
    dwell_s: float = 0.0
    timeout_s: float = 1.5


@dataclass(frozen=True)
class GlanceCommand:
    direction: str = "left"
    amp_rad: float = 0.35
    duration_s: float = 0.6
    rate_rad_s: float = 1.8


@dataclass(frozen=True)
class NodCommand:
    amp_rad: float = 0.2
    duration_s: float = 0.4
    cycles: int = 1
    rate_rad_s: float = 1.8


@dataclass(frozen=True)
class BreathCommand:
    amp_rad: float = 0.08
    period_s: float = 7.0
    rate_rad_s: float = 1.0


@dataclass(frozen=True)
class OrientToZoneCommand:
    zone: str = "center"
    amp_rad: float = 0.25
    rate_rad_s: float = 1.4


@dataclass(frozen=True)
class ScanSweepCommand:
    positions: int = 0
    camera_hfov_deg: float = 70.42
    overlap: float = 0.2
    dwell_s: float = 0.2
    rate_rad_s: float = 1.4
    edge_margin_rad: float = 0.05
    return_to_center: bool = True
    timeout_s: float = 8.0


PrimitiveCommand = Union[
    HoldCommand,
    HomeCommand,
    MoveToCommand,
    GazeToCommand,
    GlanceCommand,
    NodCommand,
    BreathCommand,
    OrientToZoneCommand,
    ScanSweepCommand,
]


@dataclass
class ActionPlan:
    primitive: PrimitiveKind
    command: PrimitiveCommand
    confidence: float
    explanation: Optional[str] = None
    style: str = "calm"
    action_id: str = field(default_factory=lambda: uuid4().hex)
    cancel_current: bool = False

    def __post_init__(self) -> None:
        self.primitive = _coerce_primitive_kind(self.primitive)
        self.command = _coerce_command(self.primitive, self.command)
        self.confidence = _clamp01(self.confidence)
        self.style = _coerce_style(self.style)
        self.action_id = _coerce_action_id(self.action_id)
        self.cancel_current = _coerce_bool(self.cancel_current, default=False)


@dataclass
class HardwareCommand:
    timestamp_monotonic_s: float
    joint_angles_rad: List[float]
    enable: bool = True


def action_plan_from_dict(data: Mapping[str, Any]) -> Optional[ActionPlan]:
    payload = data
    wrapped = payload.get("action")
    if isinstance(wrapped, Mapping):
        payload = wrapped

    primitive_raw = payload.get("primitive")
    if primitive_raw is None:
        return None
    primitive = _parse_primitive_kind(primitive_raw)
    if primitive is None:
        return None

    command_raw = payload.get("command")
    if not isinstance(command_raw, Mapping):
        return None
    try:
        command = command_from_dict(primitive, command_raw)
    except ValueError:
        return None

    try:
        confidence = _clamp01(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    explanation_raw = payload.get("explanation")
    explanation = explanation_raw if isinstance(explanation_raw, str) else None
    style = _coerce_style(payload.get("style", "calm"))
    action_id_raw = payload.get("action_id")
    action_id = _coerce_action_id(action_id_raw)
    cancel_current = _coerce_bool(payload.get("cancel_current", False), default=False)
    return ActionPlan(
        primitive=primitive,
        command=command,
        confidence=confidence,
        explanation=explanation,
        style=style,
        action_id=action_id,
        cancel_current=cancel_current,
    )


def command_from_dict(kind: PrimitiveKind, payload: Mapping[str, Any]) -> PrimitiveCommand:
    if kind == PrimitiveKind.HOLD:
        return HoldCommand()
    if kind == PrimitiveKind.HOME:
        return HomeCommand(rate_rad_s=_as_float(payload.get("rate_rad_s", 1.5), "rate_rad_s"))
    if kind == PrimitiveKind.MOVE_TO:
        target_raw = payload.get("target_rad")
        if not isinstance(target_raw, list):
            raise ValueError("target_rad is required for move_to")
        target = [_as_float(v, "target_rad") for v in target_raw]
        if len(target) == 0:
            raise ValueError("target_rad must not be empty")
        return MoveToCommand(
            target_rad=target,
            relative=_coerce_bool(payload.get("relative", False), default=False),
            rate_rad_s=_as_float(payload.get("rate_rad_s", 1.5), "rate_rad_s"),
            timeout_s=_as_float(payload.get("timeout_s", 2.0), "timeout_s"),
        )
    if kind == PrimitiveKind.GAZE_TO:
        return GazeToCommand(
            yaw_rad=_as_float(payload.get("yaw_rad"), "yaw_rad"),
            pitch_rad=_as_float(payload.get("pitch_rad"), "pitch_rad"),
            rate_rad_s=_as_float(payload.get("rate_rad_s", 1.5), "rate_rad_s"),
            dwell_s=_as_float(payload.get("dwell_s", 0.0), "dwell_s"),
            timeout_s=_as_float(payload.get("timeout_s", 1.5), "timeout_s"),
        )
    if kind == PrimitiveKind.GLANCE:
        direction = str(payload.get("direction", "left")).strip().lower()
        if direction not in {"left", "right", "up", "down"}:
            raise ValueError("direction must be left|right|up|down")
        return GlanceCommand(
            direction=direction,
            amp_rad=_as_float(payload.get("amp_rad", 0.35), "amp_rad"),
            duration_s=_as_float(payload.get("duration_s", 0.6), "duration_s"),
            rate_rad_s=_as_float(payload.get("rate_rad_s", 1.8), "rate_rad_s"),
        )
    if kind == PrimitiveKind.NOD:
        cycles = _as_int(payload.get("cycles", 1), "cycles")
        if cycles < 1:
            raise ValueError("cycles must be >= 1")
        return NodCommand(
            amp_rad=_as_float(payload.get("amp_rad", 0.2), "amp_rad"),
            duration_s=_as_float(payload.get("duration_s", 0.4), "duration_s"),
            cycles=cycles,
            rate_rad_s=_as_float(payload.get("rate_rad_s", 1.8), "rate_rad_s"),
        )
    if kind == PrimitiveKind.BREATH:
        return BreathCommand(
            amp_rad=_as_float(payload.get("amp_rad", 0.08), "amp_rad"),
            period_s=_as_float(payload.get("period_s", 7.0), "period_s"),
            rate_rad_s=_as_float(payload.get("rate_rad_s", 1.0), "rate_rad_s"),
        )
    if kind == PrimitiveKind.ORIENT_TO_ZONE:
        zone = str(payload.get("zone", "center")).strip().lower()
        if zone not in {"left", "center", "right"}:
            raise ValueError("zone must be left|center|right")
        return OrientToZoneCommand(
            zone=zone,
            amp_rad=_as_float(payload.get("amp_rad", 0.25), "amp_rad"),
            rate_rad_s=_as_float(payload.get("rate_rad_s", 1.4), "rate_rad_s"),
        )
    if kind == PrimitiveKind.SCAN_SWEEP:
        positions = _as_int(payload.get("positions", 0), "positions")
        if positions < 0:
            raise ValueError("positions must be >= 0")
        camera_hfov_deg = _as_float(payload.get("camera_hfov_deg", 70.42), "camera_hfov_deg")
        if camera_hfov_deg <= 0.0:
            raise ValueError("camera_hfov_deg must be > 0")
        overlap = _as_float(payload.get("overlap", 0.2), "overlap")
        if overlap < 0.0 or overlap >= 0.95:
            raise ValueError("overlap must be in [0.0, 0.95)")
        dwell_s = _as_float(payload.get("dwell_s", 0.2), "dwell_s")
        if dwell_s < 0.0:
            raise ValueError("dwell_s must be >= 0")
        rate_rad_s = _as_float(payload.get("rate_rad_s", 1.4), "rate_rad_s")
        if rate_rad_s <= 0.0:
            raise ValueError("rate_rad_s must be > 0")
        edge_margin_rad = _as_float(payload.get("edge_margin_rad", 0.05), "edge_margin_rad")
        if edge_margin_rad < 0.0:
            raise ValueError("edge_margin_rad must be >= 0")
        timeout_s = _as_float(payload.get("timeout_s", 8.0), "timeout_s")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be > 0")
        return ScanSweepCommand(
            positions=positions,
            camera_hfov_deg=camera_hfov_deg,
            overlap=overlap,
            dwell_s=dwell_s,
            rate_rad_s=rate_rad_s,
            edge_margin_rad=edge_margin_rad,
            return_to_center=_coerce_bool(payload.get("return_to_center", True), default=True),
            timeout_s=timeout_s,
        )
    raise ValueError(f"Unsupported primitive kind: {kind.value}")


def to_json_dict(obj: Any) -> Any:
    """Convert a dataclass (or nested dataclasses) to a JSON-serializable dict."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {f.name: to_json_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: to_json_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_dict(v) for v in obj]
    return obj


def to_json_line(obj: Any) -> str:
    return json.dumps(to_json_dict(obj), separators=(",", ":"), ensure_ascii=True)


def _coerce_primitive_kind(value: Any) -> PrimitiveKind:
    if isinstance(value, PrimitiveKind):
        return value
    parsed = _parse_primitive_kind(value)
    if parsed is None:
        raise ValueError(f"Unknown primitive kind: {value!r}")
    return parsed


def _parse_primitive_kind(value: Any) -> Optional[PrimitiveKind]:
    if isinstance(value, PrimitiveKind):
        return value
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token:
        return None
    try:
        return PrimitiveKind(token.lower())
    except ValueError:
        try:
            return PrimitiveKind[token.upper()]
        except KeyError:
            return None


def _coerce_command(kind: PrimitiveKind, command: Any) -> PrimitiveCommand:
    command_cls = _COMMAND_BY_KIND[kind]
    if isinstance(command, command_cls):
        return command
    if isinstance(command, Mapping):
        return command_from_dict(kind, command)
    raise ValueError(f"Command for {kind.value} must be {command_cls.__name__}")


def _as_float(value: Any, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} is required")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def _as_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{field_name} is required")
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an int")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an int") from exc


def _clamp01(value: Any) -> float:
    v = float(value)
    return max(0.0, min(1.0, v))


def _coerce_action_id(value: Any) -> str:
    if value is None:
        return uuid4().hex
    token = str(value).strip()
    return token if token else uuid4().hex


def _coerce_style(value: Any) -> str:
    if value is None:
        return "calm"
    token = str(value).strip().lower()
    return token if token else "calm"


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y", "on"}:
            return True
        if token in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


_COMMAND_BY_KIND = {
    PrimitiveKind.HOLD: HoldCommand,
    PrimitiveKind.HOME: HomeCommand,
    PrimitiveKind.MOVE_TO: MoveToCommand,
    PrimitiveKind.GAZE_TO: GazeToCommand,
    PrimitiveKind.GLANCE: GlanceCommand,
    PrimitiveKind.NOD: NodCommand,
    PrimitiveKind.BREATH: BreathCommand,
    PrimitiveKind.ORIENT_TO_ZONE: OrientToZoneCommand,
    PrimitiveKind.SCAN_SWEEP: ScanSweepCommand,
}
