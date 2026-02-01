from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, List
import json


@dataclass
class BBoxNorm:
    """Normalized bbox in cx, cy, w, h (0..1)."""
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class PointNorm:
    """Normalized point in x, y (0..1)."""
    x: float
    y: float


@dataclass
class PerceptionState:
    timestamp_monotonic_s: float
    timestamp_wall_s: Optional[float] = None
    fps: Optional[float] = None
    latency_ms: Optional[float] = None
    primary_person: Optional[BBoxNorm] = None
    primary_person_conf: Optional[float] = None
    pointing_target: Optional[PointNorm] = None
    pointing_conf: Optional[float] = None
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionPlan:
    primitive: str
    params: Dict[str, Any]
    confidence: float
    explanation: Optional[str] = None


@dataclass
class HardwareCommand:
    timestamp_monotonic_s: float
    joint_angles_rad: List[float]
    enable: bool = True


def to_json_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass (or nested dataclasses) to a JSON-serializable dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_json_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_dict(v) for v in obj]
    return obj


def to_json_line(obj: Any) -> str:
    return json.dumps(to_json_dict(obj), separators=(",", ":"), ensure_ascii=True)
