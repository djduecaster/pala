#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import http.server
from itertools import product
import json
import math
from pathlib import Path
import re
import socketserver
import sys
import threading
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, quote, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pala.config import load_config
from pala.types import (
    ActionPlan,
    PrimitiveKind,
    to_json_dict,
)
from tools.primitive_sim.simulate import (
    CONTINUOUS_PRIMITIVES,
    SimSegment,
    build_suite_segments,
    load_segments_from_json,
    simulate_segments,
    write_trace_json,
)


class _ReuseTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


class _ThreadingReuseTCPServer(socketserver.ThreadingMixIn, _ReuseTCPServer):
    daemon_threads = True


_PRIMITIVE_ORDER: tuple[PrimitiveKind, ...] = (
    PrimitiveKind.HOLD,
    PrimitiveKind.HOME,
    PrimitiveKind.MOVE_TO,
    PrimitiveKind.GAZE_TO,
    PrimitiveKind.GLANCE,
    PrimitiveKind.NOD,
    PrimitiveKind.BREATH,
    PrimitiveKind.ORIENT_TO_ZONE,
)

_DEFAULT_BASELINE_REL = Path("tools/primitive_sim/baseline_params.json")
_DEFAULT_SUITE_TRACE_REL = Path("logs/primitive_sim/latest_trace.json")
_DEFAULT_SCENARIO_TRACE_REL = Path("logs/primitive_sim/scenario_latest.json")
_DEFAULT_EXPERIMENTS_REL = Path("logs/primitive_sim/experiments.jsonl")
_DEFAULT_SWEEP_DIR_REL = Path("logs/primitive_sim/sweeps")
_BASELINE_VERSION = 2
_BASELINE_UPDATED_BY = "primitive_studio"
_MAX_SWEEP_CANDIDATES = 120
_DEFAULT_SWEEP_TOP_K = 10
_DEFAULT_SWEEP_WEIGHTS: dict[str, float] = {
    "min_limit_margin_rad": 4.0,
    "limit_violation_count": -12.0,
    "peak_joint_vel_rad_s": -0.35,
    "mean_abs_joint_vel_rad_s": -0.25,
    "path_length_rad": -0.04,
    "primitive_switch_count": -0.15,
    "duration_s": -0.02,
}

_GEOM_PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "baseRadius": ("baseRadius", "base_radius", "base_radius_m"),
    "baseThickness": ("baseThickness", "base_thickness", "base_thickness_m"),
    "mastHeight": ("mastHeight", "mast_height", "mast_height_m"),
    "hubRise": ("hubRise", "hub_rise", "hub_rise_m"),
    "upperArmLen": ("upperArmLen", "upper_arm_len", "upper_arm_len_m", "link1_len", "link1_len_m"),
    "foreArmLen": ("foreArmLen", "fore_arm_len", "fore_arm_len_m", "link2_len", "link2_len_m"),
    "wristStubLen": ("wristStubLen", "wrist_stub_len", "wrist_stub_len_m"),
    "shadeNeckLen": ("shadeNeckLen", "shade_neck_len", "shade_neck_len_m"),
    "shadeLen": ("shadeLen", "shade_len", "shade_len_m"),
    "shadeRearRadius": (
        "shadeRearRadius",
        "shade_rear_radius",
        "shade_rear_radius_m",
        "shade_base_radius",
        "shade_base_radius_m",
    ),
    "shadeFrontRadius": (
        "shadeFrontRadius",
        "shade_front_radius",
        "shade_front_radius_m",
        "shade_tip_radius",
        "shade_tip_radius_m",
    ),
    "pitch1ZeroOffsetRad": (
        "pitch1ZeroOffsetRad",
        "pitch1_zero_offset_rad",
        "pitch1_offset_rad",
        "theta1_offset_rad",
    ),
    "pitch2ZeroOffsetRad": (
        "pitch2ZeroOffsetRad",
        "pitch2_zero_offset_rad",
        "pitch2_offset_rad",
        "theta2_offset_rad",
    ),
    "pitch3ZeroOffsetRad": (
        "pitch3ZeroOffsetRad",
        "pitch3_zero_offset_rad",
        "pitch3_offset_rad",
        "theta3_offset_rad",
    ),
}

_GEOM_POSITIVE_KEYS: set[str] = {
    "baseRadius",
    "baseThickness",
    "mastHeight",
    "hubRise",
    "upperArmLen",
    "foreArmLen",
    "wristStubLen",
    "shadeNeckLen",
    "shadeLen",
    "shadeRearRadius",
    "shadeFrontRadius",
}


@dataclass
class _StudioContext:
    cfg: Any
    config_path: Path
    hz: float
    baseline_path: Path
    baseline: dict[str, Any]
    primitive_specs: list[dict[str, Any]]
    style_options: list[str]
    lamp_geometry: dict[str, float]
    dh_params: dict[str, float]
    fsm_sim: Any
    experiments_path: Path
    lock: threading.Lock = field(default_factory=threading.Lock)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local primitive simulation and optional 3D web viewer")
    parser.add_argument("--config", default="config/robot.yaml", help="Path to robot config")
    parser.add_argument(
        "--scenario",
        choices=["suite", "single", "script", "studio", "joint_checker", "state_machine", "scenario_lab"],
        default="suite",
    )
    parser.add_argument("--script", default=None, help="Script JSON path for --scenario script")
    parser.add_argument("--style", default="calm", help="Action style label (calm/curious/focused)")

    parser.add_argument("--primitive", default="breath", help="Primitive for --scenario single")
    parser.add_argument("--duration-s", type=float, default=3.0, help="Segment duration for single")
    parser.add_argument("--rate-rad-s", type=float, default=1.2)
    parser.add_argument("--amp-rad", type=float, default=0.1)
    parser.add_argument("--period-s", type=float, default=6.5)
    parser.add_argument("--direction", choices=["left", "right", "up", "down"], default="left")
    parser.add_argument("--zone", choices=["left", "center", "right"], default="center")
    parser.add_argument("--yaw-rad", type=float, default=0.0)
    parser.add_argument("--pitch-rad", type=float, default=0.0)
    parser.add_argument("--target-rad", default=None, help="Comma-separated move_to target list")
    parser.add_argument("--relative", action="store_true", help="Use relative target for move_to")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--dwell-s", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=1.6)
    parser.add_argument("--suite-breath-s", type=float, default=6.0)

    parser.add_argument("--hz", type=float, default=0.0, help="Sim update rate (0 uses control_hz from config)")
    parser.add_argument("--output", default="logs/primitive_sim/latest_trace.json", help="Output JSON trace path")
    parser.add_argument("--serve", action="store_true", help="Serve web viewer and block")
    parser.add_argument("--port", type=int, default=8766, help="Viewer HTTP port")
    parser.add_argument("--baseline", default=str(_DEFAULT_BASELINE_REL), help="Baseline params JSON path")
    parser.add_argument("--experiments", default=str(_DEFAULT_EXPERIMENTS_REL), help="Scenario experiments JSONL path")
    return parser.parse_args(argv)


def _abs_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def _parse_primitive(value: Any) -> PrimitiveKind:
    token = str(value).strip().lower()
    return PrimitiveKind(token)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on", "y"}:
            return True
        if token in {"0", "false", "no", "off", "n", ""}:
            return False
    return bool(value)


def _safe_slug(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return token.strip("._-") or "run"


def _default_sweep_weights() -> dict[str, float]:
    return dict(_DEFAULT_SWEEP_WEIGHTS)


def _normalize_sweep_weights(raw: Any) -> dict[str, float]:
    out = _default_sweep_weights()
    if not isinstance(raw, Mapping):
        return out
    for key, default in _DEFAULT_SWEEP_WEIGHTS.items():
        val = raw.get(key)
        if val is None:
            continue
        out[key] = _coerce_float(val, default)
    return out


def _score_trace_metrics(metrics: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        score += float(weight) * _coerce_float(metrics.get(key), 0.0)
    return float(score)


def _expand_sweep_grid(raw: Any, *, max_candidates: int = _MAX_SWEEP_CANDIDATES) -> list[dict[str, float]]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("param_grid must be a non-empty object of numeric arrays")

    keys: list[str] = []
    values: list[list[float]] = []
    for key, grid_values in raw.items():
        field = str(key).strip()
        if not field:
            continue
        if not isinstance(grid_values, list) or not grid_values:
            raise ValueError(f"param_grid[{field}] must be a non-empty array")
        numeric_vals: list[float] = []
        for value_idx, item in enumerate(grid_values):
            try:
                num = float(item)
            except Exception as exc:
                raise ValueError(
                    f"param_grid[{field}][{value_idx}] must be numeric"
                ) from exc
            if not math.isfinite(num):
                raise ValueError(f"param_grid[{field}][{value_idx}] must be finite")
            numeric_vals.append(num)
        keys.append(field)
        values.append(numeric_vals)

    if not keys:
        raise ValueError("param_grid must include at least one named parameter")

    combos: list[dict[str, float]] = []
    for row in product(*values):
        combos.append({keys[i]: float(row[i]) for i in range(len(keys))})
        if len(combos) > max_candidates:
            raise ValueError(
                f"param_grid expands to more than {max_candidates} candidates; reduce value counts"
            )
    return combos


def _target_step_index(raw_steps: Sequence[Any], token: Any) -> int:
    if not raw_steps:
        raise ValueError("steps must be non-empty")
    if token is None:
        return max(0, len(raw_steps) - 1)
    if isinstance(token, int):
        idx = token
    elif isinstance(token, str):
        t = token.strip()
        if not t:
            return max(0, len(raw_steps) - 1)
        if t.isdigit() or (t.startswith("-") and t[1:].isdigit()):
            idx = int(t)
        else:
            lowered = t.lower()
            for i, row in enumerate(raw_steps):
                if isinstance(row, Mapping):
                    name = str(row.get("name", "")).strip().lower()
                    if name and name == lowered:
                        return i
            raise ValueError(f"target_step not found by name: {t}")
    else:
        idx = _coerce_int(token, len(raw_steps) - 1)

    if idx < 0:
        idx = len(raw_steps) + idx
    if idx < 0 or idx >= len(raw_steps):
        raise ValueError(f"target_step index out of range: {idx}")
    return idx


def _default_move_target(cfg: Any) -> list[float]:
    target = [0.0 for _ in cfg.joint_names]
    for i, lim in enumerate(cfg.joint_limits_rad):
        lo, hi = float(lim[0]), float(lim[1])
        span = hi - lo
        if span <= 1e-6:
            target[i] = lo
            continue
        frac = 0.12 if i % 2 == 0 else -0.12
        target[i] = max(lo, min(hi, frac * span))
    return target


def _default_command_payload(primitive: PrimitiveKind, cfg: Any) -> dict[str, Any]:
    if primitive == PrimitiveKind.HOLD:
        return {}
    if primitive == PrimitiveKind.HOME:
        return {"rate_rad_s": 1.2}
    if primitive == PrimitiveKind.MOVE_TO:
        return {
            "target_rad": _default_move_target(cfg),
            "relative": False,
            "rate_rad_s": 1.2,
            "timeout_s": 2.0,
        }
    if primitive == PrimitiveKind.GAZE_TO:
        return {
            "yaw_rad": 0.0,
            "pitch_rad": 0.0,
            "rate_rad_s": 1.4,
            "dwell_s": 0.2,
            "timeout_s": 1.6,
        }
    if primitive == PrimitiveKind.GLANCE:
        return {
            "direction": "left",
            "amp_rad": 0.26,
            "duration_s": 0.7,
            "rate_rad_s": 1.6,
        }
    if primitive == PrimitiveKind.NOD:
        return {
            "amp_rad": 0.2,
            "duration_s": 0.9,
            "cycles": 1,
            "rate_rad_s": 1.8,
        }
    if primitive == PrimitiveKind.BREATH:
        return {
            "amp_rad": 0.08,
            "period_s": 6.5,
            "rate_rad_s": 1.0,
        }
    if primitive == PrimitiveKind.ORIENT_TO_ZONE:
        return {
            "zone": "center",
            "amp_rad": 0.22,
            "rate_rad_s": 1.3,
        }
    raise ValueError(f"unsupported primitive: {primitive.value}")


def _canonicalize_command_payload(
    primitive: PrimitiveKind,
    cfg: Any,
    *,
    style: str = "calm",
    base: Mapping[str, Any] | None = None,
    patch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = _default_command_payload(primitive, cfg)
    if isinstance(base, Mapping):
        merged.update(dict(base))
    if isinstance(patch, Mapping):
        merged.update(dict(patch))

    if primitive == PrimitiveKind.MOVE_TO:
        raw_target = merged.get("target_rad")
        if not isinstance(raw_target, list) or len(raw_target) != len(cfg.joint_names):
            merged["target_rad"] = _default_move_target(cfg)
        else:
            merged["target_rad"] = [float(v) for v in raw_target]
        merged["relative"] = _coerce_bool(merged.get("relative", False), False)

    if primitive == PrimitiveKind.NOD:
        merged["cycles"] = max(1, _coerce_int(merged.get("cycles", 1), 1))

    action = ActionPlan(
        primitive=primitive,
        command=merged,
        confidence=1.0,
        explanation="primitive_sim",
        style=str(style),
    )
    return dict(to_json_dict(action.command))


def _build_single_command_payload(args: argparse.Namespace, cfg: Any, primitive: PrimitiveKind) -> dict[str, Any]:
    payload: dict[str, Any]
    if primitive == PrimitiveKind.HOLD:
        payload = {}
    elif primitive == PrimitiveKind.HOME:
        payload = {"rate_rad_s": float(args.rate_rad_s)}
    elif primitive == PrimitiveKind.BREATH:
        payload = {
            "amp_rad": float(args.amp_rad),
            "period_s": float(args.period_s),
            "rate_rad_s": float(args.rate_rad_s),
        }
    elif primitive == PrimitiveKind.GLANCE:
        payload = {
            "direction": str(args.direction),
            "amp_rad": float(args.amp_rad),
            "duration_s": float(args.duration_s),
            "rate_rad_s": float(args.rate_rad_s),
        }
    elif primitive == PrimitiveKind.NOD:
        payload = {
            "amp_rad": float(args.amp_rad),
            "duration_s": float(args.duration_s),
            "cycles": max(1, int(args.cycles)),
            "rate_rad_s": float(args.rate_rad_s),
        }
    elif primitive == PrimitiveKind.ORIENT_TO_ZONE:
        payload = {
            "zone": str(args.zone),
            "amp_rad": float(args.amp_rad),
            "rate_rad_s": float(args.rate_rad_s),
        }
    elif primitive == PrimitiveKind.GAZE_TO:
        payload = {
            "yaw_rad": float(args.yaw_rad),
            "pitch_rad": float(args.pitch_rad),
            "rate_rad_s": float(args.rate_rad_s),
            "dwell_s": float(args.dwell_s),
            "timeout_s": float(args.timeout_s),
        }
    elif primitive == PrimitiveKind.MOVE_TO:
        payload = {
            "target_rad": _parse_move_target(args, cfg),
            "relative": bool(args.relative),
            "rate_rad_s": float(args.rate_rad_s),
            "timeout_s": float(args.timeout_s),
        }
    else:
        raise ValueError(f"unsupported primitive: {primitive.value}")
    return _canonicalize_command_payload(primitive, cfg, style=str(args.style), patch=payload)


def _build_single_action(args: argparse.Namespace, cfg: Any) -> ActionPlan:
    primitive = _parse_primitive(args.primitive)
    command_payload = _build_single_command_payload(args, cfg, primitive)
    return ActionPlan(
        primitive=primitive,
        command=command_payload,
        confidence=1.0,
        explanation="primitive_sim",
        style=str(args.style),
    )


def _parse_move_target(args: argparse.Namespace, cfg: Any) -> list[float]:
    if args.target_rad:
        parts = [p.strip() for p in str(args.target_rad).split(",") if p.strip()]
        values = [float(p) for p in parts]
        if len(values) != len(cfg.joint_names):
            raise ValueError(f"--target-rad expects {len(cfg.joint_names)} values")
        return values
    return _default_move_target(cfg)


def _build_single_segment(args: argparse.Namespace, cfg: Any) -> list[SimSegment]:
    action = _build_single_action(args, cfg)
    stop_on_done = action.primitive not in CONTINUOUS_PRIMITIVES
    max_s = max(0.1, float(args.duration_s))
    if stop_on_done:
        max_s *= 1.25
    return [
        SimSegment(
            name=f"single_{action.primitive.value}",
            action=action,
            max_s=max_s,
            stop_on_done=stop_on_done,
        )
    ]


def _utc_now_iso() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


def _with_baseline_metadata(
    baseline: Mapping[str, Any],
    *,
    updated_by: str = _BASELINE_UPDATED_BY,
    updated_at_utc: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = dict(baseline)
    out["version"] = _BASELINE_VERSION
    out["updated_by"] = str(updated_by or _BASELINE_UPDATED_BY)
    out["updated_at_utc"] = str(updated_at_utc or _utc_now_iso())
    return out


def _default_baseline(cfg: Any) -> dict[str, Any]:
    primitives: dict[str, Any] = {}
    for kind in _PRIMITIVE_ORDER:
        primitives[kind.value] = _canonicalize_command_payload(kind, cfg)
    return _with_baseline_metadata({"primitives": primitives})


def _normalize_baseline(raw: Any, cfg: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("baseline must be a JSON object")

    try:
        version = int(raw.get("version"))
    except Exception as exc:
        raise ValueError("baseline version is required") from exc
    if version != _BASELINE_VERSION:
        raise ValueError(f"unsupported baseline version: {version} (expected {_BASELINE_VERSION})")

    raw_primitives = raw.get("primitives")
    if not isinstance(raw_primitives, Mapping):
        raise ValueError("baseline.primitives must be an object")

    updated_by = str(raw.get("updated_by") or _BASELINE_UPDATED_BY)
    updated_at = str(raw.get("updated_at_utc") or "").strip() or None

    normalized = _default_baseline(cfg)
    for kind in _PRIMITIVE_ORDER:
        payload = raw_primitives.get(kind.value)
        if not isinstance(payload, Mapping):
            raise ValueError(f"missing baseline primitive payload: {kind.value}")
        normalized["primitives"][kind.value] = _canonicalize_command_payload(kind, cfg, patch=payload)

    return _with_baseline_metadata(normalized, updated_by=updated_by, updated_at_utc=updated_at)


def _load_baseline(path: Path, cfg: Any) -> dict[str, Any]:
    if not path.exists():
        return _default_baseline(cfg)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid baseline JSON at {path}: {exc}") from exc

    return _normalize_baseline(raw, cfg)


def _save_baseline(path: Path, baseline: Mapping[str, Any]) -> None:
    updated_by = str(baseline.get("updated_by") or _BASELINE_UPDATED_BY)
    updated_at = str(baseline.get("updated_at_utc") or "").strip() or None
    canonical = _with_baseline_metadata(baseline, updated_by=updated_by, updated_at_utc=updated_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(canonical, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_yaml_config(config_path: Path) -> Mapping[str, Any]:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(raw, Mapping):
        return {}
    return raw


def _extract_lamp_geometry_overrides(raw: Mapping[str, Any]) -> dict[str, float]:
    candidates: list[Any] = []
    candidates.append(raw.get("lamp_geometry"))

    sim_viewer = raw.get("sim_viewer")
    if isinstance(sim_viewer, Mapping):
        candidates.append(sim_viewer.get("lamp_geometry"))

    primitive_sim = raw.get("primitive_sim")
    if isinstance(primitive_sim, Mapping):
        candidates.append(primitive_sim.get("lamp_geometry"))

    tools = raw.get("tools")
    if isinstance(tools, Mapping):
        tools_primitive_sim = tools.get("primitive_sim")
        if isinstance(tools_primitive_sim, Mapping):
            candidates.append(tools_primitive_sim.get("lamp_geometry"))

    section = next((c for c in candidates if isinstance(c, Mapping)), None)
    if not isinstance(section, Mapping):
        return {}

    out: dict[str, float] = {}
    for out_key, aliases in _GEOM_PARAM_ALIASES.items():
        value = None
        for alias in aliases:
            if alias in section:
                value = section.get(alias)
                break
        if value is None:
            continue
        try:
            fv = float(value)
        except Exception:
            continue
        if out_key in _GEOM_POSITIVE_KEYS and fv <= 0.0:
            continue
        out[out_key] = fv
    return out


def _extract_dh_geometry_overrides(raw: Mapping[str, Any]) -> dict[str, float]:
    dh = raw.get("dh_params")
    if not isinstance(dh, Mapping):
        return {}

    out: dict[str, float] = {}

    def _float_or_none(key: str) -> float | None:
        val = dh.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None

    def _deg_or_none(key: str) -> float | None:
        v = _float_or_none(key)
        if v is None:
            return None
        return v * 0.017453292519943295

    pitch1_a = _float_or_none("pitch1_a")
    if pitch1_a is not None and pitch1_a > 0:
        out["upperArmLen"] = pitch1_a

    # Legacy DH represents the forearm translation on roll_d.
    roll_d = _float_or_none("roll_d")
    if roll_d is not None and roll_d > 0:
        out["foreArmLen"] = roll_d

    yaw_d = _float_or_none("yaw_d")
    if yaw_d is not None and yaw_d > 0:
        out["hubRise"] = yaw_d

    pitch1_offset = _deg_or_none("pitch1_theta0_deg")
    if pitch1_offset is not None:
        out["pitch1ZeroOffsetRad"] = pitch1_offset

    pitch2_offset = _deg_or_none("pitch2_theta0_deg")
    if pitch2_offset is not None:
        out["pitch2ZeroOffsetRad"] = pitch2_offset

    pitch3_offset = _deg_or_none("pitch3_theta0_deg")
    if pitch3_offset is not None:
        out["pitch3ZeroOffsetRad"] = pitch3_offset

    return out


def _extract_numeric_dh_params(raw: Mapping[str, Any]) -> dict[str, float]:
    dh = raw.get("dh_params")
    if not isinstance(dh, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in dh.items():
        if value is None:
            continue
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def _load_viewer_geometry_from_raw(raw: Mapping[str, Any]) -> dict[str, float]:
    explicit = _extract_lamp_geometry_overrides(raw)
    dh = _extract_dh_geometry_overrides(raw)

    out: dict[str, float] = {}
    out.update(dh)
    out.update(explicit)
    return out


def _load_viewer_geometry_from_config(config_path: Path) -> dict[str, float]:
    raw = _read_yaml_config(config_path)
    return _load_viewer_geometry_from_raw(raw)


def _build_state_machine_simulator(cfg: Any, raw_config: Mapping[str, Any]) -> Any:
    try:
        from tools.primitive_sim.state_machine import create_state_machine_simulator
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "primitive_sim state_machine mode failed to load. "
            "Check tools/primitive_sim/state_machine.py for import/runtime errors."
        ) from exc
    return create_state_machine_simulator(cfg=cfg, raw_config=raw_config)


def _default_joint_angles(cfg: Any) -> list[float]:
    out: list[float] = []
    names = list(getattr(cfg, "joint_names", []))
    limits = list(getattr(cfg, "joint_limits_rad", []))
    for idx in range(len(names)):
        if idx < len(limits) and isinstance(limits[idx], Sequence) and len(limits[idx]) >= 2:
            lo = float(limits[idx][0])
            hi = float(limits[idx][1])
            if lo > hi:
                lo, hi = hi, lo
        else:
            lo, hi = -1.57079632679, 1.57079632679
        out.append(max(lo, min(0.0, hi)))
    return out


def _joint_limits_by_name(cfg: Any) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(cfg.joint_names):
        lo, hi = cfg.joint_limits_rad[i]
        out[str(name).strip().lower()] = (float(lo), float(hi))
    return out


def _primitive_specs(cfg: Any) -> list[dict[str, Any]]:
    limits = _joint_limits_by_name(cfg)
    yaw_lim = limits.get("yaw", (-1.57, 1.57))
    pitch_lim = limits.get("pitch3", (-1.57, 1.57))

    move_mins = [float(l[0]) for l in cfg.joint_limits_rad]
    move_maxs = [float(l[1]) for l in cfg.joint_limits_rad]

    return [
        {"id": "hold", "label": "Hold", "params": []},
        {
            "id": "home",
            "label": "Home",
            "params": [
                {"name": "rate_rad_s", "label": "Rate", "type": "float", "min": 0.1, "max": 6.0, "step": 0.05},
            ],
        },
        {
            "id": "move_to",
            "label": "Move To",
            "params": [
                {
                    "name": "target_rad",
                    "label": "Target (rad)",
                    "type": "vector",
                    "labels": [str(n) for n in cfg.joint_names],
                    "mins": move_mins,
                    "maxs": move_maxs,
                    "step": 0.01,
                },
                {"name": "relative", "label": "Relative", "type": "bool"},
                {"name": "rate_rad_s", "label": "Rate", "type": "float", "min": 0.1, "max": 6.0, "step": 0.05},
                {"name": "timeout_s", "label": "Timeout", "type": "float", "min": 0.1, "max": 6.0, "step": 0.05},
            ],
        },
        {
            "id": "gaze_to",
            "label": "Gaze To",
            "params": [
                {
                    "name": "yaw_rad",
                    "label": "Yaw",
                    "type": "float",
                    "min": float(yaw_lim[0]),
                    "max": float(yaw_lim[1]),
                    "step": 0.01,
                },
                {
                    "name": "pitch_rad",
                    "label": "Pitch",
                    "type": "float",
                    "min": float(pitch_lim[0]),
                    "max": float(pitch_lim[1]),
                    "step": 0.01,
                },
                {"name": "rate_rad_s", "label": "Rate", "type": "float", "min": 0.1, "max": 6.0, "step": 0.05},
                {"name": "dwell_s", "label": "Dwell", "type": "float", "min": 0.0, "max": 4.0, "step": 0.05},
                {"name": "timeout_s", "label": "Timeout", "type": "float", "min": 0.1, "max": 8.0, "step": 0.05},
            ],
        },
        {
            "id": "glance",
            "label": "Glance",
            "params": [
                {"name": "direction", "label": "Direction", "type": "enum", "options": ["left", "right", "up", "down"]},
                {"name": "amp_rad", "label": "Amplitude", "type": "float", "min": 0.01, "max": 1.2, "step": 0.01},
                {"name": "duration_s", "label": "Duration", "type": "float", "min": 0.05, "max": 4.0, "step": 0.01},
                {"name": "rate_rad_s", "label": "Rate", "type": "float", "min": 0.1, "max": 8.0, "step": 0.05},
            ],
        },
        {
            "id": "nod",
            "label": "Nod",
            "params": [
                {"name": "amp_rad", "label": "Amplitude", "type": "float", "min": 0.01, "max": 1.2, "step": 0.01},
                {"name": "duration_s", "label": "Duration", "type": "float", "min": 0.05, "max": 4.0, "step": 0.01},
                {"name": "cycles", "label": "Cycles", "type": "int", "min": 1, "max": 6, "step": 1},
                {"name": "rate_rad_s", "label": "Rate", "type": "float", "min": 0.1, "max": 8.0, "step": 0.05},
            ],
        },
        {
            "id": "breath",
            "label": "Breath",
            "params": [
                {"name": "amp_rad", "label": "Amplitude", "type": "float", "min": 0.01, "max": 0.6, "step": 0.005},
                {"name": "period_s", "label": "Period", "type": "float", "min": 0.2, "max": 30.0, "step": 0.05},
                {"name": "rate_rad_s", "label": "Rate", "type": "float", "min": 0.1, "max": 4.0, "step": 0.05},
            ],
        },
        {
            "id": "orient_to_zone",
            "label": "Orient To Zone",
            "params": [
                {"name": "zone", "label": "Zone", "type": "enum", "options": ["left", "center", "right"]},
                {"name": "amp_rad", "label": "Amplitude", "type": "float", "min": 0.01, "max": 1.2, "step": 0.01},
                {"name": "rate_rad_s", "label": "Rate", "type": "float", "min": 0.1, "max": 8.0, "step": 0.05},
            ],
        },
    ]


def _style_options(cfg: Any) -> list[str]:
    styles = [str(k).strip().lower() for k in getattr(cfg, "style_profiles", {}).keys()]
    styles = [s for s in styles if s]
    if not styles:
        return ["calm"]
    ordered = ["calm", "curious", "focused"]
    remaining = sorted([s for s in styles if s not in ordered])
    return [s for s in ordered if s in styles] + remaining


def _default_duration_s(primitive: PrimitiveKind) -> float:
    if primitive == PrimitiveKind.BREATH:
        return 5.0
    if primitive == PrimitiveKind.HOLD:
        return 3.0
    if primitive == PrimitiveKind.HOME:
        return 2.0
    if primitive == PrimitiveKind.MOVE_TO:
        return 2.4
    if primitive == PrimitiveKind.GAZE_TO:
        return 2.0
    if primitive == PrimitiveKind.GLANCE:
        return 1.2
    if primitive == PrimitiveKind.NOD:
        return 1.5
    if primitive == PrimitiveKind.ORIENT_TO_ZONE:
        return 1.8
    return 2.0


def _scenario_default_steps(cfg: Any) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    seed = [
        ("home", PrimitiveKind.HOME),
        ("breath", PrimitiveKind.BREATH),
        ("glance_left", PrimitiveKind.GLANCE),
        ("orient_center", PrimitiveKind.ORIENT_TO_ZONE),
        ("nod", PrimitiveKind.NOD),
    ]
    for name, primitive in seed:
        cmd = _default_command_payload(primitive, cfg)
        if primitive == PrimitiveKind.GLANCE:
            cmd = dict(cmd)
            cmd["direction"] = "left"
        if primitive == PrimitiveKind.ORIENT_TO_ZONE:
            cmd = dict(cmd)
            cmd["zone"] = "center"
        steps.append(
            {
                "name": name,
                "primitive": primitive.value,
                "style": "calm",
                "duration_s": _default_duration_s(primitive),
                "stop_on_done": primitive not in CONTINUOUS_PRIMITIVES,
                "command": cmd,
            }
        )
    return steps


def _segments_to_payload(segments: Sequence[SimSegment]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        out.append(
            {
                "name": str(seg.name),
                "duration_s": float(seg.max_s),
                "stop_on_done": bool(seg.stop_on_done),
                "primitive": seg.action.primitive.value,
                "style": str(seg.action.style or "calm"),
                "command": dict(to_json_dict(seg.action.command)),
            }
        )
    return out


def _scenario_segments_from_payload(
    *,
    raw_steps: Any,
    cfg: Any,
    style_options: Sequence[str],
    baseline: Mapping[str, Any],
    default_style: str = "calm",
) -> list[SimSegment]:
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("steps must be a non-empty array")

    styles = [str(v).strip().lower() for v in style_options if str(v).strip()]
    if not styles:
        styles = ["calm"]

    fallback_style = str(default_style or "calm").strip().lower()
    if fallback_style not in styles:
        fallback_style = styles[0]

    out: list[SimSegment] = []
    for idx, item in enumerate(raw_steps):
        if not isinstance(item, Mapping):
            raise ValueError(f"steps[{idx}] must be an object")

        try:
            primitive = _parse_primitive(item.get("primitive", ""))
        except ValueError as exc:
            raise ValueError(f"steps[{idx}] invalid primitive: {exc}") from exc

        style = str(item.get("style", fallback_style)).strip().lower() or fallback_style
        if style not in styles:
            style = fallback_style

        name = str(item.get("name") or f"step_{idx + 1:02d}_{primitive.value}").strip()
        if not name:
            name = f"step_{idx + 1:02d}_{primitive.value}"

        duration_raw = item.get("duration_s", item.get("max_s"))
        duration_s = max(0.05, _coerce_float(duration_raw, _default_duration_s(primitive)))
        stop_default = primitive not in CONTINUOUS_PRIMITIVES
        stop_on_done = _coerce_bool(item.get("stop_on_done"), stop_default)

        command_raw = item.get("command")
        if command_raw is None:
            command_raw = {}
        if not isinstance(command_raw, Mapping):
            raise ValueError(f"steps[{idx}].command must be an object")

        use_baseline = _coerce_bool(item.get("use_baseline"), True)
        base_cmd: Mapping[str, Any] | None = None
        if use_baseline:
            src = baseline.get("primitives", {}).get(primitive.value, {})
            if isinstance(src, Mapping):
                base_cmd = src

        canonical = _canonicalize_command_payload(
            primitive,
            cfg,
            style=style,
            base=base_cmd,
            patch=command_raw,
        )
        action = ActionPlan(
            primitive=primitive,
            command=canonical,
            confidence=1.0,
            explanation="primitive_scenario",
            style=style,
        )
        out.append(
            SimSegment(
                name=name,
                action=action,
                max_s=duration_s,
                stop_on_done=stop_on_done,
            )
        )
    return out


def _trace_metrics(trace: Mapping[str, Any]) -> dict[str, Any]:
    metadata = trace.get("metadata")
    if not isinstance(metadata, Mapping):
        return {}
    names = metadata.get("joint_names")
    limits = metadata.get("joint_limits_rad")
    samples = trace.get("samples")
    if not isinstance(names, list) or not isinstance(limits, list) or not isinstance(samples, list):
        return {}
    if not names or not samples or len(limits) != len(names):
        return {
            "sample_count": len(samples) if isinstance(samples, list) else 0,
            "duration_s": 0.0,
            "limit_violation_count": 0,
            "min_limit_margin_rad": 0.0,
            "mean_abs_joint_vel_rad_s": 0.0,
            "path_length_rad": 0.0,
            "primitive_switch_count": 0,
            "peak_joint_vel_rad_s": 0.0,
        }

    duration_s = max(0.0, float(samples[-1].get("t_s", 0.0)) - float(samples[0].get("t_s", 0.0)))
    violation_count = 0
    min_margin = float("inf")
    primitive_switches = 0
    prev_active = None
    sum_vel = 0.0
    vel_count = 0
    path_length = 0.0
    prev_t = None
    prev_angles: list[float] | None = None

    for sample in samples:
        t = float(sample.get("t_s", 0.0))
        raw_angles = sample.get("joint_angles_rad")
        if not isinstance(raw_angles, list) or len(raw_angles) != len(names):
            continue
        angles = [float(v) for v in raw_angles]

        for i, angle in enumerate(angles):
            lim = limits[i]
            if not isinstance(lim, Sequence) or len(lim) < 2:
                continue
            lo = float(lim[0])
            hi = float(lim[1])
            margin = min(angle - lo, hi - angle)
            if margin < min_margin:
                min_margin = margin
            if angle < lo or angle > hi:
                violation_count += 1

        active = sample.get("active_primitive")
        if prev_active is not None and active != prev_active:
            primitive_switches += 1
        prev_active = active

        if prev_angles is not None and prev_t is not None:
            dt = max(1e-6, t - prev_t)
            for i, angle in enumerate(angles):
                delta = angle - prev_angles[i]
                path_length += abs(delta)
                sum_vel += abs(delta / dt)
                vel_count += 1
        prev_angles = angles
        prev_t = t

    summary = trace.get("summary")
    peak_vel = 0.0
    if isinstance(summary, Mapping):
        stats = summary.get("joint_stats")
        if isinstance(stats, list):
            for row in stats:
                if not isinstance(row, Mapping):
                    continue
                peak_vel = max(peak_vel, abs(float(row.get("peak_vel_rad_s", 0.0))))

    return {
        "sample_count": len(samples),
        "duration_s": float(duration_s),
        "limit_violation_count": int(violation_count),
        "min_limit_margin_rad": float(min_margin if min_margin != float("inf") else 0.0),
        "mean_abs_joint_vel_rad_s": float(sum_vel / vel_count if vel_count > 0 else 0.0),
        "path_length_rad": float(path_length),
        "primitive_switch_count": int(primitive_switches),
        "peak_joint_vel_rad_s": float(peak_vel),
    }


def _append_experiment_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), ensure_ascii=True) + "\n")


def _load_experiment_history(path: Path, *, limit: int = 25) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    lines = path.read_text(encoding="utf-8").splitlines()
    for raw in reversed(lines):
        token = raw.strip()
        if not token:
            continue
        try:
            row = json.loads(token)
        except Exception:
            continue
        if isinstance(row, Mapping):
            out.append(dict(row))
        if len(out) >= max(1, int(limit)):
            break
    return out


def _json_response(handler: http.server.BaseHTTPRequestHandler, status: int, payload: Mapping[str, Any]) -> None:
    body = (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _json_error(
    handler: http.server.BaseHTTPRequestHandler,
    status: int,
    *,
    code: str,
    error: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {"code": str(code), "error": str(error)}
    if isinstance(details, Mapping) and details:
        payload["details"] = dict(details)
    _json_response(handler, status, payload)


def _read_json_body(handler: http.server.BaseHTTPRequestHandler) -> dict[str, Any]:
    size = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(size) if size > 0 else b"{}"
    payload = json.loads(raw.decode("utf-8") if raw else "{}")
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


class _StudioRequestHandler(http.server.SimpleHTTPRequestHandler):
    studio: _StudioContext

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/primitives":
            _json_response(
                self,
                200,
                {
                    "primitives": self.studio.primitive_specs,
                    "styles": self.studio.style_options,
                    "joint_names": [str(n) for n in self.studio.cfg.joint_names],
                    "joint_limits_rad": [[float(v[0]), float(v[1])] for v in self.studio.cfg.joint_limits_rad],
                    "default_primitive": "breath",
                },
            )
            return

        if parsed.path == "/api/baseline":
            with self.studio.lock:
                payload = copy.deepcopy(self.studio.baseline)
            _json_response(self, 200, payload)
            return

        if parsed.path == "/api/joint_checker/meta":
            payload = {
                "joint_names": [str(n) for n in self.studio.cfg.joint_names],
                "joint_limits_rad": [[float(v[0]), float(v[1])] for v in self.studio.cfg.joint_limits_rad],
                "default_angles_rad": _default_joint_angles(self.studio.cfg),
                "lamp_geometry": copy.deepcopy(self.studio.lamp_geometry),
                "dh_params": copy.deepcopy(self.studio.dh_params),
                "config_path": str(self.studio.config_path),
            }
            _json_response(self, 200, payload)
            return

        if parsed.path == "/api/state_machine/meta":
            with self.studio.lock:
                payload = self.studio.fsm_sim.meta()
            payload["config_path"] = str(self.studio.config_path)
            _json_response(self, 200, payload)
            return

        if parsed.path == "/api/scenario/meta":
            with self.studio.lock:
                baseline = copy.deepcopy(self.studio.baseline)
            payload = {
                "primitives": self.studio.primitive_specs,
                "styles": self.studio.style_options,
                "default_steps": _scenario_default_steps(self.studio.cfg),
                "baseline": baseline,
                "sweep": {
                    "max_candidates": _MAX_SWEEP_CANDIDATES,
                    "default_top_k": _DEFAULT_SWEEP_TOP_K,
                    "default_weights": _default_sweep_weights(),
                },
                "config_path": str(self.studio.config_path),
            }
            _json_response(self, 200, payload)
            return

        if parsed.path == "/api/scenario/history":
            query = parse_qs(parsed.query)
            limit = _coerce_int((query.get("limit") or ["25"])[0], 25)
            limit = max(1, min(200, limit))
            with self.studio.lock:
                history = _load_experiment_history(self.studio.experiments_path, limit=limit)
            _json_response(self, 200, {"history": history})
            return

        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        try:
            payload = _read_json_body(self)
        except Exception as exc:
            _json_error(self, 400, code="invalid_json", error=f"invalid_json: {exc}")
            return

        if parsed.path == "/api/simulate":
            self._handle_simulate(payload)
            return

        if parsed.path == "/api/baseline":
            self._handle_baseline_update(payload)
            return

        if parsed.path == "/api/suite":
            self._handle_suite(payload)
            return

        if parsed.path == "/api/state_machine/reset":
            self._handle_state_machine_reset()
            return

        if parsed.path == "/api/state_machine/step":
            self._handle_state_machine_step(payload)
            return

        if parsed.path == "/api/state_machine/force":
            self._handle_state_machine_force(payload)
            return

        if parsed.path == "/api/scenario/simulate":
            self._handle_scenario_simulate(payload)
            return

        if parsed.path == "/api/scenario/sweep":
            self._handle_scenario_sweep(payload)
            return

        if parsed.path == "/api/scenario/save_experiment":
            self._handle_scenario_save_experiment(payload)
            return

        _json_error(self, 404, code="not_found", error="not_found")

    def _handle_simulate(self, payload: Mapping[str, Any]) -> None:
        try:
            primitive = _parse_primitive(payload.get("primitive", "breath"))
        except ValueError as exc:
            _json_error(self, 400, code="invalid_primitive", error=f"invalid_primitive: {exc}")
            return

        style = str(payload.get("style", "calm")).strip().lower() or "calm"
        if style not in self.studio.style_options:
            style = self.studio.style_options[0]

        cmd_patch = payload.get("command")
        if cmd_patch is None:
            cmd_patch = {}
        if not isinstance(cmd_patch, Mapping):
            _json_error(
                self,
                400,
                code="invalid_command",
                error="command must be an object",
                details={"field": "command"},
            )
            return

        duration_s = max(0.05, _coerce_float(payload.get("duration_s"), _default_duration_s(primitive)))
        stop_on_done = _coerce_bool(payload.get("stop_on_done"), primitive not in CONTINUOUS_PRIMITIVES)
        name = str(payload.get("name") or f"studio_{primitive.value}").strip() or f"studio_{primitive.value}"

        with self.studio.lock:
            baseline_cmd = self.studio.baseline["primitives"].get(primitive.value, {})

        try:
            canonical_cmd = _canonicalize_command_payload(
                primitive,
                self.studio.cfg,
                style=style,
                base=baseline_cmd,
                patch=cmd_patch,
            )
            action = ActionPlan(
                primitive=primitive,
                command=canonical_cmd,
                confidence=1.0,
                explanation="primitive_studio",
                style=style,
            )
        except ValueError as exc:
            _json_error(self, 400, code="validation_error", error=str(exc))
            return

        segment = SimSegment(name=name, action=action, max_s=duration_s, stop_on_done=stop_on_done)
        trace = simulate_segments(
            joint_names=self.studio.cfg.joint_names,
            joint_limits_rad=self.studio.cfg.joint_limits_rad,
            segments=[segment],
            hz=self.studio.hz,
            style_profiles=self.studio.cfg.style_profiles,
        )
        trace["metadata"]["scenario"] = "studio"
        trace["metadata"]["config_path"] = str(self.studio.config_path)
        trace["metadata"]["primitive"] = primitive.value
        trace["metadata"]["style"] = style
        if self.studio.lamp_geometry:
            trace["metadata"]["lamp_geometry"] = self.studio.lamp_geometry

        _json_response(self, 200, trace)

    def _handle_baseline_update(self, payload: Mapping[str, Any]) -> None:
        # Patch single primitive
        if "primitive" in payload:
            primitive_raw = payload.get("primitive")
            command_raw = payload.get("command")
            if not isinstance(command_raw, Mapping):
                _json_error(
                    self,
                    400,
                    code="invalid_command",
                    error="command must be an object",
                    details={"field": "command"},
                )
                return
            try:
                primitive = _parse_primitive(primitive_raw)
            except ValueError as exc:
                _json_error(self, 400, code="invalid_primitive", error=f"invalid_primitive: {exc}")
                return

            with self.studio.lock:
                base_cmd = self.studio.baseline["primitives"].get(primitive.value, {})
                try:
                    canonical = _canonicalize_command_payload(
                        primitive,
                        self.studio.cfg,
                        base=base_cmd,
                        patch=command_raw,
                    )
                except ValueError as exc:
                    _json_error(self, 400, code="validation_error", error=str(exc))
                    return

                updated = copy.deepcopy(self.studio.baseline)
                updated["primitives"][primitive.value] = canonical
                updated = _with_baseline_metadata(updated)
                _save_baseline(self.studio.baseline_path, updated)
                self.studio.baseline = updated
                response = copy.deepcopy(updated)

            _json_response(self, 200, response)
            return

        # Replace full baseline object
        if "primitives" in payload:
            normalized = _normalize_baseline(payload, self.studio.cfg)
            with self.studio.lock:
                normalized = _with_baseline_metadata(normalized)
                _save_baseline(self.studio.baseline_path, normalized)
                self.studio.baseline = normalized
                response = copy.deepcopy(normalized)
            _json_response(self, 200, response)
            return

        _json_error(
            self,
            400,
            code="invalid_payload",
            error="payload must include primitive+command or primitives",
        )

    def _handle_suite(self, payload: Mapping[str, Any]) -> None:
        requested = str(payload.get("style", "calm")).strip().lower()
        style = requested if requested in self.studio.style_options else self.studio.style_options[0]
        suite_breath_s = max(0.2, _coerce_float(payload.get("suite_breath_s"), 6.0))
        out_path = _abs_path(str(payload.get("output") or _DEFAULT_SUITE_TRACE_REL))

        with self.studio.lock:
            response = _generate_suite_trace(
                cfg=self.studio.cfg,
                hz=self.studio.hz,
                config_path=self.studio.config_path,
                lamp_geometry=self.studio.lamp_geometry,
                style=style,
                suite_breath_s=suite_breath_s,
                output_path=out_path,
            )
        _json_response(self, 200, response)

    def _handle_state_machine_reset(self) -> None:
        with self.studio.lock:
            state = self.studio.fsm_sim.reset()
            meta = self.studio.fsm_sim.meta()
        _json_response(
            self,
            200,
            {
                "state": state,
                "meta": meta,
            },
        )

    def _handle_state_machine_step(self, payload: Mapping[str, Any]) -> None:
        dt_s = _coerce_float(payload.get("dt_s"), 0.35)
        commit = _coerce_bool(payload.get("commit"), False)
        zone_hint = payload.get("zone_hint")
        signals_raw = payload.get("signals")
        if signals_raw is not None and not isinstance(signals_raw, Mapping):
            _json_error(
                self,
                400,
                code="invalid_signals",
                error="signals must be an object when provided",
                details={"field": "signals"},
            )
            return
        with self.studio.lock:
            result = self.studio.fsm_sim.step(
                dt_s=dt_s,
                signals=signals_raw if isinstance(signals_raw, Mapping) else None,
                zone_hint=zone_hint,
                commit=commit,
            )
        _json_response(self, 200, result)

    def _handle_state_machine_force(self, payload: Mapping[str, Any]) -> None:
        mode_raw = payload.get("mode")
        if mode_raw is None:
            _json_error(
                self,
                400,
                code="invalid_mode",
                error="mode is required",
                details={"field": "mode"},
            )
            return
        reason = str(payload.get("reason") or "force_mode").strip() or "force_mode"
        dt_s = _coerce_float(payload.get("dt_s"), 0.0)
        commit = _coerce_bool(payload.get("commit"), False)
        zone_hint = payload.get("zone_hint")
        signals_raw = payload.get("signals")
        if signals_raw is not None and not isinstance(signals_raw, Mapping):
            _json_error(
                self,
                400,
                code="invalid_signals",
                error="signals must be an object when provided",
                details={"field": "signals"},
            )
            return
        with self.studio.lock:
            try:
                result = self.studio.fsm_sim.force_mode(
                    next_mode=mode_raw,
                    reason=reason,
                    dt_s=dt_s,
                    signals=signals_raw if isinstance(signals_raw, Mapping) else None,
                    zone_hint=zone_hint,
                    commit=commit,
                )
            except ValueError as exc:
                _json_error(
                    self,
                    400,
                    code="invalid_mode",
                    error=str(exc),
                    details={"field": "mode"},
                )
                return
        _json_response(self, 200, result)

    def _handle_scenario_simulate(self, payload: Mapping[str, Any]) -> None:
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            _json_error(
                self,
                400,
                code="invalid_steps",
                error="steps must be an array",
                details={"field": "steps"},
            )
            return
        style_default = str(payload.get("style", "calm")).strip().lower() or "calm"
        run_name = str(payload.get("name") or "scenario_run").strip() or "scenario_run"
        dry_run = _coerce_bool(payload.get("dry_run"), False)
        out_path = _abs_path(str(payload.get("output") or _DEFAULT_SCENARIO_TRACE_REL))

        try:
            with self.studio.lock:
                segments = _scenario_segments_from_payload(
                    raw_steps=raw_steps,
                    cfg=self.studio.cfg,
                    style_options=self.studio.style_options,
                    baseline=self.studio.baseline,
                    default_style=style_default,
                )
        except ValueError as exc:
            _json_error(self, 400, code="scenario_validation_error", error=str(exc))
            return

        if dry_run:
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "dry_run": True,
                    "segment_count": len(segments),
                    "steps": _segments_to_payload(segments),
                },
            )
            return

        trace = simulate_segments(
            joint_names=self.studio.cfg.joint_names,
            joint_limits_rad=self.studio.cfg.joint_limits_rad,
            segments=segments,
            hz=self.studio.hz,
            style_profiles=self.studio.cfg.style_profiles,
        )
        trace["metadata"]["scenario"] = "scenario_lab"
        trace["metadata"]["config_path"] = str(self.studio.config_path)
        trace["metadata"]["run_name"] = run_name
        if self.studio.lamp_geometry:
            trace["metadata"]["lamp_geometry"] = self.studio.lamp_geometry

        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_trace_json(out_path, trace)
        rel = _relative_trace_path(out_path)
        viewer_url = "/tools/primitive_sim/web/lamp_sim.html?mode=playback"
        if rel:
            viewer_url += f"&trace={quote(rel, safe='/')}"
        else:
            viewer_url = "/tools/primitive_sim/web/lamp_sim.html?mode=studio"

        _json_response(
            self,
            200,
            {
                "ok": True,
                "dry_run": False,
                "run_name": run_name,
                "trace_path": str(out_path),
                "trace_url": rel,
                "viewer_url": viewer_url,
                "sample_count": len(trace.get("samples", [])),
                "metrics": _trace_metrics(trace),
                "steps": _segments_to_payload(segments),
            },
        )

    def _handle_scenario_sweep(self, payload: Mapping[str, Any]) -> None:
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            _json_error(
                self,
                400,
                code="invalid_steps",
                error="steps must be a non-empty array",
                details={"field": "steps"},
            )
            return

        run_name = str(payload.get("name") or "scenario_sweep").strip() or "scenario_sweep"
        style_default = str(payload.get("style", "calm")).strip().lower() or "calm"
        save_traces = _coerce_bool(payload.get("save_traces"), False)
        top_k = _coerce_int(payload.get("top_k"), _DEFAULT_SWEEP_TOP_K)
        top_k = max(1, min(_MAX_SWEEP_CANDIDATES, top_k))

        try:
            target_idx = _target_step_index(raw_steps, payload.get("target_step"))
            combos = _expand_sweep_grid(payload.get("param_grid"), max_candidates=_MAX_SWEEP_CANDIDATES)
        except ValueError as exc:
            _json_error(self, 400, code="invalid_sweep_request", error=str(exc))
            return

        score_weights = _normalize_sweep_weights(payload.get("score_weights"))
        sweep_id = f"{_safe_slug(run_name)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        output_root = _abs_path(str(payload.get("output_dir") or _DEFAULT_SWEEP_DIR_REL))
        sweep_dir = output_root / sweep_id

        with self.studio.lock:
            baseline = copy.deepcopy(self.studio.baseline)

        candidates: list[dict[str, Any]] = []
        errors: list[str] = []
        for i, patch in enumerate(combos):
            steps_variant = copy.deepcopy(raw_steps)
            target_step = steps_variant[target_idx]
            if not isinstance(target_step, Mapping):
                errors.append(f"candidate {i + 1}: steps[{target_idx}] must be an object")
                continue

            step_copy: dict[str, Any] = dict(target_step)
            command_src = step_copy.get("command")
            command_copy = dict(command_src) if isinstance(command_src, Mapping) else {}
            command_copy.update(patch)
            step_copy["command"] = command_copy
            steps_variant[target_idx] = step_copy

            try:
                segments = _scenario_segments_from_payload(
                    raw_steps=steps_variant,
                    cfg=self.studio.cfg,
                    style_options=self.studio.style_options,
                    baseline=baseline,
                    default_style=style_default,
                )
            except ValueError as exc:
                errors.append(f"candidate {i + 1}: {exc}")
                continue

            trace = simulate_segments(
                joint_names=self.studio.cfg.joint_names,
                joint_limits_rad=self.studio.cfg.joint_limits_rad,
                segments=segments,
                hz=self.studio.hz,
                style_profiles=self.studio.cfg.style_profiles,
            )
            trace["metadata"]["scenario"] = "scenario_sweep"
            trace["metadata"]["config_path"] = str(self.studio.config_path)
            trace["metadata"]["run_name"] = run_name
            trace["metadata"]["sweep_id"] = sweep_id
            trace["metadata"]["sweep_candidate_index"] = i + 1
            trace["metadata"]["sweep_patch"] = dict(patch)
            if self.studio.lamp_geometry:
                trace["metadata"]["lamp_geometry"] = self.studio.lamp_geometry

            metrics = _trace_metrics(trace)
            score = _score_trace_metrics(metrics, score_weights)

            trace_path = ""
            trace_url = ""
            if save_traces:
                sweep_dir.mkdir(parents=True, exist_ok=True)
                trace_file = sweep_dir / f"candidate_{i + 1:03d}.json"
                write_trace_json(trace_file, trace)
                trace_path = str(trace_file)
                rel = _relative_trace_path(trace_file)
                if rel:
                    trace_url = rel

            candidates.append(
                {
                    "candidate_index": i + 1,
                    "score": float(score),
                    "param_patch": dict(patch),
                    "metrics": metrics,
                    "steps": _segments_to_payload(segments),
                    "trace_path": trace_path,
                    "trace_url": trace_url,
                }
            )

        if not candidates:
            _json_error(
                self,
                400,
                code="sweep_no_candidates",
                error="all sweep candidates failed validation",
                details={"errors": errors[:10]},
            )
            return

        candidates.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        for i, row in enumerate(candidates, start=1):
            row["rank"] = i

        top_rows = candidates[: max(1, min(top_k, len(candidates)))]
        best = dict(top_rows[0])
        _json_response(
            self,
            200,
            {
                "ok": True,
                "run_name": run_name,
                "sweep_id": sweep_id,
                "target_step_index": target_idx,
                "candidate_count": len(combos),
                "valid_count": len(candidates),
                "error_count": len(errors),
                "errors": errors[:20],
                "score_weights": score_weights,
                "ranking": top_rows,
                "best": best,
            },
        )

    def _handle_scenario_save_experiment(self, payload: Mapping[str, Any]) -> None:
        name = str(payload.get("name") or "").strip()
        if not name:
            _json_error(
                self,
                400,
                code="invalid_name",
                error="name is required",
                details={"field": "name"},
            )
            return
        notes = str(payload.get("notes") or "").strip()
        steps = payload.get("steps")
        if not isinstance(steps, list) or not steps:
            _json_error(
                self,
                400,
                code="invalid_steps",
                error="steps must be a non-empty array",
                details={"field": "steps"},
            )
            return
        metrics = payload.get("metrics")
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, Mapping):
            _json_error(
                self,
                400,
                code="invalid_metrics",
                error="metrics must be an object",
                details={"field": "metrics"},
            )
            return
        trace_path = str(payload.get("trace_path") or "").strip()
        trace_url = str(payload.get("trace_url") or "").strip()

        record = {
            "saved_at_utc": _utc_now_iso(),
            "name": name,
            "notes": notes,
            "steps": steps,
            "metrics": dict(metrics),
            "trace_path": trace_path,
            "trace_url": trace_url,
        }
        with self.studio.lock:
            _append_experiment_record(self.studio.experiments_path, record)
            history = _load_experiment_history(self.studio.experiments_path, limit=25)

        _json_response(
            self,
            200,
            {
                "ok": True,
                "record": record,
                "history": history,
            },
        )


def _relative_trace_path(path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(ROOT)
    except ValueError:
        return None
    return "/" + rel.as_posix()


def _generate_suite_trace(
    *,
    cfg: Any,
    hz: float,
    config_path: Path,
    lamp_geometry: Mapping[str, float],
    style: str,
    suite_breath_s: float,
    output_path: Path,
) -> dict[str, Any]:
    segments = build_suite_segments(suite_breath_s=float(suite_breath_s), style=str(style))
    trace = simulate_segments(
        joint_names=cfg.joint_names,
        joint_limits_rad=cfg.joint_limits_rad,
        segments=segments,
        hz=hz,
        style_profiles=cfg.style_profiles,
    )
    trace["metadata"]["scenario"] = "suite"
    trace["metadata"]["config_path"] = str(config_path)
    trace["metadata"]["style"] = str(style)
    if lamp_geometry:
        trace["metadata"]["lamp_geometry"] = dict(lamp_geometry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trace_json(output_path, trace)

    rel = _relative_trace_path(output_path)
    viewer_url = "/tools/primitive_sim/web/lamp_sim.html?mode=playback"
    if rel:
        viewer_url += f"&trace={quote(rel, safe='/')}"
    else:
        viewer_url = "/tools/primitive_sim/web/lamp_sim.html?mode=studio"

    return {
        "ok": True,
        "trace_path": str(output_path),
        "trace_url": rel,
        "viewer_url": viewer_url,
        "sample_count": len(trace.get("samples", [])),
        "style": str(style),
    }


def _serve_toolbox(
    context: _StudioContext,
    port: int,
    *,
    landing: str = "studio",
    trace_path: Path | None = None,
) -> None:
    handler_cls = type("PrimitiveStudioHandler", (_StudioRequestHandler,), {"studio": context})
    with _ThreadingReuseTCPServer(("127.0.0.1", int(port)), handler_cls) as server:
        if landing == "joint_checker":
            url = f"http://127.0.0.1:{port}/tools/primitive_sim/web/lamp_sim.html?mode=joint_checker"
        elif landing == "state_machine":
            url = f"http://127.0.0.1:{port}/tools/primitive_sim/web/lamp_sim.html?mode=state_machine"
        elif landing == "scenario_lab":
            url = f"http://127.0.0.1:{port}/tools/primitive_sim/web/lamp_sim.html?mode=scenario_lab"
        elif landing == "playback":
            rel = _relative_trace_path(trace_path) if trace_path is not None else None
            url = f"http://127.0.0.1:{port}/tools/primitive_sim/web/lamp_sim.html?mode=playback"
            if rel:
                url += f"&trace={quote(rel, safe='/')}"
            else:
                url = f"http://127.0.0.1:{port}/tools/primitive_sim/web/lamp_sim.html?mode=studio"
        else:
            url = f"http://127.0.0.1:{port}/tools/primitive_sim/web/lamp_sim.html?mode=studio"
        print(f"lamp sim toolbox: {url}")
        print(f"shell: http://127.0.0.1:{port}/tools/primitive_sim/web/lamp_sim.html")
        print(f"studio(raw): http://127.0.0.1:{port}/tools/primitive_sim/web/index.html?studio=1")
        print(f"joint checker(raw): http://127.0.0.1:{port}/tools/primitive_sim/web/joint_checker.html")
        print(f"state machine(raw): http://127.0.0.1:{port}/tools/primitive_sim/web/state_machine.html")
        print(f"scenario lab(raw): http://127.0.0.1:{port}/tools/primitive_sim/web/scenario_lab.html")
        print(f"baseline: {context.baseline_path}")
        print(f"experiments: {context.experiments_path}")
        print("Press Ctrl-C to stop viewer server.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("viewer server stopped")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)

    hz = float(args.hz) if float(args.hz) > 0.0 else float(cfg.loop_rates.control_hz)
    hz = max(1.0, hz)

    config_path = _abs_path(args.config)
    raw_config = _read_yaml_config(config_path)
    lamp_geometry = _load_viewer_geometry_from_raw(raw_config)
    dh_params = _extract_numeric_dh_params(raw_config)

    if args.scenario in {"studio", "joint_checker", "state_machine", "scenario_lab"}:
        baseline_path = _abs_path(args.baseline)
        experiments_path = _abs_path(args.experiments)
        baseline = _load_baseline(baseline_path, cfg)
        if not baseline_path.exists():
            _save_baseline(baseline_path, baseline)
        fsm_sim = _build_state_machine_simulator(cfg, raw_config)

        context = _StudioContext(
            cfg=cfg,
            config_path=config_path,
            hz=hz,
            baseline_path=baseline_path,
            baseline=baseline,
            primitive_specs=_primitive_specs(cfg),
            style_options=_style_options(cfg),
            lamp_geometry=lamp_geometry,
            dh_params=dh_params,
            fsm_sim=fsm_sim,
            experiments_path=experiments_path,
        )
        if args.scenario == "joint_checker":
            landing = "joint_checker"
        elif args.scenario == "state_machine":
            landing = "state_machine"
        elif args.scenario == "scenario_lab":
            landing = "scenario_lab"
        else:
            landing = "studio"
        _serve_toolbox(context, int(args.port), landing=landing)
        return 0

    if args.scenario == "suite":
        segments = build_suite_segments(suite_breath_s=float(args.suite_breath_s), style=str(args.style))
    elif args.scenario == "single":
        segments = _build_single_segment(args, cfg)
    else:
        if not args.script:
            raise ValueError("--script is required when --scenario script")
        segments = load_segments_from_json(Path(args.script))

    trace = simulate_segments(
        joint_names=cfg.joint_names,
        joint_limits_rad=cfg.joint_limits_rad,
        segments=segments,
        hz=hz,
        style_profiles=cfg.style_profiles,
    )
    trace["metadata"]["scenario"] = str(args.scenario)
    trace["metadata"]["config_path"] = str(args.config)
    if lamp_geometry:
        trace["metadata"]["lamp_geometry"] = lamp_geometry

    output = Path(args.output)
    write_trace_json(output, trace)

    sample_count = len(trace["samples"])
    summary = trace.get("summary", {})
    print(f"primitive_sim: scenario={args.scenario} hz={hz:.1f} segments={len(segments)} samples={sample_count}")
    print(f"trace: {output}")
    print(f"status_counts: {summary.get('status_counts', {})}")

    joint_stats = summary.get("joint_stats", [])
    for row in joint_stats:
        print(
            "joint="
            f"{row.get('joint')} "
            f"min={float(row.get('min_rad', 0.0)):+.3f} "
            f"max={float(row.get('max_rad', 0.0)):+.3f} "
            f"peak_vel={float(row.get('peak_vel_rad_s', 0.0)):.3f}"
        )

    if args.serve:
        baseline_path = _abs_path(args.baseline)
        experiments_path = _abs_path(args.experiments)
        baseline = _load_baseline(baseline_path, cfg)
        if not baseline_path.exists():
            _save_baseline(baseline_path, baseline)
        fsm_sim = _build_state_machine_simulator(cfg, raw_config)

        context = _StudioContext(
            cfg=cfg,
            config_path=config_path,
            hz=hz,
            baseline_path=baseline_path,
            baseline=baseline,
            primitive_specs=_primitive_specs(cfg),
            style_options=_style_options(cfg),
            lamp_geometry=lamp_geometry,
            dh_params=dh_params,
            fsm_sim=fsm_sim,
            experiments_path=experiments_path,
        )
        _serve_toolbox(context, int(args.port), landing="playback", trace_path=output)
    else:
        rel = _relative_trace_path(output)
        if rel:
            print(
                "viewer (after starting any static server at repo root): "
                f"/tools/primitive_sim/web/index.html?trace={rel}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
