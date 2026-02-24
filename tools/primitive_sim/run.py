#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import partial
import http.server
from pathlib import Path
import socketserver
import sys
from typing import Any, Sequence
from urllib.parse import quote

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pala.config import load_config
from pala.types import (
    ActionPlan,
    BreathCommand,
    GazeToCommand,
    GlanceCommand,
    HoldCommand,
    HomeCommand,
    MoveToCommand,
    NodCommand,
    OrientToZoneCommand,
    PrimitiveKind,
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local primitive simulator trace and optional 3D web viewer")
    parser.add_argument("--config", default="config/robot.yaml", help="Path to robot config")
    parser.add_argument("--scenario", choices=["suite", "single", "script"], default="suite")
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
    return parser.parse_args(argv)


def _build_single_action(args: argparse.Namespace, cfg: Any) -> ActionPlan:
    primitive = PrimitiveKind(str(args.primitive).strip().lower())

    if primitive == PrimitiveKind.HOLD:
        command = HoldCommand()
    elif primitive == PrimitiveKind.HOME:
        command = HomeCommand(rate_rad_s=float(args.rate_rad_s))
    elif primitive == PrimitiveKind.BREATH:
        command = BreathCommand(
            amp_rad=float(args.amp_rad),
            period_s=float(args.period_s),
            rate_rad_s=float(args.rate_rad_s),
        )
    elif primitive == PrimitiveKind.GLANCE:
        command = GlanceCommand(
            direction=str(args.direction),
            amp_rad=float(args.amp_rad),
            duration_s=float(args.duration_s),
            rate_rad_s=float(args.rate_rad_s),
        )
    elif primitive == PrimitiveKind.NOD:
        command = NodCommand(
            amp_rad=float(args.amp_rad),
            duration_s=float(args.duration_s),
            cycles=max(1, int(args.cycles)),
            rate_rad_s=float(args.rate_rad_s),
        )
    elif primitive == PrimitiveKind.ORIENT_TO_ZONE:
        command = OrientToZoneCommand(
            zone=str(args.zone),
            amp_rad=float(args.amp_rad),
            rate_rad_s=float(args.rate_rad_s),
        )
    elif primitive == PrimitiveKind.GAZE_TO:
        command = GazeToCommand(
            yaw_rad=float(args.yaw_rad),
            pitch_rad=float(args.pitch_rad),
            rate_rad_s=float(args.rate_rad_s),
            dwell_s=float(args.dwell_s),
            timeout_s=float(args.timeout_s),
        )
    elif primitive == PrimitiveKind.MOVE_TO:
        target = _parse_move_target(args, cfg)
        command = MoveToCommand(
            target_rad=target,
            relative=bool(args.relative),
            rate_rad_s=float(args.rate_rad_s),
            timeout_s=float(args.timeout_s),
        )
    else:
        raise ValueError(f"unsupported primitive: {primitive.value}")

    return ActionPlan(
        primitive=primitive,
        command=command,
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


def _load_viewer_geometry_from_config(config_path: Path) -> dict[str, float]:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    candidates: list[Any] = []
    candidates.append(raw.get("lamp_geometry"))
    sim_viewer = raw.get("sim_viewer")
    if isinstance(sim_viewer, dict):
        candidates.append(sim_viewer.get("lamp_geometry"))
    primitive_sim = raw.get("primitive_sim")
    if isinstance(primitive_sim, dict):
        candidates.append(primitive_sim.get("lamp_geometry"))
    tools = raw.get("tools")
    if isinstance(tools, dict):
        tools_primitive_sim = tools.get("primitive_sim")
        if isinstance(tools_primitive_sim, dict):
            candidates.append(tools_primitive_sim.get("lamp_geometry"))

    section = next((c for c in candidates if isinstance(c, dict)), None)
    if not isinstance(section, dict):
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


def _relative_trace_path(path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(ROOT)
    except ValueError:
        return None
    return "/" + rel.as_posix()


def _serve(trace_path: Path, port: int) -> None:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    rel = _relative_trace_path(trace_path)
    url = f"http://127.0.0.1:{port}/tools/primitive_sim/web/index.html"
    if rel:
        url += f"?trace={quote(rel, safe='/')}"

    with _ReuseTCPServer(("127.0.0.1", int(port)), handler) as server:
        print(f"simulator viewer: {url}")
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
    lamp_geometry = _load_viewer_geometry_from_config(Path(args.config))
    if lamp_geometry:
        trace["metadata"]["lamp_geometry"] = lamp_geometry

    output = Path(args.output)
    write_trace_json(output, trace)

    sample_count = len(trace["samples"])
    summary = trace.get("summary", {})
    print(
        f"primitive_sim: scenario={args.scenario} hz={hz:.1f} segments={len(segments)} samples={sample_count}"
    )
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
        _serve(output, int(args.port))
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
