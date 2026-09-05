from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_joint_pairs(command: Dict[str, Any]) -> List[Tuple[str, float]]:
    names = command.get("joint_names")
    angles = command.get("joint_angles_rad")
    if not isinstance(names, list) or not isinstance(angles, list):
        return []

    count = min(len(names), len(angles))
    pairs: List[Tuple[str, float]] = []
    for i in range(count):
        pairs.append((str(names[i]), _as_float(angles[i])))
    return pairs


def _lookup_angle(joints: Dict[str, float], name: str, default: float = 0.0) -> float:
    if name in joints:
        return joints[name]
    return default


def _draw_gauge(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    w: int,
    label: str,
    value: float,
    vmin: float = -1.57,
    vmax: float = 1.57,
) -> None:
    draw.text((x, y), f"{label}: {value:+.2f}", fill=(220, 220, 220))
    bar_y = y + 14
    draw.rectangle((x, bar_y, x + w, bar_y + 8), outline=(80, 80, 80), fill=(20, 20, 20))
    span = max(1e-6, vmax - vmin)
    t = max(0.0, min(1.0, (value - vmin) / span))
    px = x + int(t * w)
    draw.rectangle((x + 1, bar_y + 1, px, bar_y + 7), fill=(60, 150, 255))


def draw_lamp_panel(
    *,
    height: int,
    width: int = 260,
    command: Dict[str, Any] | None = None,
) -> Image.Image:
    panel_h = max(120, int(height))
    panel_w = max(180, int(width))
    image = Image.new("RGB", (panel_w, panel_h), (17, 20, 24))
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=(70, 70, 70))
    draw.text((10, 8), "Commanded Joint View", fill=(245, 245, 245))

    if not isinstance(command, dict):
        draw.text((10, 30), "No command data", fill=(200, 140, 120))
        return image

    pairs = _extract_joint_pairs(command)
    joints = {name: angle for name, angle in pairs}
    enable_raw = command.get("enable")
    enable_text = str(bool(enable_raw)) if isinstance(enable_raw, bool) else "unavailable"
    enable_color = (130, 220, 140) if enable_raw is True else (255, 120, 120) if enable_raw is False else (190, 170, 150)
    draw.text((10, 30), f"commanded enable={enable_text}", fill=enable_color)
    draw.text((10, 104), "applied/deadman: unavailable", fill=(190, 170, 150))

    yaw = _lookup_angle(joints, "yaw")
    roll = _lookup_angle(joints, "roll")
    _draw_gauge(draw, x=10, y=48, w=panel_w - 20, label="yaw", value=yaw)
    _draw_gauge(draw, x=10, y=74, w=panel_w - 20, label="roll", value=roll)

    # Side-view chain for the three pitch joints.
    p1 = _lookup_angle(joints, "pitch1")
    p2 = _lookup_angle(joints, "pitch2")
    p3 = _lookup_angle(joints, "pitch3")
    chain = [("pitch1", p1), ("pitch2", p2), ("pitch3", p3)]

    base_x = panel_w // 2 + 12
    base_y = panel_h - 24
    seg_lens = [max(26, panel_h // 6), max(24, panel_h // 7), max(22, panel_h // 8)]

    # Base stand.
    draw.rectangle((base_x - 26, base_y - 10, base_x + 8, base_y + 8), fill=(90, 90, 100), outline=(150, 150, 160))

    x = float(base_x - 8)
    y = float(base_y - 10)
    theta = -math.pi / 2.0
    for idx, (name, angle) in enumerate(chain):
        theta += angle
        seg_len = seg_lens[min(idx, len(seg_lens) - 1)]
        x2 = x + math.cos(theta) * seg_len
        y2 = y + math.sin(theta) * seg_len
        draw.line((x, y, x2, y2), fill=(235, 235, 235), width=5)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(70, 130, 255))
        draw.text((10, panel_h - 52 + idx * 14), f"{name}={angle:+.2f}", fill=(210, 210, 210))
        x, y = x2, y2

    # End-effector marker with roll cue.
    head_r = 9
    draw.ellipse((x - head_r, y - head_r, x + head_r, y + head_r), outline=(255, 220, 80), width=2)
    roll_dx = math.cos(roll) * 8
    roll_dy = math.sin(roll) * 8
    draw.line((x - roll_dx, y - roll_dy, x + roll_dx, y + roll_dy), fill=(255, 220, 80), width=2)

    return image
