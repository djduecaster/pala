#!/usr/bin/env python3
"""Manual camera FPS/latency probe (Jetson only)."""
from __future__ import annotations

import argparse
import time
from typing import List, Optional

import numpy as np

from pala.config import load_config
from pala.hardware.camera_gst import GStreamerCamera


def _stats(values: List[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return f"avg={arr.mean():.2f} p50={np.percentile(arr,50):.2f} p95={np.percentile(arr,95):.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0, help="Capture duration")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Override config mode for this run (e.g., jetson_full)",
    )
    parser.add_argument("--save-npy", type=str, default=None, help="Optional path to save last frame (.npy)")
    parser.add_argument("--force", action="store_true", help="Run even if config mode != jetson_full")
    args = parser.parse_args()

    cfg = load_config("config/robot.yaml")
    mode = args.mode or cfg.mode
    if mode != "jetson_full" and not args.force:
        print("camera_fps: config mode is not jetson_full; use --force to run anyway.")
        return 2

    camera = GStreamerCamera(
        device=cfg.camera.device,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fps=cfg.camera.fps,
        pipeline=cfg.camera.pipeline,
    )

    mono_ns: List[int] = []
    pts_ns: List[Optional[int]] = []
    last_frame = None

    t_end = time.monotonic() + max(0.1, float(args.seconds))
    try:
        while time.monotonic() < t_end:
            frame, pts, mono = camera.get_frame()
            last_frame = frame
            mono_ns.append(mono)
            pts_ns.append(pts)
    finally:
        camera.shutdown()

    if len(mono_ns) < 2:
        print("camera_fps: not enough frames captured")
        return 1

    intervals_ms = [(mono_ns[i] - mono_ns[i - 1]) / 1_000_000.0 for i in range(1, len(mono_ns))]
    fps = 1000.0 / (sum(intervals_ms) / len(intervals_ms))
    print(f"frames={len(mono_ns)} avg_fps={fps:.2f} interval_ms({_stats(intervals_ms)})")

    pts_values = [p for p in pts_ns if p is not None]
    if len(pts_values) >= 2:
        pts_intervals_ms = [
            (pts_values[i] - pts_values[i - 1]) / 1_000_000.0
            for i in range(1, len(pts_values))
        ]
        print(f"pts_interval_ms({_stats(pts_intervals_ms)})")

    if args.save_npy and last_frame is not None:
        np.save(args.save_npy, last_frame)
        print(f"saved_frame={args.save_npy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
