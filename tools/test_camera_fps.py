#!/usr/bin/env python3
"""Manual camera FPS/latency probe (Jetson only)."""
from __future__ import annotations

import argparse
import time
from typing import List, Optional

import numpy as np
from PIL import Image

from pala.config import load_config
from pala.hardware.camera_gst import GStreamerCamera
from pala.perception.frame_source import CameraFrameSource, ThreadedFrameSource


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
    parser.add_argument(
        "--threaded",
        action="store_true",
        help="Use ThreadedFrameSource wrapper (drops old frames)",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Save a single snapshot frame to logs/",
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

    frame_source = CameraFrameSource(camera)
    if args.threaded:
        reader = ThreadedFrameSource(frame_source)
    else:
        reader = frame_source

    mono_ns: List[int] = []
    pts_ns: List[Optional[int]] = []
    last_frame = None
    age_ms: List[float] = []

    t_start = time.monotonic()
    t_end = t_start + max(0.1, float(args.seconds))
    try:
        while time.monotonic() < t_end:
            if args.threaded:
                packet = reader.get_latest(timeout_s=0.01)
                if packet is None:
                    continue
            else:
                packet = reader.get_packet()
            last_frame = packet.frame
            mono_ns.append(packet.mono_ns)
            pts_ns.append(packet.pts_ns)
            age_ms.append((time.monotonic_ns() - packet.mono_ns) / 1_000_000.0)
    except KeyboardInterrupt:
        pass
    finally:
        if args.threaded:
            reader.shutdown()
        else:
            frame_source.shutdown()

    if len(mono_ns) < 2:
        print("camera_fps: not enough frames captured")
        return 1

    elapsed_s = max(0.001, time.monotonic() - t_start)
    fps = len(mono_ns) / elapsed_s
    if len(mono_ns) >= 2:
        intervals_ms = [(mono_ns[i] - mono_ns[i - 1]) / 1_000_000.0 for i in range(1, len(mono_ns))]
        interval_label = "consumer_interval_ms"
        print(f"frames={len(mono_ns)} elapsed_s={elapsed_s:.2f} consumer_fps={fps:.2f} {interval_label}({_stats(intervals_ms)})")
    else:
        print(f"frames={len(mono_ns)} elapsed_s={elapsed_s:.2f} consumer_fps={fps:.2f}")

    if not args.threaded:
        pts_values = [p for p in pts_ns if p is not None]
        if len(pts_values) >= 2:
            pts_intervals_ms = [
                (pts_values[i] - pts_values[i - 1]) / 1_000_000.0
                for i in range(1, len(pts_values))
            ]
            print(f"pts_interval_ms({_stats(pts_intervals_ms)})")

    if args.threaded:
        stats = reader.stats()
        print(
            "threaded_stats:"
            f" captured={stats['captured_count']}"
            f" dropped={stats['dropped_count']}"
            f" last_pts_ns={stats['last_pts_ns']}"
        )
        if age_ms:
            print(f"frame_age_ms({_stats(age_ms)})")
    elif age_ms:
        print(f"frame_age_ms({_stats(age_ms)})")

    if args.snapshot and last_frame is not None:
        import os
        from pathlib import Path

        out_dir = Path("logs")
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / "camera_snapshot.npy"
        jpg_path = out_dir / "camera_snapshot.jpg"
        np.save(npy_path, last_frame)
        Image.fromarray(last_frame).save(jpg_path, quality=90)
        print(f"saved_snapshot={npy_path}")
        print(f"saved_snapshot={jpg_path}")

    if args.save_npy and last_frame is not None:
        np.save(args.save_npy, last_frame)
        print(f"saved_frame={args.save_npy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
