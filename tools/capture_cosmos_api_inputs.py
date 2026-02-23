#!/usr/bin/env python3
"""Capture a fixed image set for Cosmos API probing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pala.config import load_config
from pala.hardware.camera import DummyCamera


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=40, help="Number of images to capture.")
    p.add_argument("--interval-s", type=float, default=1.0, help="Seconds between captures.")
    p.add_argument("--out-dir", default="logs/cosmos_api_inputs", help="Output directory.")
    p.add_argument("--config", default="config/robot.yaml", help="Path to robot config.")
    p.add_argument(
        "--source",
        choices=["auto", "jetson", "dummy"],
        default="auto",
        help="Camera source selection.",
    )
    p.add_argument("--max-width", type=int, default=0, help="Optional resize width. 0 keeps original.")
    p.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality (1-100).")
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing images in output directory before capture.",
    )
    return p.parse_args()


def _select_camera(args: argparse.Namespace, cfg) -> tuple[Any, str]:
    source = args.source
    if source == "dummy":
        return DummyCamera(width=cfg.camera.width, height=cfg.camera.height), "dummy"

    if source == "jetson":
        from pala.hardware.camera_gst import GStreamerCamera

        return (
            GStreamerCamera(
                device=cfg.camera.device,
                width=cfg.camera.width,
                height=cfg.camera.height,
                fps=cfg.camera.fps,
                pipeline=cfg.camera.pipeline,
            ),
            "jetson",
        )

    if Path(cfg.camera.device).exists():
        try:
            from pala.hardware.camera_gst import GStreamerCamera

            camera = GStreamerCamera(
                device=cfg.camera.device,
                width=cfg.camera.width,
                height=cfg.camera.height,
                fps=cfg.camera.fps,
                pipeline=cfg.camera.pipeline,
            )
            return camera, "jetson"
        except Exception as exc:  # noqa: BLE001
            print(f"warning: failed to init jetson camera, falling back to dummy ({exc})")
    return DummyCamera(width=cfg.camera.width, height=cfg.camera.height), "dummy"


def _to_uint8_rgb(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 frame, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _save_frame(
    frame: np.ndarray,
    *,
    path: Path,
    max_width: int,
    jpeg_quality: int,
) -> tuple[list[int], list[int], int]:
    img = Image.fromarray(frame, mode="RGB")
    raw_shape = [img.height, img.width, 3]
    if max_width > 0 and img.width > max_width:
        new_h = int(round((max_width / float(img.width)) * img.height))
        img = img.resize((max_width, max(1, new_h)))
    resized_shape = [img.height, img.width, 3]
    img.save(path, format="JPEG", quality=jpeg_quality, optimize=True)
    return raw_shape, resized_shape, int(path.stat().st_size)


def _clean_output_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp", "capture_manifest.json"):
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def main() -> int:
    args = _parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be > 0")
    if args.interval_s <= 0:
        raise SystemExit("--interval-s must be > 0")
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        raise SystemExit("--jpeg-quality must be in [1, 100]")

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        _clean_output_dir(out_dir)

    camera, camera_name = _select_camera(args, cfg)
    print(f"capture source: {camera_name}")
    print(f"output dir: {out_dir}")
    print(f"capturing {args.count} images at {args.interval_s:.2f}s intervals")

    rows: list[dict[str, Any]] = []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    start_mono = time.monotonic()
    next_capture_mono = start_mono

    try:
        for idx in range(1, args.count + 1):
            now = time.monotonic()
            if now < next_capture_mono:
                time.sleep(next_capture_mono - now)

            capture_started = time.monotonic()
            frame, pts_ns, mono_ns = camera.get_frame()
            capture_ms = (time.monotonic() - capture_started) * 1000.0
            frame_u8 = _to_uint8_rgb(frame)

            wall_now = datetime.now(timezone.utc)
            filename = f"{idx:03d}_{wall_now.strftime('%Y%m%d_%H%M%S')}.jpg"
            path = out_dir / filename
            original_shape, resized_shape, file_bytes = _save_frame(
                frame_u8,
                path=path,
                max_width=max(0, int(args.max_width)),
                jpeg_quality=int(args.jpeg_quality),
            )

            row = {
                "index": idx,
                "file": filename,
                "path": str(path),
                "wall_time_utc": wall_now.isoformat(),
                "mono_ns": int(mono_ns),
                "pts_ns": None if pts_ns is None else int(pts_ns),
                "capture_ms": round(capture_ms, 2),
                "original_shape": original_shape,
                "saved_shape": resized_shape,
                "jpeg_bytes": file_bytes,
            }
            rows.append(row)
            print(f"[{idx:02d}/{args.count}] {filename} ({file_bytes} bytes)")

            next_capture_mono += float(args.interval_s)
    finally:
        camera.shutdown()

    manifest = {
        "run_id": run_id,
        "config_path": str(Path(args.config)),
        "source": camera_name,
        "count": args.count,
        "interval_s": float(args.interval_s),
        "max_width": int(args.max_width),
        "jpeg_quality": int(args.jpeg_quality),
        "output_dir": str(out_dir),
        "captured": rows,
    }
    manifest_path = out_dir / "capture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"done: wrote {len(rows)} images")
    print(f"manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
