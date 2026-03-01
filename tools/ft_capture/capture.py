from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from pala.config import load_config
from pala.hardware.camera import CameraInterface, DummyCamera

from .catalog import ScenarioDefinition
from .storage import (
    create_take_layout,
    ensure_session_dir,
    next_take_id,
    update_session_manifest_take,
    write_catalog_snapshot,
    write_initial_label,
    write_take_manifest,
)


@dataclass(frozen=True)
class CaptureSettings:
    out_root: str
    session_id: str
    catalog_path: str
    scenario: ScenarioDefinition
    takes: int
    countdown_s: float
    duration_s: float
    sample_fps: float
    camera_source: str
    camera_device: str
    width: int
    height: int
    capture_fps: int
    camera_pipeline: Optional[str]
    jpeg_quality: int


@dataclass(frozen=True)
class CapturedFrame:
    index: int
    file_name: str
    mono_ns: int
    pts_ns: Optional[int]
    captured_wall_s: float
    rel_s: float


@dataclass(frozen=True)
class TakeResult:
    take_id: str
    take_dir: str
    clip_path: str
    frame_count: int
    duration_s: float
    sample_frame_count: int


class _GstCameraAdapter(CameraInterface):
    def __init__(
        self,
        *,
        device: str,
        width: int,
        height: int,
        fps: int,
        pipeline: Optional[str],
    ) -> None:
        from pala.hardware.camera_gst import GStreamerCamera

        self._inner = GStreamerCamera(
            device=device,
            width=width,
            height=height,
            fps=fps,
            pipeline=pipeline,
        )

    def get_frame(self) -> Tuple[np.ndarray, Optional[int], int]:
        return self._inner.get_frame()

    def shutdown(self) -> None:
        self._inner.shutdown()



def _as_uint8_rgb(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 frame, got {arr.shape!r}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr



def _save_frame(path: Path, frame: np.ndarray, *, jpeg_quality: int) -> None:
    image = Image.fromarray(frame, mode="RGB")
    image.save(path, format="JPEG", quality=max(30, min(95, int(jpeg_quality))), optimize=False)



def _assert_ffmpeg() -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg to generate clip.mp4")



def _run_ffmpeg(
    *,
    raw_frames_dir: Path,
    clip_path: Path,
    frame_rate: float,
) -> str:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{max(1.0, float(frame_rate)):.3f}",
        "-i",
        str(raw_frames_dir / "frame_%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(clip_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "ffmpeg failed").strip()
        raise RuntimeError(f"ffmpeg failed: {detail}")
    return " ".join(cmd)



def _sample_frame_indices(frames: Sequence[CapturedFrame], *, duration_s: float, sample_fps: float) -> List[int]:
    if not frames:
        return []
    spacing = 1.0 / max(0.01, float(sample_fps))
    target = 0.0
    picked: List[int] = []
    cursor = 0

    while target <= max(0.0, duration_s):
        while cursor < len(frames) and frames[cursor].rel_s < target:
            cursor += 1
        if cursor >= len(frames):
            picked.append(len(frames) - 1)
        else:
            picked.append(cursor)
        target += spacing

    deduped: List[int] = []
    seen = set()
    for idx in picked:
        if idx in seen:
            continue
        deduped.append(idx)
        seen.add(idx)
    return deduped



def _copy_sampled_frames(
    *,
    frames: Sequence[CapturedFrame],
    sampled_indices: Sequence[int],
    raw_dir: Path,
    sampled_dir: Path,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for dst_i, src_i in enumerate(sampled_indices, start=1):
        frame = frames[src_i]
        src = raw_dir / frame.file_name
        dst_name = f"frame_{dst_i:04d}.jpg"
        dst = sampled_dir / dst_name
        shutil.copyfile(src, dst)
        out.append(
            {
                "index": dst_i,
                "source_frame_index": frame.index,
                "source_file": frame.file_name,
                "file": dst_name,
                "rel_s": round(frame.rel_s, 3),
            }
        )
    return out



def _effective_fps(frames: Sequence[CapturedFrame], *, default_fps: float) -> float:
    if len(frames) < 2:
        return max(1.0, default_fps)
    span = max(1e-6, frames[-1].rel_s - frames[0].rel_s)
    return max(1.0, (len(frames) - 1) / span)



def _build_camera(
    *,
    source: str,
    device: str,
    width: int,
    height: int,
    fps: int,
    pipeline: Optional[str],
) -> Tuple[CameraInterface, str]:
    token = str(source).strip().lower()
    if token not in {"auto", "gst", "dummy"}:
        raise ValueError("camera_source must be one of auto|gst|dummy")

    if token == "dummy":
        return DummyCamera(width=width, height=height), "dummy"

    if token == "gst":
        camera = _GstCameraAdapter(
            device=device,
            width=width,
            height=height,
            fps=fps,
            pipeline=pipeline,
        )
        return camera, "gst"

    # auto
    if Path(device).exists():
        try:
            camera = _GstCameraAdapter(
                device=device,
                width=width,
                height=height,
                fps=fps,
                pipeline=pipeline,
            )
            return camera, "gst"
        except Exception:
            pass
    return DummyCamera(width=width, height=height), "dummy"



def _run_countdown(seconds: float) -> None:
    total = max(0, int(math.ceil(seconds)))
    if total <= 0:
        return
    print(f"countdown: {total}s")
    for remaining in range(total, 0, -1):
        print(f"  {remaining}...")
        time.sleep(1.0)
    print("recording start")



def _capture_frames(
    camera: CameraInterface,
    *,
    raw_dir: Path,
    duration_s: float,
    jpeg_quality: int,
) -> List[CapturedFrame]:
    frames: List[CapturedFrame] = []
    t0 = time.monotonic()
    index = 0
    while True:
        now = time.monotonic()
        rel_s = now - t0
        if rel_s >= duration_s:
            break

        frame, pts_ns, mono_ns = camera.get_frame()
        frame_u8 = _as_uint8_rgb(frame)
        index += 1
        file_name = f"frame_{index:06d}.jpg"
        _save_frame(raw_dir / file_name, frame_u8, jpeg_quality=jpeg_quality)
        frames.append(
            CapturedFrame(
                index=index,
                file_name=file_name,
                mono_ns=int(mono_ns),
                pts_ns=None if pts_ns is None else int(pts_ns),
                captured_wall_s=time.time(),
                rel_s=max(0.0, time.monotonic() - t0),
            )
        )
    return frames



def capture_scenario_takes(settings: CaptureSettings) -> List[TakeResult]:
    _assert_ffmpeg()

    session_dir = ensure_session_dir(
        out_root=settings.out_root,
        session_id=settings.session_id,
        catalog_path=settings.catalog_path,
    )
    write_catalog_snapshot(session_dir, source_catalog_path=settings.catalog_path)

    camera, camera_backend = _build_camera(
        source=settings.camera_source,
        device=settings.camera_device,
        width=settings.width,
        height=settings.height,
        fps=settings.capture_fps,
        pipeline=settings.camera_pipeline,
    )

    print(
        f"capture session={settings.session_id} scenario={settings.scenario.scenario_id} "
        f"camera={camera_backend} takes={settings.takes}"
    )

    results: List[TakeResult] = []
    try:
        for take_idx in range(1, max(1, int(settings.takes)) + 1):
            take_id = next_take_id(session_dir, scenario_id=settings.scenario.scenario_id)
            take_dir = create_take_layout(
                session_dir,
                scenario_id=settings.scenario.scenario_id,
                take_id=take_id,
            )

            print(f"take {take_idx}/{settings.takes}: {take_id}")
            _run_countdown(settings.countdown_s)

            raw_dir = take_dir / "raw_frames"
            sampled_dir = take_dir / "frames_1fps"
            clip_path = take_dir / "clip.mp4"

            capture_started_wall = time.time()
            capture_started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            frames = _capture_frames(
                camera,
                raw_dir=raw_dir,
                duration_s=settings.duration_s,
                jpeg_quality=settings.jpeg_quality,
            )
            capture_ended_wall = time.time()
            capture_ended_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

            if not frames:
                raise RuntimeError("no frames captured")

            effective_fps = _effective_fps(frames, default_fps=float(settings.capture_fps))
            ffmpeg_cmd = _run_ffmpeg(raw_frames_dir=raw_dir, clip_path=clip_path, frame_rate=effective_fps)

            sampled_indices = _sample_frame_indices(
                frames,
                duration_s=float(settings.duration_s),
                sample_fps=float(settings.sample_fps),
            )
            sampled_rows = _copy_sampled_frames(
                frames=frames,
                sampled_indices=sampled_indices,
                raw_dir=raw_dir,
                sampled_dir=sampled_dir,
            )

            manifest = {
                "schema_version": 1,
                "created_at_utc": capture_started_utc,
                "updated_at_utc": capture_ended_utc,
                "session_id": settings.session_id,
                "scenario_id": settings.scenario.scenario_id,
                "scenario_title": settings.scenario.title,
                "scenario_description": settings.scenario.description,
                "scenario_tags": list(settings.scenario.tags),
                "take_id": take_id,
                "capture": {
                    "camera_source_requested": settings.camera_source,
                    "camera_source_effective": camera_backend,
                    "camera_device": settings.camera_device,
                    "width": int(settings.width),
                    "height": int(settings.height),
                    "capture_fps_requested": int(settings.capture_fps),
                    "effective_fps": round(float(effective_fps), 3),
                    "countdown_s": float(settings.countdown_s),
                    "duration_s": float(settings.duration_s),
                    "sample_fps": float(settings.sample_fps),
                    "jpeg_quality": int(settings.jpeg_quality),
                },
                "timing": {
                    "capture_started_wall_s": capture_started_wall,
                    "capture_ended_wall_s": capture_ended_wall,
                    "capture_elapsed_s": max(0.0, capture_ended_wall - capture_started_wall),
                },
                "files": {
                    "clip_path": "clip.mp4",
                    "raw_frames_dir": "raw_frames",
                    "sampled_frames_dir": "frames_1fps",
                    "label_path": "label.json",
                },
                "ffmpeg": {
                    "command": ffmpeg_cmd,
                },
                "frame_index": [
                    {
                        "index": item.index,
                        "file": item.file_name,
                        "mono_ns": item.mono_ns,
                        "pts_ns": item.pts_ns,
                        "captured_wall_s": item.captured_wall_s,
                        "rel_s": round(item.rel_s, 4),
                    }
                    for item in frames
                ],
                "sampled_frames": sampled_rows,
            }
            write_take_manifest(take_dir, manifest)
            write_initial_label(take_dir, label_template=settings.scenario.label_template)

            rel_take_dir = str(take_dir.relative_to(session_dir))
            update_session_manifest_take(
                session_dir,
                scenario_id=settings.scenario.scenario_id,
                take_id=take_id,
                take_rel_dir=rel_take_dir,
            )

            results.append(
                TakeResult(
                    take_id=take_id,
                    take_dir=str(take_dir),
                    clip_path=str(clip_path),
                    frame_count=len(frames),
                    duration_s=max(0.0, capture_ended_wall - capture_started_wall),
                    sample_frame_count=len(sampled_rows),
                )
            )
            print(
                f"  saved {take_id}: frames={len(frames)} sampled={len(sampled_rows)} clip={clip_path.name}"
            )
    finally:
        camera.shutdown()

    return results



def build_capture_settings(
    *,
    out_root: str,
    session_id: str,
    catalog_path: str,
    scenario: ScenarioDefinition,
    takes: int,
    countdown_s: Optional[float],
    duration_s: Optional[float],
    sample_fps: Optional[float],
    camera_source: str,
    camera_device: Optional[str],
    width: Optional[int],
    height: Optional[int],
    capture_fps: Optional[int],
    camera_pipeline: Optional[str],
    jpeg_quality: int,
    runtime_config_path: str,
) -> CaptureSettings:
    cfg = load_config(runtime_config_path)

    effective_countdown = scenario.countdown_s if countdown_s is None else float(countdown_s)
    effective_duration = scenario.duration_s if duration_s is None else float(duration_s)
    effective_sample_fps = scenario.sample_fps if sample_fps is None else float(sample_fps)

    return CaptureSettings(
        out_root=out_root,
        session_id=session_id,
        catalog_path=catalog_path,
        scenario=scenario,
        takes=max(1, int(takes)),
        countdown_s=max(0.0, float(effective_countdown)),
        duration_s=max(0.1, float(effective_duration)),
        sample_fps=max(0.1, float(effective_sample_fps)),
        camera_source=str(camera_source).strip().lower(),
        camera_device=str(camera_device or cfg.camera.device),
        width=max(16, int(width or cfg.camera.width)),
        height=max(16, int(height or cfg.camera.height)),
        capture_fps=max(1, int(capture_fps or cfg.camera.fps)),
        camera_pipeline=str(camera_pipeline) if camera_pipeline else cfg.camera.pipeline,
        jpeg_quality=max(30, min(95, int(jpeg_quality))),
    )
