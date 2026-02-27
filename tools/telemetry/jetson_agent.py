from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import queue
import re
import signal
import subprocess
import threading
import time
from typing import Any, Dict, Optional, Protocol, Sequence

import numpy as np
from PIL import Image

from tools.telemetry.capture import CaptureConfig, SessionCaptureWriter
from tools.telemetry.filters import FieldFilter, matches_field_filters, parse_field_filters
from tools.telemetry.packs import apply_pack_overrides, list_packs, resolve_packs
from tools.telemetry.protocol import encode_message, event
from tools.telemetry.schema_v3 import TELEMETRY_SCHEMA_VERSION_V3


class DropCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def inc(self) -> None:
        with self._lock:
            self._value += 1

    def value(self) -> int:
        with self._lock:
            return self._value


class WarningLimiter:
    """Coalesce repeated warning events to avoid flooding the stream."""

    def __init__(self, min_interval_s: float) -> None:
        self._min_interval_s = max(0.1, float(min_interval_s))
        self._lock = threading.Lock()
        self._last_emit_s: Dict[str, float] = {}
        self._suppressed: Dict[str, int] = {}

    def acquire(self, key: str) -> Optional[int]:
        now = time.monotonic()
        with self._lock:
            last = self._last_emit_s.get(key)
            if last is None or (now - last) >= self._min_interval_s:
                self._last_emit_s[key] = now
                suppressed = self._suppressed.pop(key, 0)
                return suppressed
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
            return None


class _VideoSource(Protocol):
    def get_frame(self) -> tuple[np.ndarray, Optional[int], int]:
        ...

    def shutdown(self) -> None:
        ...


class _NoFrameAvailable(RuntimeError):
    """Transient condition for pull-based sources (for example, tap file not yet written)."""


class _DummyVideoSource:
    def __init__(self, width: int, height: int) -> None:
        self._width = max(16, int(width))
        self._height = max(16, int(height))
        self._t0 = time.monotonic()

    def get_frame(self) -> tuple[np.ndarray, Optional[int], int]:
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        elapsed = time.monotonic() - self._t0

        # Subtle background gradient to make compression artifacts visible.
        xv = np.linspace(0, 255, self._width, dtype=np.uint8)
        yv = np.linspace(40, 160, self._height, dtype=np.uint8)
        frame[:, :, 0] = xv[None, :]
        frame[:, :, 1] = yv[:, None]
        frame[:, :, 2] = 30

        # Moving marker box (simulates tracked subject motion).
        cx = int((0.5 + 0.35 * math.sin(elapsed * 0.8)) * (self._width - 1))
        cy = int((0.5 + 0.2 * math.cos(elapsed * 0.5)) * (self._height - 1))
        half_w = max(10, self._width // 12)
        half_h = max(10, self._height // 10)
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(self._width - 1, cx + half_w)
        y2 = min(self._height - 1, cy + half_h)
        frame[y1:y2, x1 : x1 + 2] = (0, 255, 0)
        frame[y1:y2, x2 - 2 : x2] = (0, 255, 0)
        frame[y1 : y1 + 2, x1:x2] = (0, 255, 0)
        frame[y2 - 2 : y2, x1:x2] = (0, 255, 0)

        return frame, None, time.monotonic_ns()

    def shutdown(self) -> None:
        return None


def _as_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class _TapVideoSource:
    def __init__(self, *, jpeg_path: str, meta_path: str) -> None:
        self._jpeg_path = str(jpeg_path)
        self._meta_path = str(meta_path)
        self._cached: Optional[tuple[np.ndarray, Optional[int], int]] = None
        self._last_sig: Optional[tuple[int, int]] = None
        self.last_meta: Dict[str, Any] = {}

    def _read_meta(self) -> Dict[str, Any]:
        if not os.path.exists(self._meta_path):
            return {}
        try:
            with open(self._meta_path, "r", encoding="utf-8") as fh:
                decoded = json.load(fh)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid tap metadata JSON: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"tap metadata read failed: {exc}") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("tap metadata must be a JSON object")
        return decoded

    def get_frame(self) -> tuple[np.ndarray, Optional[int], int]:
        try:
            stat = os.stat(self._jpeg_path)
        except FileNotFoundError:
            if self._cached is not None:
                return self._cached
            raise _NoFrameAvailable("tap jpeg missing")
        except OSError as exc:
            raise RuntimeError(f"tap jpeg stat failed: {exc}") from exc

        sig = (int(stat.st_mtime_ns), int(stat.st_size))
        if self._cached is not None and sig == self._last_sig:
            return self._cached

        meta = self._read_meta()
        self.last_meta = meta
        try:
            with open(self._jpeg_path, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise RuntimeError(f"tap jpeg read failed: {exc}") from exc
        if not data:
            if self._cached is not None:
                return self._cached
            raise _NoFrameAvailable("tap jpeg is empty")

        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"tap jpeg decode failed: {exc}") from exc

        frame = np.asarray(image, dtype=np.uint8)
        pts_ns = _as_optional_int(meta.get("pts_ns"))
        mono_ns = _as_optional_int(meta.get("mono_ns"))
        if mono_ns is None:
            mono_ns = time.monotonic_ns()

        packet = (frame, pts_ns, mono_ns)
        self._cached = packet
        self._last_sig = sig
        return packet

    def shutdown(self) -> None:
        return None


def _emit(
    out_q: "queue.Queue[Dict[str, Any]]",
    drops: DropCounter,
    source: str,
    payload: Dict[str, Any],
    *,
    level: str = "info",
    msg_type: str = "event",
    ts_mono_s: Optional[float] = None,
) -> None:
    msg = event(source=source, payload=payload, level=level, msg_type=msg_type, ts_mono_s=ts_mono_s)
    try:
        out_q.put_nowait(msg)
    except queue.Full:
        drops.inc()


def _emit_warning_limited(
    *,
    out_q: "queue.Queue[Dict[str, Any]]",
    drops: DropCounter,
    source: str,
    payload: Dict[str, Any],
    limiter: WarningLimiter,
    key: str,
) -> None:
    suppressed = limiter.acquire(key)
    if suppressed is None:
        return
    out = dict(payload)
    if suppressed > 0:
        out["suppressed"] = suppressed
    _emit(out_q, drops, source, out, level="warning")


def _truncate(text: str, max_len: int = 240) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _source_enabled(source: str, enabled_sources: set[str]) -> bool:
    return source in enabled_sources


def _passes_filters(msg: Dict[str, Any], field_filters: Sequence[FieldFilter]) -> bool:
    return matches_field_filters(msg, field_filters)


def _parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    text = line.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _tail_jsonl_file(
    *,
    stop: threading.Event,
    out_q: "queue.Queue[Dict[str, Any]]",
    drops: DropCounter,
    source: str,
    path: str,
    poll_s: float,
    start_at_end: bool,
    warning_interval_s: float,
) -> None:
    fh = None
    inode_key = None
    warning_limiter = WarningLimiter(min_interval_s=warning_interval_s)

    while not stop.is_set():
        if fh is None:
            if not os.path.exists(path):
                time.sleep(poll_s)
                continue
            try:
                fh = open(path, "r", encoding="utf-8", errors="replace")
            except OSError as exc:
                _emit(out_q, drops, source, {"path": path, "error": repr(exc)}, level="warning")
                time.sleep(poll_s)
                continue
            st = os.fstat(fh.fileno())
            inode_key = (st.st_dev, st.st_ino)
            if start_at_end:
                fh.seek(0, os.SEEK_END)

        line = fh.readline()
        if line:
            parsed = _parse_json_line(line)
            if parsed is None:
                _emit_warning_limited(
                    out_q=out_q,
                    drops=drops,
                    source=source,
                    payload={
                        "path": path,
                        "raw_line": _truncate(line.strip()),
                        "parse_error": "invalid_json_line",
                    },
                    limiter=warning_limiter,
                    key=f"{source}:invalid_json_line",
                )
            else:
                _emit(out_q, drops, source, {"path": path, "data": parsed})
            continue

        # Handle rotation/recreate.
        try:
            st = os.stat(path)
        except FileNotFoundError:
            fh.close()
            fh = None
            inode_key = None
            time.sleep(poll_s)
            continue

        curr_key = (st.st_dev, st.st_ino)
        if curr_key != inode_key or st.st_size < fh.tell():
            fh.close()
            fh = None
            inode_key = None
            start_at_end = False
            continue

        time.sleep(poll_s)

    if fh is not None:
        fh.close()


_RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
_GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
_EMC_RE = re.compile(r"EMC_FREQ\s+(\d+)%")
_CPU_RE = re.compile(r"CPU\s+\[([^\]]*)\]")
_TEMP_RE = re.compile(r"([A-Za-z0-9_]+)@([0-9]+(?:\.[0-9]+)?)C")


def _parse_tegrastats(line: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"raw": line.strip()}

    ram_match = _RAM_RE.search(line)
    if ram_match:
        payload["ram_used_mb"] = int(ram_match.group(1))
        payload["ram_total_mb"] = int(ram_match.group(2))

    gr3d_match = _GR3D_RE.search(line)
    if gr3d_match:
        payload["gpu_util_pct"] = int(gr3d_match.group(1))

    emc_match = _EMC_RE.search(line)
    if emc_match:
        payload["emc_util_pct"] = int(emc_match.group(1))

    cpu_match = _CPU_RE.search(line)
    if cpu_match:
        cpu_tokens = [tok.strip() for tok in cpu_match.group(1).split(",") if tok.strip()]
        cpu_values = []
        for tok in cpu_tokens:
            if tok == "off":
                continue
            pct_match = re.match(r"(\d+)%@", tok)
            if pct_match:
                cpu_values.append(int(pct_match.group(1)))
        if cpu_values:
            payload["cpu_util_avg_pct"] = round(sum(cpu_values) / len(cpu_values), 1)
            payload["cpu_cores_online"] = len(cpu_values)

    temps = []
    for name, temp in _TEMP_RE.findall(line):
        temps.append((name, float(temp)))
    if temps:
        payload["temps_c"] = {name: value for name, value in temps}
        payload["temp_max_c"] = max(value for _, value in temps)

    return payload


def _run_tegrastats(
    *,
    stop: threading.Event,
    out_q: "queue.Queue[Dict[str, Any]]",
    drops: DropCounter,
    interval_ms: int,
    restart_delay_s: float,
    warning_interval_s: float,
) -> None:
    cmd = ["tegrastats", "--interval", str(interval_ms)]
    warning_limiter = WarningLimiter(min_interval_s=warning_interval_s)

    while not stop.is_set():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            _emit_warning_limited(
                out_q=out_q,
                drops=drops,
                source="tegrastats",
                payload={"error": "command_not_found"},
                limiter=warning_limiter,
                key="tegrastats:command_not_found",
            )
            return
        except OSError as exc:
            _emit_warning_limited(
                out_q=out_q,
                drops=drops,
                source="tegrastats",
                payload={"error": f"spawn_failed: {exc!r}"},
                limiter=warning_limiter,
                key="tegrastats:spawn_failed",
            )
            stop.wait(max(0.2, float(restart_delay_s)))
            continue

        try:
            assert proc.stdout is not None
            while not stop.is_set():
                line = proc.stdout.readline()
                if line == "":
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue
                payload = _parse_tegrastats(line)
                _emit(out_q, drops, "tegrastats", payload)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

        if stop.is_set():
            break
        _emit_warning_limited(
            out_q=out_q,
            drops=drops,
            source="tegrastats",
            payload={"error": "process_exited", "returncode": proc.returncode},
            limiter=warning_limiter,
            key="tegrastats:process_exited",
        )
        stop.wait(max(0.2, float(restart_delay_s)))


def _run_journalctl(
    *,
    stop: threading.Event,
    out_q: "queue.Queue[Dict[str, Any]]",
    drops: DropCounter,
    filter_re: re.Pattern[str],
    restart_delay_s: float,
    warning_interval_s: float,
) -> None:
    cmd = ["journalctl", "-f", "-n", "0", "-o", "cat"]
    warning_limiter = WarningLimiter(min_interval_s=warning_interval_s)

    while not stop.is_set():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            _emit_warning_limited(
                out_q=out_q,
                drops=drops,
                source="journal",
                payload={"error": "command_not_found"},
                limiter=warning_limiter,
                key="journal:command_not_found",
            )
            return
        except OSError as exc:
            _emit_warning_limited(
                out_q=out_q,
                drops=drops,
                source="journal",
                payload={"error": f"spawn_failed: {exc!r}"},
                limiter=warning_limiter,
                key="journal:spawn_failed",
            )
            stop.wait(max(0.2, float(restart_delay_s)))
            continue

        try:
            assert proc.stdout is not None
            while not stop.is_set():
                line = proc.stdout.readline()
                if line == "":
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue
                text = line.rstrip("\n")
                if not text:
                    continue
                if not filter_re.search(text):
                    continue
                _emit(out_q, drops, "journal", {"line": _truncate(text)}, level="warning")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

        if stop.is_set():
            break
        _emit_warning_limited(
            out_q=out_q,
            drops=drops,
            source="journal",
            payload={"error": "process_exited", "returncode": proc.returncode},
            limiter=warning_limiter,
            key="journal:process_exited",
        )
        stop.wait(max(0.2, float(restart_delay_s)))


def _build_video_source(args: argparse.Namespace) -> Optional[_VideoSource]:
    source = args.video_source
    if source == "off":
        return None

    if source == "tap":
        return _TapVideoSource(
            jpeg_path=args.video_tap_jpeg,
            meta_path=args.video_tap_meta,
        )

    if source == "dummy":
        return _DummyVideoSource(args.video_width, args.video_height)

    if source == "gst":
        from pala.hardware.camera_gst import GStreamerCamera

        return GStreamerCamera(
            device=args.video_device,
            width=args.video_width,
            height=args.video_height,
            fps=args.video_capture_fps,
            pipeline=args.video_pipeline,
        )

    raise ValueError(f"unknown video source: {source}")


def _resample_filter() -> int:
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.BILINEAR)
    return int(Image.BILINEAR)


def _encode_jpeg_frame(
    frame: np.ndarray,
    *,
    max_width: int,
    max_height: int,
    quality: int,
) -> tuple[bytes, int, int]:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected frame shape HxWx3, got {frame.shape!r}")

    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    max_w = max(1, int(max_width))
    max_h = max(1, int(max_height))

    if image.width > max_w or image.height > max_h:
        image.thumbnail((max_w, max_h), resample=_resample_filter())

    q = max(30, min(95, int(quality)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=q, optimize=False)
    data = buf.getvalue()
    return data, image.width, image.height


def _run_video_stream(
    *,
    stop: threading.Event,
    out_q: "queue.Queue[Dict[str, Any]]",
    drops: DropCounter,
    args: argparse.Namespace,
) -> None:
    warning_limiter = WarningLimiter(min_interval_s=max(0.2, float(args.warning_throttle_s)))

    try:
        source = _build_video_source(args)
    except Exception as exc:
        _emit_warning_limited(
            out_q=out_q,
            drops=drops,
            source="video",
            payload={"error": f"video_init_failed: {exc!r}"},
            limiter=warning_limiter,
            key="video:init_failed",
        )
        return

    if source is None:
        return

    _emit(
        out_q,
        drops,
        "video",
        {
            "status": "started",
            "video_source": args.video_source,
            "capture_size": [args.video_width, args.video_height],
            "stream_size": [args.video_max_width, args.video_max_height],
            "stream_fps": float(args.video_fps),
            "jpeg_quality": int(args.video_jpeg_quality),
            "tap_jpeg": args.video_tap_jpeg if args.video_source == "tap" else None,
            "tap_meta": args.video_tap_meta if args.video_source == "tap" else None,
        },
    )

    frame_id = 0
    last_emit_s = 0.0
    emit_period_s = 1.0 / max(0.2, float(args.video_fps))
    last_frame_seen_s = time.monotonic()

    try:
        while not stop.is_set():
            try:
                frame, pts_ns, mono_ns = source.get_frame()
            except _NoFrameAvailable:
                now_s = time.monotonic()
                if (now_s - last_frame_seen_s) >= max(0.5, float(args.video_wait_warn_s)):
                    _emit_warning_limited(
                        out_q=out_q,
                        drops=drops,
                        source="video",
                        payload={
                            "error": "video_frame_wait_timeout",
                            "video_source": args.video_source,
                            "tap_jpeg": args.video_tap_jpeg if args.video_source == "tap" else None,
                            "tap_meta": args.video_tap_meta if args.video_source == "tap" else None,
                        },
                        limiter=warning_limiter,
                        key="video:no_frame",
                    )
                stop.wait(0.05)
                continue
            except Exception as exc:
                _emit_warning_limited(
                    out_q=out_q,
                    drops=drops,
                    source="video",
                    payload={"error": f"video_capture_failed: {exc!r}"},
                    limiter=warning_limiter,
                    key="video:capture_failed",
                )
                stop.wait(0.2)
                continue
            last_frame_seen_s = time.monotonic()

            now_s = time.monotonic()
            if (now_s - last_emit_s) < emit_period_s:
                continue
            last_emit_s = now_s

            try:
                jpeg, width, height = _encode_jpeg_frame(
                    frame,
                    max_width=args.video_max_width,
                    max_height=args.video_max_height,
                    quality=args.video_jpeg_quality,
                )
            except Exception as exc:
                _emit_warning_limited(
                    out_q=out_q,
                    drops=drops,
                    source="video",
                    payload={"error": f"video_encode_failed: {exc!r}"},
                    limiter=warning_limiter,
                    key="video:encode_failed",
                )
                continue

            max_frame_bytes = int(args.video_max_bytes)
            if max_frame_bytes > 0 and len(jpeg) > max_frame_bytes:
                _emit_warning_limited(
                    out_q=out_q,
                    drops=drops,
                    source="video",
                    payload={
                        "error": "video_frame_too_large",
                        "frame_bytes": len(jpeg),
                        "max_bytes": max_frame_bytes,
                        "size": [width, height],
                    },
                    limiter=warning_limiter,
                    key="video:frame_too_large",
                )
                continue

            payload = {
                "frame_id": frame_id,
                "codec": "jpeg",
                "width": width,
                "height": height,
                "pts_ns": pts_ns,
                "bytes_b64": base64.b64encode(jpeg).decode("ascii"),
            }
            if args.video_source == "tap" and isinstance(source, _TapVideoSource):
                tap_extra = source.last_meta.get("extra")
                if isinstance(tap_extra, dict):
                    payload["tap_extra"] = tap_extra
            _emit(
                out_q,
                drops,
                "video_frame",
                payload,
                msg_type="frame",
                ts_mono_s=mono_ns / 1_000_000_000.0,
            )
            frame_id += 1
    finally:
        try:
            source.shutdown()
        finally:
            _emit(out_q, drops, "video", {"status": "stopped"})


def _heartbeat_loop(
    *,
    stop: threading.Event,
    out_q: "queue.Queue[Dict[str, Any]]",
    drops: DropCounter,
    interval_s: float,
) -> None:
    while not stop.is_set():
        _emit(
            out_q,
            drops,
            "agent",
            {
                "alive": True,
                "pid": os.getpid(),
                "dropped_events": drops.value(),
            },
        )
        stop.wait(interval_s)


def _transport_stats_loop(
    *,
    stop: threading.Event,
    out_q: "queue.Queue[Dict[str, Any]]",
    drops: DropCounter,
    interval_s: float,
) -> None:
    while not stop.is_set():
        _emit(
            out_q,
            drops,
            "transport_stats",
            {
                "queue_depth": out_q.qsize(),
                "queue_capacity": out_q.maxsize,
                "dropped_events": drops.value(),
            },
        )
        stop.wait(interval_s)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jetson telemetry sidecar agent.")
    parser.add_argument("--list-packs", action="store_true", help="List built-in signal packs and exit.")
    parser.add_argument(
        "--pack",
        action="append",
        default=[],
        help="Signal pack to enable (repeatable). Default: runtime_core.",
    )
    parser.add_argument(
        "--pack-override",
        action="append",
        default=[],
        help="Pack override key=value. Keys: include_sources, exclude_sources, add_journal, set_journal.",
    )
    parser.add_argument(
        "--field-filter",
        action="append",
        default=[],
        help="Field predicate source.path<op>value where op is one of =,!=,<,>,~.",
    )
    parser.add_argument("--perception-log", default="logs/perception.jsonl")
    parser.add_argument("--actions-log", default="logs/actions.jsonl")
    parser.add_argument("--memory-log", default="logs/orchestrator_memory.jsonl")
    parser.add_argument("--timeline-log", default="logs/orchestrator_timeline.jsonl")
    parser.add_argument("--behavior-env-log", default="logs/behavior_env.jsonl")
    parser.add_argument("--behavior-planner-log", default="logs/behavior_planner.jsonl")
    parser.add_argument("--behavior-reasoning-log", default="logs/behavior_reasoning.jsonl")
    parser.add_argument("--poll-ms", type=int, default=200)
    parser.add_argument("--from-start", action="store_true", help="Read logs from beginning instead of tailing.")
    parser.add_argument("--no-tegrastats", action="store_true")
    parser.add_argument("--tegrastats-interval-ms", type=int, default=1000)
    parser.add_argument("--no-journal", action="store_true")
    parser.add_argument(
        "--journal-filter",
        default=r"(deepstream|nvinfer|gstreamer|gst|error|timeout|engine)",
        help="Case-insensitive regex for filtering journal lines.",
    )
    parser.add_argument("--warning-throttle-s", type=float, default=2.0)
    parser.add_argument("--worker-restart-delay-s", type=float, default=1.0)
    parser.add_argument("--heartbeat-s", type=float, default=1.0)
    parser.add_argument("--queue-size", type=int, default=1024)

    parser.add_argument("--video-source", choices=["off", "dummy", "gst", "tap"], default="off")
    parser.add_argument("--video-device", default="/dev/video0")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--video-capture-fps", type=int, default=30)
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument("--video-max-width", type=int, default=640)
    parser.add_argument("--video-max-height", type=int, default=360)
    parser.add_argument("--video-jpeg-quality", type=int, default=70)
    parser.add_argument("--video-max-bytes", type=int, default=700_000)
    parser.add_argument("--video-wait-warn-s", type=float, default=5.0)
    parser.add_argument("--video-tap-jpeg", default="logs/telemetry/preview/latest.jpg")
    parser.add_argument("--video-tap-meta", default="logs/telemetry/preview/latest.json")
    parser.add_argument("--video-pipeline", default=None)
    parser.add_argument("--capture-dir", default="", help="Optional capture bundle output directory.")
    parser.add_argument("--capture-frames", choices=["off", "keyframes", "all"], default="off")
    parser.add_argument("--capture-max-seconds", type=float, default=0.0)
    parser.add_argument("--capture-manifest-version", type=int, default=TELEMETRY_SCHEMA_VERSION_V3)
    parser.add_argument("--trace-match-window-s", type=float, default=2.0)
    parser.add_argument("--trace-max-events", type=int, default=20_000)
    parser.add_argument("--index-live-every", type=int, default=0)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.list_packs:
        for pack in list_packs():
            print(f"{pack.name}: {pack.description}")
        return 0

    try:
        resolved_packs = resolve_packs(args.pack)
        resolved_packs = apply_pack_overrides(resolved_packs, args.pack_override)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        field_filters = parse_field_filters(args.field_filter)
    except ValueError as exc:
        parser.error(str(exc))

    enabled_sources = set(resolved_packs.sources)
    if args.no_tegrastats:
        enabled_sources.discard("tegrastats")
    if args.no_journal:
        enabled_sources.discard("journal")
    if args.video_source == "off":
        enabled_sources.discard("video")
        enabled_sources.discard("video_frame")

    journal_filters = list(resolved_packs.journal_filters)
    if args.journal_filter:
        journal_filters.append(str(args.journal_filter))
    if not journal_filters:
        journal_filters.append(r"(error|timeout)")
    try:
        journal_re = re.compile("|".join(f"(?:{expr})" for expr in journal_filters), re.IGNORECASE)
    except re.error as exc:
        parser.error(f"invalid combined journal filter regex: {exc}")

    poll_s = max(0.01, float(args.poll_ms) / 1000.0)
    start_at_end = not bool(args.from_start)

    out_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(128, int(args.queue_size)))
    drops = DropCounter()
    stop = threading.Event()
    seq_counter = 0
    capture_writer: Optional[SessionCaptureWriter] = None
    capture_dir = str(args.capture_dir or "").strip()
    if capture_dir:
        enabled_sources.add("capture_status")
        capture_cfg = CaptureConfig(
            directory=capture_dir,
            frames_mode=args.capture_frames,
            max_seconds=max(0.0, float(args.capture_max_seconds)),
            manifest_version=int(args.capture_manifest_version),
            trace_match_window_s=max(0.1, float(args.trace_match_window_s)),
            trace_max_events=max(128, int(args.trace_max_events)),
            index_live_every=max(0, int(args.index_live_every)),
            metadata={
                "packs": list(resolved_packs.names),
                "field_filters": list(args.field_filter or []),
            },
        )
        try:
            capture_writer = SessionCaptureWriter(capture_cfg)
        except Exception as exc:
            capture_writer = None
            _emit(
                out_q,
                drops,
                "capture_status",
                {"status": "init_failed", "error": repr(exc), "capture_dir": capture_dir},
                level="warning",
            )
        else:
            _emit(
                out_q,
                drops,
                "capture_status",
                {
                    "status": "started",
                    "capture_dir": capture_dir,
                    "frames_mode": args.capture_frames,
                    "max_seconds": float(args.capture_max_seconds),
                },
            )

    def _stop_handler(_sig, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    def _make_worker_thread(
        *,
        name: str,
        target,
        kwargs: Dict[str, Any],
    ) -> threading.Thread:
        def _runner() -> None:
            try:
                target(**kwargs)
            except Exception as exc:
                _emit(
                    out_q,
                    drops,
                    "agent",
                    {"error": "worker_crashed", "worker": name, "detail": repr(exc)},
                    level="warning",
                )

        return threading.Thread(target=_runner, name=f"telemetry-{name}", daemon=True)

    threads = [
        _make_worker_thread(
            name="perception_log",
            target=_tail_jsonl_file,
            kwargs={
                "stop": stop,
                "out_q": out_q,
                "drops": drops,
                "source": "perception_log",
                "path": args.perception_log,
                "poll_s": poll_s,
                "start_at_end": start_at_end,
                "warning_interval_s": max(0.2, float(args.warning_throttle_s)),
            },
        )
        if _source_enabled("perception_log", enabled_sources)
        else None,
        _make_worker_thread(
            name="actions_log",
            target=_tail_jsonl_file,
            kwargs={
                "stop": stop,
                "out_q": out_q,
                "drops": drops,
                "source": "actions_log",
                "path": args.actions_log,
                "poll_s": poll_s,
                "start_at_end": start_at_end,
                "warning_interval_s": max(0.2, float(args.warning_throttle_s)),
            },
        )
        if _source_enabled("actions_log", enabled_sources)
        else None,
        _make_worker_thread(
            name="memory_log",
            target=_tail_jsonl_file,
            kwargs={
                "stop": stop,
                "out_q": out_q,
                "drops": drops,
                "source": "memory_log",
                "path": args.memory_log,
                "poll_s": poll_s,
                "start_at_end": start_at_end,
                "warning_interval_s": max(0.2, float(args.warning_throttle_s)),
            },
        )
        if _source_enabled("memory_log", enabled_sources)
        else None,
        _make_worker_thread(
            name="timeline_log",
            target=_tail_jsonl_file,
            kwargs={
                "stop": stop,
                "out_q": out_q,
                "drops": drops,
                "source": "timeline_log",
                "path": args.timeline_log,
                "poll_s": poll_s,
                "start_at_end": start_at_end,
                "warning_interval_s": max(0.2, float(args.warning_throttle_s)),
            },
        )
        if _source_enabled("timeline_log", enabled_sources)
        else None,
        _make_worker_thread(
            name="behavior_env_log",
            target=_tail_jsonl_file,
            kwargs={
                "stop": stop,
                "out_q": out_q,
                "drops": drops,
                "source": "behavior_env_log",
                "path": args.behavior_env_log,
                "poll_s": poll_s,
                "start_at_end": start_at_end,
                "warning_interval_s": max(0.2, float(args.warning_throttle_s)),
            },
        )
        if _source_enabled("behavior_env_log", enabled_sources)
        else None,
        _make_worker_thread(
            name="behavior_planner_log",
            target=_tail_jsonl_file,
            kwargs={
                "stop": stop,
                "out_q": out_q,
                "drops": drops,
                "source": "behavior_planner_log",
                "path": args.behavior_planner_log,
                "poll_s": poll_s,
                "start_at_end": start_at_end,
                "warning_interval_s": max(0.2, float(args.warning_throttle_s)),
            },
        )
        if _source_enabled("behavior_planner_log", enabled_sources)
        else None,
        _make_worker_thread(
            name="behavior_reasoning_log",
            target=_tail_jsonl_file,
            kwargs={
                "stop": stop,
                "out_q": out_q,
                "drops": drops,
                "source": "behavior_reasoning_log",
                "path": args.behavior_reasoning_log,
                "poll_s": poll_s,
                "start_at_end": start_at_end,
                "warning_interval_s": max(0.2, float(args.warning_throttle_s)),
            },
        )
        if _source_enabled("behavior_reasoning_log", enabled_sources)
        else None,
        _make_worker_thread(
            name="heartbeat",
            target=_heartbeat_loop,
            kwargs={
                "stop": stop,
                "out_q": out_q,
                "drops": drops,
                "interval_s": max(0.2, float(args.heartbeat_s)),
            },
        )
        if _source_enabled("agent", enabled_sources)
        else None,
        _make_worker_thread(
            name="transport_stats",
            target=_transport_stats_loop,
            kwargs={
                "stop": stop,
                "out_q": out_q,
                "drops": drops,
                "interval_s": max(0.2, float(args.heartbeat_s)),
            },
        )
        if _source_enabled("transport_stats", enabled_sources)
        else None,
    ]
    threads = [t for t in threads if t is not None]

    if _source_enabled("tegrastats", enabled_sources):
        threads.append(
            _make_worker_thread(
                name="tegrastats",
                target=_run_tegrastats,
                kwargs={
                    "stop": stop,
                    "out_q": out_q,
                    "drops": drops,
                    "interval_ms": max(100, int(args.tegrastats_interval_ms)),
                    "restart_delay_s": max(0.2, float(args.worker_restart_delay_s)),
                    "warning_interval_s": max(0.2, float(args.warning_throttle_s)),
                },
            )
        )

    if _source_enabled("journal", enabled_sources):
        threads.append(
            _make_worker_thread(
                name="journal",
                target=_run_journalctl,
                kwargs={
                    "stop": stop,
                    "out_q": out_q,
                    "drops": drops,
                    "filter_re": journal_re,
                    "restart_delay_s": max(0.2, float(args.worker_restart_delay_s)),
                    "warning_interval_s": max(0.2, float(args.warning_throttle_s)),
                },
            )
        )

    if args.video_source != "off" and (
        _source_enabled("video", enabled_sources) or _source_enabled("video_frame", enabled_sources)
    ):
        threads.append(
            _make_worker_thread(
                name="video",
                target=_run_video_stream,
                kwargs={
                    "stop": stop,
                    "out_q": out_q,
                    "drops": drops,
                    "args": args,
                },
            )
        )

    for thread in threads:
        thread.start()

    try:
        while not stop.is_set():
            try:
                msg = out_q.get(timeout=0.2)
            except queue.Empty:
                continue
            source = msg.get("source")
            if not isinstance(source, str):
                continue
            if not _source_enabled(source, enabled_sources):
                continue
            if not _passes_filters(msg, field_filters):
                continue
            if "seq" not in msg:
                msg["seq"] = seq_counter
                seq_counter += 1
            if capture_writer is not None:
                keep_writing = capture_writer.write(msg)
                if not keep_writing:
                    capture_writer.close()
                    capture_writer = None
                    _emit(
                        out_q,
                        drops,
                        "capture_status",
                        {"status": "stopped", "reason": "max_seconds_elapsed"},
                    )
            print(encode_message(msg), flush=True)
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=1.0)
        if capture_writer is not None:
            capture_writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
