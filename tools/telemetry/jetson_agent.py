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
from typing import Any, Dict, Optional, Protocol

import numpy as np
from PIL import Image

from tools.telemetry.protocol import encode_message, event


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
) -> None:
    fh = None
    inode_key = None

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
                _emit(
                    out_q,
                    drops,
                    source,
                    {"path": path, "raw_line": line.strip(), "parse_error": "invalid_json_line"},
                    level="warning",
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
) -> None:
    cmd = ["tegrastats", "--interval", str(interval_ms)]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        _emit(out_q, drops, "tegrastats", {"error": "command_not_found"}, level="warning")
        return
    except OSError as exc:
        _emit(out_q, drops, "tegrastats", {"error": repr(exc)}, level="warning")
        return

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


def _run_journalctl(
    *,
    stop: threading.Event,
    out_q: "queue.Queue[Dict[str, Any]]",
    drops: DropCounter,
    filter_re: re.Pattern[str],
) -> None:
    cmd = ["journalctl", "-f", "-n", "0", "-o", "cat"]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        _emit(out_q, drops, "journal", {"error": "command_not_found"}, level="warning")
        return
    except OSError as exc:
        _emit(out_q, drops, "journal", {"error": repr(exc)}, level="warning")
        return

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
            _emit(out_q, drops, "journal", {"line": text}, level="warning")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()


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
    try:
        source = _build_video_source(args)
    except Exception as exc:
        _emit(out_q, drops, "video", {"error": f"video_init_failed: {exc!r}"}, level="warning")
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

    try:
        while not stop.is_set():
            try:
                frame, pts_ns, mono_ns = source.get_frame()
            except _NoFrameAvailable:
                stop.wait(0.05)
                continue
            except Exception as exc:
                _emit(out_q, drops, "video", {"error": f"video_capture_failed: {exc!r}"}, level="warning")
                stop.wait(0.2)
                continue

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
                _emit(out_q, drops, "video", {"error": f"video_encode_failed: {exc!r}"}, level="warning")
                continue

            payload = {
                "frame_id": frame_id,
                "codec": "jpeg",
                "width": width,
                "height": height,
                "pts_ns": pts_ns,
                "bytes_b64": base64.b64encode(jpeg).decode("ascii"),
            }
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jetson telemetry sidecar agent.")
    parser.add_argument("--perception-log", default="logs/perception.jsonl")
    parser.add_argument("--actions-log", default="logs/actions.jsonl")
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
    parser.add_argument("--heartbeat-s", type=float, default=1.0)
    parser.add_argument("--queue-size", type=int, default=4096)

    parser.add_argument("--video-source", choices=["off", "dummy", "gst", "tap"], default="off")
    parser.add_argument("--video-device", default="/dev/video0")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--video-capture-fps", type=int, default=30)
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument("--video-max-width", type=int, default=640)
    parser.add_argument("--video-max-height", type=int, default=360)
    parser.add_argument("--video-jpeg-quality", type=int, default=70)
    parser.add_argument("--video-tap-jpeg", default="logs/telemetry/preview/latest.jpg")
    parser.add_argument("--video-tap-meta", default="logs/telemetry/preview/latest.json")
    parser.add_argument("--video-pipeline", default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    poll_s = max(0.01, float(args.poll_ms) / 1000.0)
    start_at_end = not bool(args.from_start)

    out_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(128, int(args.queue_size)))
    drops = DropCounter()
    stop = threading.Event()

    def _stop_handler(_sig, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    threads = [
        threading.Thread(
            target=_tail_jsonl_file,
            kwargs={
                "stop": stop,
                "out_q": out_q,
                "drops": drops,
                "source": "perception_log",
                "path": args.perception_log,
                "poll_s": poll_s,
                "start_at_end": start_at_end,
            },
            daemon=True,
        ),
        threading.Thread(
            target=_tail_jsonl_file,
            kwargs={
                "stop": stop,
                "out_q": out_q,
                "drops": drops,
                "source": "actions_log",
                "path": args.actions_log,
                "poll_s": poll_s,
                "start_at_end": start_at_end,
            },
            daemon=True,
        ),
        threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "stop": stop,
                "out_q": out_q,
                "drops": drops,
                "interval_s": max(0.2, float(args.heartbeat_s)),
            },
            daemon=True,
        ),
    ]

    if not args.no_tegrastats:
        threads.append(
            threading.Thread(
                target=_run_tegrastats,
                kwargs={
                    "stop": stop,
                    "out_q": out_q,
                    "drops": drops,
                    "interval_ms": max(100, int(args.tegrastats_interval_ms)),
                },
                daemon=True,
            )
        )

    if not args.no_journal:
        threads.append(
            threading.Thread(
                target=_run_journalctl,
                kwargs={
                    "stop": stop,
                    "out_q": out_q,
                    "drops": drops,
                    "filter_re": re.compile(args.journal_filter, re.IGNORECASE),
                },
                daemon=True,
            )
        )

    if args.video_source != "off":
        threads.append(
            threading.Thread(
                target=_run_video_stream,
                kwargs={
                    "stop": stop,
                    "out_q": out_q,
                    "drops": drops,
                    "args": args,
                },
                daemon=True,
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
            print(encode_message(msg), flush=True)
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=1.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
