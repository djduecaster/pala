from __future__ import annotations

import argparse
import base64
import collections
import io
import queue
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

from PIL import Image, ImageDraw

try:
    from PIL import ImageTk
except Exception:  # pragma: no cover - depends on Tk availability.
    ImageTk = None

try:
    import tkinter as tk
except Exception:  # pragma: no cover - depends on Tk availability.
    tk = None

from tools.telemetry.protocol import decode_message


def _resample_filter() -> int:
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.BILINEAR)
    return int(Image.BILINEAR)


def _fmt_age(now_s: float, ts_s: Optional[float]) -> str:
    if ts_s is None:
        return "n/a"
    age = max(0.0, now_s - ts_s)
    return f"{age:.1f}s"


def _shorten(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _normalize_jetson_dir(path: str) -> tuple[str, Optional[str]]:
    raw = (path or "").strip()
    if not raw:
        return "~/pala", "jetson_dir was empty; defaulted to ~/pala"
    if raw.startswith("~/"):
        return raw, None
    if raw == "~":
        return "~/pala", "jetson_dir '~' mapped to ~/pala"
    if raw.startswith("/Users/"):
        base = raw.rstrip("/").split("/")[-1]
        if base:
            mapped = f"~/{base}"
            return mapped, f"jetson_dir mapped from local path to {mapped}"
    return raw, None


@dataclass
class DashboardState:
    host: str
    started_wall_s: float = field(default_factory=time.time)
    connected: bool = False
    last_event_wall_s: Optional[float] = None
    connection_note: str = "starting"
    event_counts: Dict[str, int] = field(default_factory=dict)
    perception: Optional[Dict[str, Any]] = None
    action: Optional[Dict[str, Any]] = None
    tegrastats: Optional[Dict[str, Any]] = None
    agent: Optional[Dict[str, Any]] = None
    logs: Deque[str] = field(default_factory=lambda: collections.deque(maxlen=12))
    warnings: Deque[str] = field(default_factory=lambda: collections.deque(maxlen=8))
    dropped_events_reported: int = 0

    video_frame_bytes: Optional[bytes] = None
    video_frame_meta: Optional[Dict[str, Any]] = None
    video_frames_received: int = 0
    video_decode_errors: int = 0
    last_video_wall_s: Optional[float] = None

    def apply(self, msg: Dict[str, Any]) -> None:
        source = str(msg.get("source", "unknown"))
        self.last_event_wall_s = time.time()
        self.event_counts[source] = self.event_counts.get(source, 0) + 1

        payload = msg.get("payload", {})
        if not isinstance(payload, dict):
            return

        if source == "perception_log":
            data = payload.get("data")
            if isinstance(data, dict):
                self.perception = data
            return

        if source == "actions_log":
            data = payload.get("data")
            if isinstance(data, dict):
                self.action = data
            return

        if source == "tegrastats":
            self.tegrastats = payload
            return

        if source == "agent":
            self.agent = payload
            dropped = payload.get("dropped_events")
            if isinstance(dropped, int):
                self.dropped_events_reported = dropped
            return

        if source == "video":
            status = payload.get("status")
            if isinstance(status, str):
                self.logs.append(f"video: {status}")
            if "error" in payload:
                self.warnings.append(f"video: {payload['error']}")
            return

        if source == "video_frame":
            frame_b64 = payload.get("bytes_b64")
            if not isinstance(frame_b64, str):
                self.video_decode_errors += 1
                self.warnings.append("video_frame: missing bytes_b64")
                return
            try:
                frame_bytes = base64.b64decode(frame_b64, validate=True)
            except Exception:
                self.video_decode_errors += 1
                self.warnings.append("video_frame: invalid base64")
                return
            meta = dict(payload)
            meta.pop("bytes_b64", None)
            self.video_frame_bytes = frame_bytes
            self.video_frame_meta = meta
            self.video_frames_received += 1
            self.last_video_wall_s = time.time()
            return

        if source == "journal":
            line = payload.get("line")
            if isinstance(line, str):
                self.logs.append(line)
            return

        if "error" in payload:
            self.warnings.append(f"{source}: {payload['error']}")


@dataclass
class _VideoWindow:
    root: Any
    container: Any
    label: Any
    photo: Any = None
    last_frame_id: int = -1


def _build_remote_agent_command(args: argparse.Namespace) -> str:
    agent_args = [
        "uv",
        "run",
        "python",
        "-m",
        "tools.telemetry.jetson_agent",
        "--perception-log",
        str(args.perception_log),
        "--actions-log",
        str(args.actions_log),
        "--poll-ms",
        str(int(args.poll_ms)),
        "--heartbeat-s",
        str(float(args.heartbeat_s)),
        "--queue-size",
        str(int(args.queue_size)),
    ]

    if args.from_start:
        agent_args.append("--from-start")
    if args.no_tegrastats:
        agent_args.append("--no-tegrastats")
    else:
        agent_args.extend(["--tegrastats-interval-ms", str(int(args.tegrastats_interval_ms))])
    if args.no_journal:
        agent_args.append("--no-journal")
    else:
        agent_args.extend(["--journal-filter", str(args.journal_filter)])

    if args.no_video:
        agent_args.extend(["--video-source", "off"])
    else:
        agent_args.extend(
            [
                "--video-source",
                str(args.video_source),
                "--video-device",
                str(args.video_device),
                "--video-width",
                str(int(args.video_width)),
                "--video-height",
                str(int(args.video_height)),
                "--video-capture-fps",
                str(int(args.video_capture_fps)),
                "--video-fps",
                str(float(args.video_fps)),
                "--video-max-width",
                str(int(args.video_max_width)),
                "--video-max-height",
                str(int(args.video_max_height)),
                "--video-jpeg-quality",
                str(int(args.video_jpeg_quality)),
                "--video-tap-jpeg",
                str(args.video_tap_jpeg),
                "--video-tap-meta",
                str(args.video_tap_meta),
            ]
        )
        if args.video_pipeline:
            agent_args.extend(["--video-pipeline", str(args.video_pipeline)])

    agent_cmd = " ".join(shlex.quote(part) for part in agent_args)

    parts = [
        f"PALA_TELEMETRY_JETSON_DIR={shlex.quote(args.jetson_dir)}",
        'if [ "$PALA_TELEMETRY_JETSON_DIR" = "~" ]; then PALA_TELEMETRY_JETSON_DIR="$HOME"; fi',
        'if [ "${PALA_TELEMETRY_JETSON_DIR#\\~/}" != "$PALA_TELEMETRY_JETSON_DIR" ]; then '
        'PALA_TELEMETRY_JETSON_DIR="$HOME/${PALA_TELEMETRY_JETSON_DIR#\\~/}"; fi',
        'if [ ! -d "$PALA_TELEMETRY_JETSON_DIR" ]; then '
        'echo "telemetry_agent_error: jetson dir not found: $PALA_TELEMETRY_JETSON_DIR"; exit 1; fi',
        'cd "$PALA_TELEMETRY_JETSON_DIR"',
        'export PATH="$HOME/.local/bin:$PATH"',
        'if [ -f "$HOME/.config/pala/env.sh" ]; then source "$HOME/.config/pala/env.sh"; fi',
        agent_cmd,
    ]

    remote_script = " && ".join(parts)
    return f"bash -lc {shlex.quote(remote_script)}"


def _start_ssh_agent(args: argparse.Namespace) -> subprocess.Popen[str]:
    remote_cmd = _build_remote_agent_command(args)
    return subprocess.Popen(
        ["ssh", "-T", args.jetson_host, remote_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _reader_loop(
    *,
    stop: threading.Event,
    proc: subprocess.Popen[str],
    out_q: "queue.Queue[Dict[str, Any]]",
) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        if stop.is_set():
            break
        msg = decode_message(line)
        if msg is not None:
            try:
                out_q.put_nowait(msg)
            except queue.Full:
                continue
        else:
            text = line.strip()
            if text:
                out_q.put(
                    {
                        "type": "event",
                        "source": "viewer_note",
                        "payload": {"line": text},
                    }
                )


def _init_video_window(args: argparse.Namespace, state: DashboardState) -> Optional[_VideoWindow]:
    if args.no_video or args.no_video_window:
        return None
    if tk is None or ImageTk is None:
        state.warnings.append("video_window: Tk unavailable; use --no-video-window")
        return None

    try:
        root = tk.Tk()
        root.title(f"PALA Video ({args.jetson_host})")
        root.resizable(True, True)
        init_scale = max(0.2, float(args.video_window_scale))
        init_w = max(160, int(max(1, int(args.video_max_width)) * init_scale))
        init_h = max(120, int(max(1, int(args.video_max_height)) * init_scale))
        root.geometry(f"{init_w}x{init_h}")
        root.minsize(160, 120)

        # Prevent image requested size from forcing window geometry updates.
        container = tk.Frame(root, width=init_w, height=init_h, bg="black")
        container.pack(fill="both", expand=True)
        container.pack_propagate(False)

        label = tk.Label(container, bg="black")
        label.pack(fill="both", expand=True)
    except Exception as exc:
        message = str(exc)
        if "init.tcl" in message:
            state.warnings.append(
                "video_window unavailable (Tk/Tcl missing/mismatched); use --no-video-window "
                "or recreate .venv with Homebrew Python."
            )
        elif exc.__class__.__name__ == "TclError":
            state.warnings.append(f"video_window_init_failed_tcl: {_shorten(message, 160)}")
        else:
            state.warnings.append(f"video_window_init_failed: {exc.__class__.__name__}")
        return None

    return _VideoWindow(root=root, container=container, label=label)


def _overlay_image(image: Image.Image, state: DashboardState) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Top status panel.
    draw.rectangle((0, 0, width, 58), fill=(0, 0, 0))

    primitive = None
    confidence = None
    explanation = None
    if state.action:
        primitive = state.action.get("primitive")
        confidence = state.action.get("confidence")
        explanation = state.action.get("explanation")

    p_fps = None
    p_latency = None
    zone = None
    primary_person = None
    person_conf = None
    if state.perception:
        p_fps = state.perception.get("fps")
        p_latency = state.perception.get("latency_ms")
        person_conf = state.perception.get("primary_person_conf")
        primary_person = state.perception.get("primary_person")
        debug = state.perception.get("debug")
        if isinstance(debug, dict):
            zone = debug.get("zone_hint")

    gpu = None
    cpu = None
    if state.tegrastats:
        gpu = state.tegrastats.get("gpu_util_pct")
        cpu = state.tegrastats.get("cpu_util_avg_pct")

    draw.text((8, 6), f"primitive={primitive} conf={confidence}", fill=(255, 255, 255))
    draw.text(
        (8, 23),
        f"perception_fps={p_fps} latency_ms={p_latency} zone={zone} gpu={gpu}% cpu={cpu}%",
        fill=(255, 255, 255),
    )
    if explanation is not None:
        draw.text((8, 40), _shorten(str(explanation), 90), fill=(220, 220, 220))

    if isinstance(primary_person, dict):
        try:
            cx = float(primary_person.get("cx", 0.5))
            cy = float(primary_person.get("cy", 0.5))
            bw = float(primary_person.get("w", 0.2))
            bh = float(primary_person.get("h", 0.4))

            x1 = int((cx - bw * 0.5) * width)
            y1 = int((cy - bh * 0.5) * height)
            x2 = int((cx + bw * 0.5) * width)
            y2 = int((cy + bh * 0.5) * height)

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            if x2 > x1 and y2 > y1:
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
                label = "person"
                if person_conf is not None:
                    label = f"person {person_conf:.2f}"
                text_y = max(60, y1 - 14)
                draw.text((x1 + 3, text_y), label, fill=(0, 255, 0))
        except Exception:
            pass

    return image


def _fit_image_to_window(image: Image.Image, *, target_w: int, target_h: int) -> Image.Image:
    if target_w <= 1 or target_h <= 1:
        return image
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        return image

    scale = min(float(target_w) / float(src_w), float(target_h) / float(src_h))
    out_w = max(1, int(src_w * scale))
    out_h = max(1, int(src_h * scale))
    resized = image.resize((out_w, out_h), resample=_resample_filter())
    if out_w == target_w and out_h == target_h:
        return resized

    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    x = max(0, (target_w - out_w) // 2)
    y = max(0, (target_h - out_h) // 2)
    canvas.paste(resized, (x, y))
    return canvas


def _pump_video_window(
    window: Optional[_VideoWindow],
    state: DashboardState,
    args: argparse.Namespace,
) -> Optional[_VideoWindow]:
    if window is None:
        return None

    try:
        window.root.update_idletasks()
        window.root.update()
    except Exception:
        return None

    meta = state.video_frame_meta
    frame_bytes = state.video_frame_bytes
    if meta is None or frame_bytes is None:
        return window

    frame_id = meta.get("frame_id")
    if isinstance(frame_id, int) and frame_id == window.last_frame_id:
        return window

    try:
        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    except Exception as exc:
        state.video_decode_errors += 1
        state.warnings.append(f"video_render_decode_failed: {exc!r}")
        return window

    image = _overlay_image(image, state)

    scale = max(0.2, float(args.video_window_scale))
    if abs(scale - 1.0) > 1e-3:
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            resample=_resample_filter(),
        )

    # Keep window freely resizable: render into current label dimensions.
    target_w = int(window.label.winfo_width() or 0)
    target_h = int(window.label.winfo_height() or 0)
    image = _fit_image_to_window(image, target_w=target_w, target_h=target_h)

    try:
        photo = ImageTk.PhotoImage(image)
        window.label.configure(image=photo)
        window.label.image = photo
        window.photo = photo
    except Exception as exc:
        state.video_decode_errors += 1
        state.warnings.append(f"video_window_update_failed: {exc!r}")
        return window

    if isinstance(frame_id, int):
        window.last_frame_id = frame_id
    else:
        window.last_frame_id += 1

    return window


def _render(state: DashboardState, *, now_wall_s: float, args: argparse.Namespace) -> str:
    lines = []
    uptime = now_wall_s - state.started_wall_s
    lines.append(
        f"PALA Telemetry Phase 1 | host={state.host} | connected={state.connected} | uptime={uptime:.0f}s"
    )
    lines.append(
        f"Last event age={_fmt_age(now_wall_s, state.last_event_wall_s)} | dropped(agent)={state.dropped_events_reported}"
    )
    lines.append(f"Connection note: {state.connection_note}")
    lines.append("")

    lines.append("Video")
    if args.no_video:
        lines.append("  disabled")
    elif state.video_frame_meta is None:
        lines.append("  waiting for frames")
    else:
        meta = state.video_frame_meta
        lines.append(
            "  "
            f"frame_id={meta.get('frame_id')} size={meta.get('width')}x{meta.get('height')} "
            f"age={_fmt_age(now_wall_s, state.last_video_wall_s)} received={state.video_frames_received} "
            f"decode_errors={state.video_decode_errors}"
        )

    lines.append("")
    lines.append("Perception")
    if state.perception is None:
        lines.append("  no data yet")
    else:
        zone = None
        debug = state.perception.get("debug")
        if isinstance(debug, dict):
            zone = debug.get("zone_hint")
        person_conf = state.perception.get("primary_person_conf")
        fps = state.perception.get("fps")
        latency = state.perception.get("latency_ms")
        lines.append(
            f"  fps={fps} latency_ms={latency} zone={zone} person_conf={person_conf}"
        )

    lines.append("")
    lines.append("Action")
    if state.action is None:
        lines.append("  no data yet")
    else:
        primitive = state.action.get("primitive")
        confidence = state.action.get("confidence")
        explanation = state.action.get("explanation")
        lines.append(f"  primitive={primitive} confidence={confidence}")
        lines.append(f"  explanation={_shorten(str(explanation), 120)}")

    lines.append("")
    lines.append("System (tegrastats)")
    if state.tegrastats is None:
        lines.append("  no data yet")
    else:
        gpu = state.tegrastats.get("gpu_util_pct")
        cpu = state.tegrastats.get("cpu_util_avg_pct")
        ram_used = state.tegrastats.get("ram_used_mb")
        ram_total = state.tegrastats.get("ram_total_mb")
        temp_max = state.tegrastats.get("temp_max_c")
        lines.append(
            f"  cpu_avg_pct={cpu} gpu_pct={gpu} ram_mb={ram_used}/{ram_total} temp_max_c={temp_max}"
        )

    lines.append("")
    lines.append("Recent Logs")
    if state.logs:
        for item in list(state.logs)[-args.max_log_lines :]:
            lines.append(f"  {_shorten(item, 160)}")
    else:
        lines.append("  no journal matches yet")

    if state.warnings:
        lines.append("")
        lines.append("Warnings")
        for item in state.warnings:
            lines.append(f"  {_shorten(item, 160)}")

    lines.append("")
    lines.append("Event Counts")
    if not state.event_counts:
        lines.append("  no events yet")
    else:
        for source in sorted(state.event_counts):
            lines.append(f"  {source}: {state.event_counts[source]}")

    lines.append("")
    lines.append("Exit: Ctrl-C")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mac-side telemetry dashboard for PALA.")
    parser.add_argument("--jetson-host", default="jetson")
    parser.add_argument("--jetson-dir", default="~/pala")
    parser.add_argument("--perception-log", default="logs/perception.jsonl")
    parser.add_argument("--actions-log", default="logs/actions.jsonl")
    parser.add_argument("--poll-ms", type=int, default=200)
    parser.add_argument("--from-start", action="store_true")
    parser.add_argument("--heartbeat-s", type=float, default=1.0)
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--no-tegrastats", action="store_true")
    parser.add_argument("--tegrastats-interval-ms", type=int, default=1000)
    parser.add_argument("--no-journal", action="store_true")
    parser.add_argument(
        "--journal-filter",
        default=r"(deepstream|nvinfer|gstreamer|gst|error|timeout|engine)",
    )

    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-source", choices=["dummy", "gst", "tap"], default="tap")
    parser.add_argument("--video-device", default="/dev/video0")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--video-capture-fps", type=int, default=30)
    parser.add_argument("--video-fps", type=float, default=6.0)
    parser.add_argument("--video-max-width", type=int, default=640)
    parser.add_argument("--video-max-height", type=int, default=360)
    parser.add_argument("--video-jpeg-quality", type=int, default=70)
    parser.add_argument("--video-tap-jpeg", default="logs/telemetry/preview/latest.jpg")
    parser.add_argument("--video-tap-meta", default="logs/telemetry/preview/latest.json")
    parser.add_argument("--video-pipeline", default="")
    parser.add_argument("--no-video-window", action="store_true")
    parser.add_argument("--video-window-scale", type=float, default=1.0)

    parser.add_argument("--refresh-hz", type=float, default=4.0)
    parser.add_argument("--reconnect-delay-s", type=float, default=2.0)
    parser.add_argument("--max-log-lines", type=int, default=8)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    stop = threading.Event()
    mapped_dir, map_note = _normalize_jetson_dir(args.jetson_dir)
    args.jetson_dir = mapped_dir

    def _stop_handler(_sig, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    state = DashboardState(host=args.jetson_host)
    if map_note:
        state.logs.append(map_note)
    in_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(256, int(args.queue_size)))

    proc: Optional[subprocess.Popen[str]] = None
    reader_thread: Optional[threading.Thread] = None
    next_connect_time = 0.0
    refresh_s = 1.0 / max(1.0, float(args.refresh_hz))
    last_draw = 0.0

    video_window = _init_video_window(args, state)

    while not stop.is_set():
        now = time.time()

        if proc is None and now >= next_connect_time:
            state.connection_note = "connecting via ssh"
            try:
                proc = _start_ssh_agent(args)
            except OSError as exc:
                state.connected = False
                state.connection_note = f"ssh start failed: {exc!r}"
                next_connect_time = now + max(1.0, float(args.reconnect_delay_s))
            else:
                state.connected = True
                state.connection_note = "connected"
                reader_thread = threading.Thread(
                    target=_reader_loop,
                    kwargs={"stop": stop, "proc": proc, "out_q": in_q},
                    daemon=True,
                )
                reader_thread.start()

        # Drain queue quickly before rendering.
        drained = 0
        while drained < 500:
            try:
                msg = in_q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            if msg.get("source") == "viewer_note":
                payload = msg.get("payload", {})
                if isinstance(payload, dict):
                    text = payload.get("line")
                    if isinstance(text, str):
                        state.logs.append(text)
                continue
            state.apply(msg)

        # Handle process exit / reconnect.
        if proc is not None and proc.poll() is not None:
            state.connected = False
            state.connection_note = f"disconnected (exit={proc.returncode})"
            proc = None
            next_connect_time = time.time() + max(0.5, float(args.reconnect_delay_s))

        video_window = _pump_video_window(video_window, state, args)

        if now - last_draw >= refresh_s:
            screen = _render(state, now_wall_s=now, args=args)
            print("\x1b[2J\x1b[H" + screen, end="", flush=True)
            last_draw = now

        stop.wait(0.05)

    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    if reader_thread is not None:
        reader_thread.join(timeout=1.0)

    if video_window is not None:
        try:
            video_window.root.destroy()
        except Exception:
            pass

    print("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
