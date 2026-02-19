from __future__ import annotations

import argparse
import base64
import collections
import io
import os
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

from tools.telemetry.capture import CaptureConfig, SessionCaptureWriter
from tools.telemetry.filters import matches_field_filters, parse_field_filters
from tools.telemetry.packs import apply_pack_overrides, list_packs, resolve_packs
from tools.telemetry.protocol import decode_message
from tools.telemetry.lamp_viz import draw_lamp_panel
from tools.telemetry.replay import SessionReplayReader


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
    last_rx_wall_s: Optional[float] = None
    last_event_wall_s: Optional[float] = None
    connection_note: str = "starting"
    event_counts: Dict[str, int] = field(default_factory=dict)
    perception: Optional[Dict[str, Any]] = None
    action: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, Any]] = None
    timeline: Optional[Dict[str, Any]] = None
    command: Optional[Dict[str, Any]] = None
    tegrastats: Optional[Dict[str, Any]] = None
    transport: Optional[Dict[str, Any]] = None
    capture_status: Optional[Dict[str, Any]] = None
    agent: Optional[Dict[str, Any]] = None
    logs: Deque[str] = field(default_factory=lambda: collections.deque(maxlen=12))
    warnings: Deque[str] = field(default_factory=lambda: collections.deque(maxlen=8))
    dropped_events_reported: int = 0

    video_frame_bytes: Optional[bytes] = None
    video_frame_meta: Optional[Dict[str, Any]] = None
    video_frames_received: int = 0
    video_decode_errors: int = 0
    video_frames_rejected: int = 0
    last_video_wall_s: Optional[float] = None
    last_agent_wall_s: Optional[float] = None
    max_frame_bytes: int = 2_000_000

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

        if source == "memory_log":
            data = payload.get("data")
            if isinstance(data, dict):
                self.memory = data
            return

        if source == "timeline_log":
            data = payload.get("data")
            if isinstance(data, dict):
                self.timeline = data
            return

        if source == "command_log":
            data = payload.get("data")
            if isinstance(data, dict):
                self.command = data
            return

        if source == "tegrastats":
            self.tegrastats = payload
            return

        if source == "transport_stats":
            self.transport = payload
            return

        if source == "capture_status":
            self.capture_status = payload
            status = payload.get("status")
            if isinstance(status, str):
                self.logs.append(f"capture: {status}")
            if "error" in payload:
                self.warnings.append(f"capture: {payload['error']}")
            return

        if source == "agent":
            self.agent = payload
            self.last_agent_wall_s = time.time()
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
            max_bytes = max(0, int(self.max_frame_bytes))
            est_bytes = (len(frame_b64) * 3) // 4
            if max_bytes > 0 and est_bytes > max_bytes:
                self.video_frames_rejected += 1
                self.warnings.append(f"video_frame: rejected oversized payload est={est_bytes}B")
                return
            try:
                frame_bytes = base64.b64decode(frame_b64, validate=True)
            except Exception:
                self.video_decode_errors += 1
                self.warnings.append("video_frame: invalid base64")
                return
            if max_bytes > 0 and len(frame_bytes) > max_bytes:
                self.video_frames_rejected += 1
                self.warnings.append(f"video_frame: rejected oversized frame bytes={len(frame_bytes)}")
                return
            meta = dict(payload)
            meta.pop("bytes_b64", None)
            tap_extra = payload.get("tap_extra")
            if isinstance(tap_extra, dict):
                cmd = tap_extra.get("command")
                if isinstance(cmd, dict):
                    self.command = cmd
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
        "--memory-log",
        str(args.memory_log),
        "--timeline-log",
        str(args.timeline_log),
        "--poll-ms",
        str(int(args.poll_ms)),
        "--heartbeat-s",
        str(float(args.heartbeat_s)),
        "--queue-size",
        str(int(args.queue_size)),
        "--warning-throttle-s",
        str(float(args.warning_throttle_s)),
        "--worker-restart-delay-s",
        str(float(args.worker_restart_delay_s)),
    ]
    for pack in args.pack:
        agent_args.extend(["--pack", str(pack)])
    for override in args.pack_override:
        agent_args.extend(["--pack-override", str(override)])
    for expr in args.field_filter:
        agent_args.extend(["--field-filter", str(expr)])

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
                "--video-max-bytes",
                str(int(args.video_max_bytes)),
                "--video-wait-warn-s",
                str(float(args.video_wait_warn_s)),
                "--video-tap-jpeg",
                str(args.video_tap_jpeg),
                "--video-tap-meta",
                str(args.video_tap_meta),
            ]
        )
        if args.video_pipeline:
            agent_args.extend(["--video-pipeline", str(args.video_pipeline)])

    if args.agent_capture_dir:
        agent_args.extend(
            [
                "--capture-dir",
                str(args.agent_capture_dir),
                "--capture-frames",
                str(args.capture_frames),
                "--capture-max-seconds",
                str(float(args.capture_max_seconds)),
                "--capture-manifest-version",
                str(int(args.capture_manifest_version)),
            ]
        )

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
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={max(1, int(args.ssh_connect_timeout_s))}",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=3",
            args.jetson_host,
            remote_cmd,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )


def _stop_process(proc: subprocess.Popen[str], *, grace_s: float = 1.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=max(0.2, float(grace_s)))
    except subprocess.TimeoutExpired:
        proc.kill()


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
                try:
                    out_q.put_nowait(
                        {
                            "type": "event",
                            "source": "viewer_note",
                            "payload": {"line": _shorten(text, 200)},
                        }
                    )
                except queue.Full:
                    continue


def _replay_loop(
    *,
    stop: threading.Event,
    out_q: "queue.Queue[Dict[str, Any]]",
    session_dir: str,
    speed: float,
    no_timing: bool,
) -> None:
    try:
        reader = SessionReplayReader(session_dir)
        if reader.manifest:
            try:
                out_q.put_nowait(
                    {
                        "type": "event",
                        "source": "viewer_note",
                        "payload": {
                            "line": f"replay manifest schema_version={reader.manifest.get('schema_version')} "
                            f"events={reader.manifest.get('event_count')}"
                        },
                    }
                )
            except queue.Full:
                pass

        scale = max(0.01, float(speed))
        for msg, delay_s in reader.iter_events():
            if stop.is_set():
                break
            if not no_timing and delay_s is not None and delay_s > 0.0:
                stop.wait(delay_s / scale)
                if stop.is_set():
                    break
            try:
                out_q.put_nowait(msg)
            except queue.Full:
                continue
        try:
            out_q.put_nowait(
                {
                    "type": "event",
                    "source": "viewer_note",
                    "payload": {"line": "replay complete"},
                }
            )
        except queue.Full:
            pass
    except Exception as exc:
        try:
            out_q.put_nowait(
                {
                    "type": "event",
                    "source": "viewer_note",
                    "payload": {"line": f"replay_error: {exc!r}"},
                }
            )
        except queue.Full:
            pass


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


def _compose_video_and_lamp_panel(
    image: Image.Image,
    state: DashboardState,
    *,
    panel_width: int,
) -> Image.Image:
    panel = draw_lamp_panel(height=image.height, width=panel_width, command=state.command)
    canvas = Image.new("RGB", (image.width + panel.width, image.height), (0, 0, 0))
    canvas.paste(image, (0, 0))
    canvas.paste(panel, (image.width, 0))
    return canvas


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
    except Exception as exc:
        state.warnings.append(f"video_window_closed: {_shorten(str(exc), 120)}")
        try:
            window.root.destroy()
        except Exception:
            pass
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
    if not args.no_lamp_panel:
        image = _compose_video_and_lamp_panel(
            image,
            state,
            panel_width=max(180, int(args.lamp_panel_width)),
        )

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


def _panel_enabled(args: argparse.Namespace, panel: str) -> bool:
    selected = getattr(args, "panel", None)
    if not selected:
        return True
    return panel in set(selected)


def _render(state: DashboardState, *, now_wall_s: float, args: argparse.Namespace) -> str:
    lines = []
    if _panel_enabled(args, "summary"):
        uptime = now_wall_s - state.started_wall_s
        mode = "replay" if args.replay else "live"
        lines.append(
            f"PALA Telemetry | mode={mode} | host={state.host} | connected={state.connected} | uptime={uptime:.0f}s"
        )
        lines.append(
            f"Last event age={_fmt_age(now_wall_s, state.last_event_wall_s)} | dropped(agent)={state.dropped_events_reported}"
        )
        lines.append(f"Last rx age={_fmt_age(now_wall_s, state.last_rx_wall_s)}")
        lines.append(f"Last heartbeat age={_fmt_age(now_wall_s, state.last_agent_wall_s)}")
        lines.append(f"Connection note: {state.connection_note}")
        if args.pack:
            lines.append(f"Packs: {', '.join(args.pack)}")
        if args.field_filter:
            lines.append(f"Field filters: {', '.join(args.field_filter)}")
        lines.append("")

    if _panel_enabled(args, "video"):
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
                f"decode_errors={state.video_decode_errors} rejected={state.video_frames_rejected}"
            )
        lines.append("")

    if _panel_enabled(args, "perception"):
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

    if _panel_enabled(args, "action"):
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

    if _panel_enabled(args, "command"):
        lines.append("Command")
        if state.command is None:
            lines.append("  no data yet")
        else:
            enabled = state.command.get("enable")
            angles = state.command.get("joint_angles_rad")
            names = state.command.get("joint_names")
            if isinstance(angles, list) and isinstance(names, list) and names:
                pairs = []
                for i in range(min(len(names), len(angles))):
                    try:
                        pairs.append(f"{names[i]}={float(angles[i]):+.2f}")
                    except (TypeError, ValueError):
                        continue
                lines.append(f"  enable={enabled} angles=[{', '.join(pairs[:6])}]")
            else:
                lines.append(f"  enable={enabled} angles={angles}")
        lines.append("")

    if _panel_enabled(args, "system"):
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

    if _panel_enabled(args, "memory"):
        lines.append("Memory")
        if state.memory is None:
            lines.append("  no data yet")
        else:
            lines.append(
                f"  type={state.memory.get('type')} ts_wall_s={state.memory.get('ts_wall_s')}"
            )
            payload = state.memory.get("payload")
            if isinstance(payload, dict):
                highlights = payload.get("highlights")
                if isinstance(highlights, list):
                    lines.append(f"  highlights={_shorten(', '.join(str(x) for x in highlights), 140)}")
        lines.append("")

    if _panel_enabled(args, "timeline"):
        lines.append("Timeline")
        if state.timeline is None:
            lines.append("  no data yet")
        else:
            lines.append(
                f"  type={state.timeline.get('type')} ts_wall_s={state.timeline.get('ts_wall_s')} "
                f"ts_mono_s={state.timeline.get('ts_mono_s')}"
            )
            payload = state.timeline.get("payload")
            if isinstance(payload, dict):
                lines.append(f"  payload_keys={','.join(sorted(payload.keys())[:8])}")
        lines.append("")

    if _panel_enabled(args, "transport"):
        lines.append("Transport")
        if state.transport is None:
            lines.append("  no data yet")
        else:
            lines.append(
                f"  queue_depth={state.transport.get('queue_depth')} "
                f"queue_capacity={state.transport.get('queue_capacity')} "
                f"dropped={state.transport.get('dropped_events')}"
            )
        if state.capture_status is not None:
            lines.append(
                f"  capture_status={state.capture_status.get('status')} "
                f"capture_dir={state.capture_status.get('capture_dir')}"
            )
        lines.append("")

    if _panel_enabled(args, "logs"):
        lines.append("Recent Logs")
        if state.logs:
            for item in list(state.logs)[-args.max_log_lines :]:
                lines.append(f"  {_shorten(item, 160)}")
        else:
            lines.append("  no journal matches yet")
        lines.append("")

    if _panel_enabled(args, "warnings") and state.warnings:
        lines.append("Warnings")
        for item in state.warnings:
            lines.append(f"  {_shorten(item, 160)}")
        lines.append("")

    if _panel_enabled(args, "events"):
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
    parser.add_argument("--list-packs", action="store_true", help="List built-in signal packs and exit.")
    parser.add_argument("--pack", action="append", default=[], help="Signal pack to enable (repeatable).")
    parser.add_argument(
        "--pack-override",
        action="append",
        default=[],
        help="Pack override key=value (forwarded to agent).",
    )
    parser.add_argument(
        "--field-filter",
        action="append",
        default=[],
        help="Field predicate source.path<op>value where op in =,!=,<,>,~.",
    )
    parser.add_argument(
        "--panel",
        action="append",
        default=[],
        choices=["summary", "video", "perception", "action", "command", "system", "memory", "timeline", "transport", "logs", "warnings", "events"],
        help="Restrict visible dashboard panels (repeatable). Default derives from active packs.",
    )
    parser.add_argument("--jetson-host", default="jetson")
    parser.add_argument("--jetson-dir", default="~/pala")
    parser.add_argument("--perception-log", default="logs/perception.jsonl")
    parser.add_argument("--actions-log", default="logs/actions.jsonl")
    parser.add_argument("--memory-log", default="logs/orchestrator_memory.jsonl")
    parser.add_argument("--timeline-log", default="logs/orchestrator_timeline.jsonl")
    parser.add_argument("--poll-ms", type=int, default=200)
    parser.add_argument("--from-start", action="store_true")
    parser.add_argument("--heartbeat-s", type=float, default=1.0)
    parser.add_argument("--queue-size", type=int, default=1024)
    parser.add_argument("--warning-throttle-s", type=float, default=2.0)
    parser.add_argument("--worker-restart-delay-s", type=float, default=1.0)
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
    parser.add_argument("--video-max-bytes", type=int, default=700_000)
    parser.add_argument("--video-wait-warn-s", type=float, default=5.0)
    parser.add_argument("--video-tap-jpeg", default="logs/telemetry/preview/latest.jpg")
    parser.add_argument("--video-tap-meta", default="logs/telemetry/preview/latest.json")
    parser.add_argument("--video-pipeline", default="")
    parser.add_argument("--max-frame-bytes", type=int, default=2_000_000)
    parser.add_argument("--no-video-window", action="store_true")
    parser.add_argument("--video-window-scale", type=float, default=1.0)
    parser.add_argument("--no-lamp-panel", action="store_true")
    parser.add_argument("--lamp-panel-width", type=int, default=260)

    parser.add_argument("--refresh-hz", type=float, default=4.0)
    parser.add_argument("--reconnect-delay-s", type=float, default=2.0)
    parser.add_argument("--reconnect-backoff", type=float, default=1.6)
    parser.add_argument("--reconnect-max-delay-s", type=float, default=15.0)
    parser.add_argument("--stale-timeout-s", type=float, default=10.0)
    parser.add_argument("--ssh-connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--agent-capture-dir", default="", help="Remote capture directory on Jetson for agent-side capture.")
    parser.add_argument("--capture-frames", choices=["off", "keyframes", "all"], default="off")
    parser.add_argument("--capture-max-seconds", type=float, default=0.0)
    parser.add_argument("--capture-manifest-version", type=int, default=1)
    parser.add_argument("--save-session", default="", help="Local capture directory on Mac for viewer-side session bundle.")
    parser.add_argument("--replay", default="", help="Replay an existing local capture directory instead of SSH live mode.")
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--replay-no-timing", action="store_true")
    parser.add_argument("--max-log-lines", type=int, default=8)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.list_packs:
        for pack in list_packs():
            print(f"{pack.name}: {pack.description}")
        return 0
    if not args.pack:
        args.pack = ["runtime_core"]
    try:
        resolved_packs = resolve_packs(args.pack)
        resolved_packs = apply_pack_overrides(resolved_packs, args.pack_override)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.panel:
        args.panel = sorted(set(resolved_packs.panels) | {"summary"})
    try:
        field_filters = parse_field_filters(args.field_filter)
    except ValueError as exc:
        parser.error(str(exc))

    stop = threading.Event()
    replay_mode = bool(str(args.replay or "").strip())
    map_note: Optional[str] = None
    if not replay_mode:
        mapped_dir, map_note = _normalize_jetson_dir(args.jetson_dir)
        args.jetson_dir = mapped_dir

    def _stop_handler(_sig, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    state = DashboardState(host=args.jetson_host, max_frame_bytes=max(0, int(args.max_frame_bytes)))
    if map_note:
        state.logs.append(map_note)
    in_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(256, int(args.queue_size)))

    proc: Optional[subprocess.Popen[str]] = None
    proc_started_wall_s: Optional[float] = None
    reader_thread: Optional[threading.Thread] = None
    replay_thread: Optional[threading.Thread] = None
    next_connect_time = 0.0
    reconnect_attempt = 0
    reconnect_base_s = max(0.5, float(args.reconnect_delay_s))
    reconnect_backoff = max(1.0, float(args.reconnect_backoff))
    reconnect_max_s = max(reconnect_base_s, float(args.reconnect_max_delay_s))
    stale_timeout_s = max(2.0, float(args.stale_timeout_s))
    refresh_s = 1.0 / max(1.0, float(args.refresh_hz))
    last_draw = 0.0
    capture_writer: Optional[SessionCaptureWriter] = None
    save_session_dir = str(args.save_session or "").strip()
    if save_session_dir:
        cfg = CaptureConfig(
            directory=save_session_dir,
            frames_mode=args.capture_frames,
            max_seconds=max(0.0, float(args.capture_max_seconds)),
            manifest_version=int(args.capture_manifest_version),
            metadata={
                "mode": "replay" if replay_mode else "live",
                "packs": list(args.pack),
                "field_filters": list(args.field_filter),
            },
        )
        try:
            capture_writer = SessionCaptureWriter(cfg)
            state.logs.append(f"local capture started: {save_session_dir}")
        except Exception as exc:
            capture_writer = None
            state.warnings.append(f"local capture init failed: {exc!r}")

    def _schedule_reconnect(reason: str) -> None:
        nonlocal next_connect_time, reconnect_attempt
        reconnect_attempt += 1
        delay = min(reconnect_max_s, reconnect_base_s * (reconnect_backoff ** (reconnect_attempt - 1)))
        next_connect_time = time.time() + delay
        state.connection_note = f"{reason}; retry in {delay:.1f}s"

    video_window = _init_video_window(args, state)

    if replay_mode:
        replay_dir = str(args.replay).strip()
        if not os.path.isdir(replay_dir):
            parser.error(f"replay directory not found: {replay_dir}")
        state.connected = True
        state.connection_note = f"replay from {replay_dir}"
        replay_thread = threading.Thread(
            target=_replay_loop,
            kwargs={
                "stop": stop,
                "out_q": in_q,
                "session_dir": replay_dir,
                "speed": max(0.01, float(args.replay_speed)),
                "no_timing": bool(args.replay_no_timing),
            },
            daemon=True,
        )
        replay_thread.start()

    while not stop.is_set():
        now = time.time()

        if (not replay_mode) and proc is None and now >= next_connect_time:
            state.connection_note = "connecting via ssh"
            try:
                proc = _start_ssh_agent(args)
            except OSError as exc:
                state.connected = False
                _schedule_reconnect(f"ssh start failed: {exc!r}")
            else:
                state.connected = True
                reconnect_attempt = 0
                state.connection_note = "connected (awaiting events)"
                proc_started_wall_s = time.time()
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
                        if text.startswith("telemetry_agent_error:"):
                            state.warnings.append(text)
                        state.logs.append(text)
                continue
            state.last_rx_wall_s = time.time()
            if not matches_field_filters(msg, field_filters):
                continue
            state.apply(msg)
            if capture_writer is not None:
                keep_writing = capture_writer.write(msg)
                if not keep_writing:
                    capture_writer.close()
                    capture_writer = None
                    state.logs.append("local capture stopped: max_seconds_elapsed")

        if (not replay_mode) and proc is not None and proc.poll() is None:
            stale_ref_s = state.last_rx_wall_s
            if stale_ref_s is None:
                stale_ref_s = proc_started_wall_s
            stale_age = None if stale_ref_s is None else (now - stale_ref_s)
            if stale_age is None:
                pass
            elif stale_age <= stale_timeout_s:
                pass
            else:
                state.warnings.append(f"stream stale for {stale_age:.1f}s; reconnecting")
                _stop_process(proc, grace_s=0.8)
                proc = None
                proc_started_wall_s = None
                state.connected = False
                if reader_thread is not None:
                    reader_thread.join(timeout=1.0)
                    reader_thread = None
                _schedule_reconnect("reconnecting after stale stream")

        # Handle process exit / reconnect.
        if (not replay_mode) and proc is not None and proc.poll() is not None:
            state.connected = False
            exit_code = proc.returncode
            proc = None
            proc_started_wall_s = None
            if reader_thread is not None:
                reader_thread.join(timeout=1.0)
                reader_thread = None
            _schedule_reconnect(f"disconnected (exit={exit_code})")

        if replay_mode and replay_thread is not None and not replay_thread.is_alive() and in_q.empty():
            stop.set()

        video_window = _pump_video_window(video_window, state, args)

        if now - last_draw >= refresh_s:
            screen = _render(state, now_wall_s=now, args=args)
            print("\x1b[2J\x1b[H" + screen, end="", flush=True)
            last_draw = now

        stop.wait(0.05)

    if proc is not None and proc.poll() is None:
        _stop_process(proc, grace_s=1.0)

    if reader_thread is not None:
        reader_thread.join(timeout=1.0)
    if replay_thread is not None:
        replay_thread.join(timeout=1.0)
    if capture_writer is not None:
        capture_writer.close()

    if video_window is not None:
        try:
            video_window.root.destroy()
        except Exception:
            pass

    print("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
