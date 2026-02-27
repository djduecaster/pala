from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import os
import queue
import select
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional

from PIL import Image, ImageDraw

try:
    from PIL import ImageTk
except Exception:  # pragma: no cover - depends on Tk availability.
    ImageTk = None

try:
    import tkinter as tk
except Exception:  # pragma: no cover - depends on Tk availability.
    tk = None

from tools.telemetry.annotations import annotation_key, append_annotation, load_annotations
from tools.telemetry.capture import CaptureConfig, SessionCaptureWriter
from tools.telemetry.dataset_export import export_dataset_rows
from tools.telemetry.protocol import decode_message
from tools.telemetry.lamp_viz import draw_lamp_panel
from tools.telemetry.quality import evaluate_quality_gate, load_quality_report
from tools.telemetry.reasoning import ReasoningEvent, format_reasoning_snippet, normalize_reasoning_message
from tools.telemetry.replay import SessionReplayReader
from tools.telemetry.schema_v3 import TELEMETRY_SCHEMA_VERSION_V3
from tools.telemetry.storage_sqlite import query_cases_db, query_session_db, resolve_session_db_path, review_case
from tools.telemetry.integrity import verify_integrity_report
from tools.telemetry.trace_graph import TraceGraphBuilder, TraceRecord, resolve_trace_index_path, load_trace_index

try:
    import termios
    import tty
except Exception:  # pragma: no cover - depends on platform.
    termios = None
    tty = None


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


@dataclass(frozen=True)
class TimelineRow:
    row_id: int
    case_id: str
    event_index: int
    trace_id: str
    req_id: Optional[int]
    component: str
    source: str
    ts_wall_s: Optional[float]
    phase: str
    status: str
    severity: str
    latency_ms: Optional[float]
    confidence: Optional[float]
    snippet: str
    labels_csv: str
    decision: str
    perception_frame_id: Optional[int]
    video_frame_id: Optional[int]
    perception_zone_hint: str
    provenance: str = "sqlite.reasoning_traces"


@dataclass(frozen=True)
class TimelineRowsResult:
    rows: List[TimelineRow]
    total_count: int
    source: str
    note: str = ""
    unavailable_reason: str = ""


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
    command: Optional[Dict[str, Any]] = None
    tegrastats: Optional[Dict[str, Any]] = None
    logs: Deque[str] = field(default_factory=lambda: collections.deque(maxlen=12))
    warnings: Deque[str] = field(default_factory=lambda: collections.deque(maxlen=8))
    dropped_events_reported: int = 0
    local_dropped_events: int = 0

    video_frame_bytes: Optional[bytes] = None
    video_frame_meta: Optional[Dict[str, Any]] = None
    video_frames_received: int = 0
    video_decode_errors: int = 0
    video_frames_rejected: int = 0
    last_video_wall_s: Optional[float] = None
    last_agent_wall_s: Optional[float] = None
    max_frame_bytes: int = 2_000_000
    reasoning_events: Deque[ReasoningEvent] = field(default_factory=lambda: collections.deque(maxlen=400))
    reasoning_selected_seq: Optional[int] = None
    reasoning_filter_mode: str = "all"  # all | errors | slow
    reasoning_show_help: bool = False
    focus_panel: str = "reasoning_stream"
    key_reader_enabled: bool = False
    reasoning_seq_counter: int = 0
    active_panels: List[str] = field(default_factory=list)
    trace_records: List[TraceRecord] = field(default_factory=list)
    trace_selected_id: Optional[str] = None
    trace_pinned_id: Optional[str] = None
    query_text: str = ""
    query_rows: List[Dict[str, Any]] = field(default_factory=list)
    query_note: str = ""
    timeline_rows: List[TimelineRow] = field(default_factory=list)
    timeline_note: str = ""
    timeline_source: str = "unavailable"
    timeline_unavailable_reason: str = "timeline not initialized"
    timeline_total_rows: int = 0
    timeline_selected_row_id: Optional[int] = None
    integrity_report: Optional[Dict[str, Any]] = None
    annotations: Deque[Dict[str, Any]] = field(default_factory=lambda: collections.deque(maxlen=120))
    annotation_keys: set[str] = field(default_factory=set)
    annotation_session_dir: str = ""
    active_preset: str = ""
    quality_report: Optional[Dict[str, Any]] = None
    quality_gate_note: str = ""
    quality_gate_passed: Optional[bool] = None

    def configure_panels(self, panels: List[str], *, focus_panel: str = "") -> None:
        self.active_panels = list(panels)
        if focus_panel and focus_panel in self.active_panels:
            self.focus_panel = focus_panel
        elif self.focus_panel not in self.active_panels:
            self.focus_panel = self.active_panels[0] if self.active_panels else "summary"

    def apply(self, msg: Dict[str, Any]) -> None:
        source = str(msg.get("source", "unknown"))
        self.last_event_wall_s = time.time()
        self.event_counts[source] = self.event_counts.get(source, 0) + 1

        payload = msg.get("payload", {})
        if not isinstance(payload, dict):
            return
        self._ingest_reasoning_event(msg)

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

        if source == "timeline_log":
            return

        if source == "tegrastats":
            self.tegrastats = payload
            return

        if source == "transport_stats":
            dropped = payload.get("dropped_events")
            if isinstance(dropped, int):
                self.dropped_events_reported = max(self.dropped_events_reported, dropped)
            return

        if source == "capture_status":
            status = payload.get("status")
            if isinstance(status, str):
                self.logs.append(f"capture: {status}")
            if "error" in payload:
                self.warnings.append(f"capture: {payload['error']}")
            return

        if source == "agent":
            self.last_agent_wall_s = time.time()
            dropped = payload.get("dropped_events")
            if isinstance(dropped, int):
                self.dropped_events_reported = max(self.dropped_events_reported, dropped)
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

    def _ingest_reasoning_event(self, msg: Dict[str, Any]) -> None:
        event = normalize_reasoning_message(msg)
        if event is None:
            return
        self.reasoning_seq_counter += 1
        # Keep seq in-band using phase prefix so rendering can still show it.
        enriched = ReasoningEvent(
            source=event.source,
            ts_wall_s=event.ts_wall_s,
            req_id=event.req_id,
            phase=f"{event.phase}",
            status=event.status,
            latency_ms=event.latency_ms,
            primitive=event.primitive,
            confidence=event.confidence,
            target_zone=event.target_zone,
            model=event.model,
            provider=event.provider,
            snippet=event.snippet,
            severity=event.severity,
        )
        self.reasoning_events.append(enriched)
        if self.reasoning_selected_seq is None:
            self.reasoning_selected_seq = self.reasoning_seq_counter

    def _iter_reasoning_with_seq(self) -> List[tuple[int, ReasoningEvent]]:
        start_seq = max(1, self.reasoning_seq_counter - len(self.reasoning_events) + 1)
        out: List[tuple[int, ReasoningEvent]] = []
        for idx, event in enumerate(self.reasoning_events):
            out.append((start_seq + idx, event))
        return out

    def filtered_reasoning_events(self, *, slow_ms: float) -> List[tuple[int, ReasoningEvent]]:
        mode = str(self.reasoning_filter_mode or "all")
        events = self._iter_reasoning_with_seq()
        if mode == "errors":
            return [(seq, ev) for seq, ev in events if ev.severity in {"error", "warning"}]
        if mode == "slow":
            threshold = max(1.0, float(slow_ms))
            return [(seq, ev) for seq, ev in events if ev.latency_ms is not None and ev.latency_ms >= threshold]
        return events

    def selected_reasoning_event(self, *, slow_ms: float) -> Optional[tuple[int, ReasoningEvent]]:
        events = self.filtered_reasoning_events(slow_ms=slow_ms)
        if not events:
            return None
        if self.reasoning_selected_seq is None:
            self.reasoning_selected_seq = events[-1][0]
            return events[-1]
        for seq, event in events:
            if seq == self.reasoning_selected_seq:
                return seq, event
        self.reasoning_selected_seq = events[-1][0]
        return events[-1]

    def move_reasoning_selection(self, *, delta: int, slow_ms: float) -> None:
        events = self.filtered_reasoning_events(slow_ms=slow_ms)
        if not events:
            self.reasoning_selected_seq = None
            return
        current_idx = len(events) - 1
        if self.reasoning_selected_seq is not None:
            for idx, (seq, _) in enumerate(events):
                if seq == self.reasoning_selected_seq:
                    current_idx = idx
                    break
        target_idx = max(0, min(len(events) - 1, current_idx + int(delta)))
        self.reasoning_selected_seq = events[target_idx][0]

    def cycle_focus_panel(self, *, delta: int) -> None:
        if not self.active_panels:
            return
        if self.focus_panel not in self.active_panels:
            self.focus_panel = self.active_panels[0]
            return
        idx = self.active_panels.index(self.focus_panel)
        self.focus_panel = self.active_panels[(idx + int(delta)) % len(self.active_panels)]

    def set_traces(self, traces: List[TraceRecord]) -> None:
        self.trace_records = list(traces)
        trace_ids = {trace.trace_id for trace in self.trace_records}
        if self.trace_pinned_id not in trace_ids:
            self.trace_pinned_id = None
        if self.trace_selected_id not in trace_ids:
            self.trace_selected_id = self.trace_records[0].trace_id if self.trace_records else None

    def selected_trace(self) -> Optional[TraceRecord]:
        if not self.trace_records:
            return None
        if self.trace_pinned_id:
            for trace in self.trace_records:
                if trace.trace_id == self.trace_pinned_id:
                    return trace
        if self.trace_selected_id:
            for trace in self.trace_records:
                if trace.trace_id == self.trace_selected_id:
                    return trace
        self.trace_selected_id = self.trace_records[0].trace_id
        return self.trace_records[0]

    def move_trace_selection(self, delta: int) -> None:
        if not self.trace_records:
            self.trace_selected_id = None
            return
        current_idx = 0
        if self.trace_selected_id:
            for idx, trace in enumerate(self.trace_records):
                if trace.trace_id == self.trace_selected_id:
                    current_idx = idx
                    break
        target_idx = max(0, min(len(self.trace_records) - 1, current_idx + int(delta)))
        self.trace_selected_id = self.trace_records[target_idx].trace_id

    def set_timeline_rows(
        self,
        rows: List[TimelineRow],
        *,
        total_count: int,
        source: str,
        note: str = "",
        unavailable_reason: str = "",
    ) -> None:
        self.timeline_rows = list(rows)
        self.timeline_total_rows = max(0, int(total_count))
        self.timeline_source = str(source or "unavailable")
        self.timeline_note = str(note or "")
        self.timeline_unavailable_reason = str(unavailable_reason or "")

        if self.timeline_rows:
            row_ids = {row.row_id for row in self.timeline_rows}
            if self.timeline_selected_row_id not in row_ids:
                self.timeline_selected_row_id = self.timeline_rows[-1].row_id
        else:
            self.timeline_selected_row_id = None

    def selected_timeline_row(self) -> Optional[TimelineRow]:
        if not self.timeline_rows:
            return None
        if self.timeline_selected_row_id is not None:
            for row in self.timeline_rows:
                if row.row_id == self.timeline_selected_row_id:
                    return row
        self.timeline_selected_row_id = self.timeline_rows[-1].row_id
        return self.timeline_rows[-1]

    def move_timeline_selection(self, delta: int) -> None:
        if not self.timeline_rows:
            self.timeline_selected_row_id = None
            return
        current_idx = len(self.timeline_rows) - 1
        if self.timeline_selected_row_id is not None:
            for idx, row in enumerate(self.timeline_rows):
                if row.row_id == self.timeline_selected_row_id:
                    current_idx = idx
                    break
        target_idx = max(0, min(len(self.timeline_rows) - 1, current_idx + int(delta)))
        self.timeline_selected_row_id = self.timeline_rows[target_idx].row_id

    def toggle_trace_pin(self) -> None:
        selected = self.selected_trace()
        if selected is None:
            return
        if self.trace_pinned_id == selected.trace_id:
            self.trace_pinned_id = None
            self.logs.append("trace pin cleared")
            return
        self.trace_pinned_id = selected.trace_id
        self.logs.append(f"trace pinned: {selected.trace_id}")

    def add_annotation(self, row: Dict[str, Any]) -> bool:
        entry = dict(row)
        key = annotation_key(entry)
        if key in self.annotation_keys:
            return False
        max_len = self.annotations.maxlen
        if max_len is not None and len(self.annotations) >= max_len:
            dropped = self.annotations.popleft()
            self.annotation_keys.discard(annotation_key(dropped))
        self.annotations.append(entry)
        self.annotation_keys.add(key)
        return True


@dataclass
class _VideoWindow:
    root: Any
    label: Any
    last_frame_id: int = -1


@dataclass(frozen=True)
class ModeConfig:
    pack: str
    panels: tuple[str, ...]
    default_query: str = ""
    quality_gate: str = "warn"
    index_mode: str = "auto"
    no_video: bool = False


MODE_CONFIGS: Dict[str, ModeConfig] = {
    "live": ModeConfig(
        pack="reasoning_live",
        panels=("summary", "trace_list", "reasoning_stream", "case_list", "case_detail", "quality", "video"),
        quality_gate="warn",
        index_mode="auto",
        no_video=False,
    ),
    "replay": ModeConfig(
        pack="reasoning_live",
        panels=(
            "summary",
            "trace_list",
            "trace_detail",
            "reasoning_stream",
            "case_list",
            "case_detail",
            "query",
            "quality",
            "video",
        ),
        quality_gate="warn",
        index_mode="auto",
        no_video=False,
    ),
    "curate": ModeConfig(
        pack="behavior_v2_debug",
        panels=("summary", "trace_list", "trace_detail", "case_list", "case_detail", "query", "quality", "annotations"),
        default_query="kind:joined severity:error|warning status:parse_fail|timeout sort:severity",
        quality_gate="strict",
        index_mode="sqlite",
        no_video=True,
    ),
}


def _cycle_reasoning_filter(state: DashboardState) -> None:
    modes = ["all", "errors", "slow"]
    current = str(state.reasoning_filter_mode or "all")
    try:
        idx = modes.index(current)
    except ValueError:
        idx = 0
    state.reasoning_filter_mode = modes[(idx + 1) % len(modes)]
    state.logs.append(f"reasoning filter={state.reasoning_filter_mode}")


class _KeyboardReader:
    def __init__(self) -> None:
        self.enabled = False
        self._fd: Optional[int] = None
        self._saved_termios = None

    def start(self) -> bool:
        if termios is None or tty is None:
            return False
        if not sys.stdin.isatty():
            return False
        try:
            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            return False
        self._fd = fd
        self._saved_termios = saved
        self.enabled = True
        return True

    def stop(self) -> None:
        if not self.enabled:
            return
        self.enabled = False
        if self._fd is not None and self._saved_termios is not None and termios is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_termios)
            except Exception:
                pass

    def poll(self) -> str:
        if not self.enabled or self._fd is None:
            return ""
        chars: List[str] = []
        while True:
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.0)
            except Exception:
                break
            if not ready:
                break
            try:
                buf = os.read(self._fd, 64)
            except Exception:
                break
            if not buf:
                break
            chars.append(buf.decode("utf-8", errors="ignore"))
            if len(chars) >= 4:
                break
        return "".join(chars)


class _Counter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def inc(self, value: int = 1) -> None:
        delta = max(0, int(value))
        if delta <= 0:
            return
        with self._lock:
            self._value += delta

    def value(self) -> int:
        with self._lock:
            return int(self._value)


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
        "--behavior-env-log",
        str(args.behavior_env_log),
        "--behavior-planner-log",
        str(args.behavior_planner_log),
        "--behavior-reasoning-log",
        str(args.behavior_reasoning_log),
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
        "--trace-match-window-s",
        str(float(args.trace_match_window_s)),
        "--trace-max-events",
        str(max(128, int(args.trace_max_events))),
    ]
    packs = list(getattr(args, "pack", []) or ["reasoning_live"])
    for pack in packs:
        agent_args.extend(["--pack", str(pack)])

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
                "--index-live-every",
                str(max(0, int(args.index_live_every))),
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
    local_drops: Optional[_Counter] = None,
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
                if local_drops is not None:
                    local_drops.inc()
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
                    if local_drops is not None:
                        local_drops.inc()
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
        if isinstance(reader.integrity, dict):
            try:
                out_q.put_nowait(
                    {
                        "type": "event",
                        "source": "viewer_note",
                        "payload": {
                            "line": (
                                f"replay integrity ok={bool(reader.integrity.get('ok'))} "
                                f"missing={len(reader.integrity.get('missing') or [])} "
                                f"mismatch={len(reader.integrity.get('mismatch') or [])}"
                            )
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
                    "payload": {"line": f"replay complete (decode_errors={reader.decode_errors})"},
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

    return _VideoWindow(root=root, label=label)


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
    except Exception as exc:
        state.video_decode_errors += 1
        state.warnings.append(f"video_window_update_failed: {exc!r}")
        return window

    if isinstance(frame_id, int):
        window.last_frame_id = frame_id
    else:
        window.last_frame_id += 1

    return window


def _focus_prefix(state: DashboardState, panel: str) -> str:
    if state.focus_panel == panel:
        return "> "
    return "  "


def _query_text_match(query: str, text: str) -> bool:
    raw_tokens = [tok for tok in str(query or "").strip().split() if tok]
    if not raw_tokens:
        return True
    hay = str(text or "").lower()
    groups: List[List[str]] = []
    for raw in raw_tokens:
        token = str(raw).strip()
        low = token.lower()
        if low.startswith("sort:") or low.startswith("order:") or low.startswith("kind:"):
            continue
        if (
            low.startswith("latency_ms>")
            or low.startswith("latency_ms<")
            or low.startswith("duration_ms>")
            or low.startswith("duration_ms<")
            or low.startswith("ts:[")
        ):
            # Structured numeric/time filters do not map cleanly to text fallback.
            continue
        if ":" in token:
            key, value = token.split(":", 1)
            if key.strip().lower() in {"source", "severity", "status", "phase", "req", "trace", "component"}:
                token = value.strip()
        options = [part.strip().lower() for part in token.split("|") if part.strip()]
        if options:
            groups.append(options)
    if not groups:
        return True
    return all(any(opt in hay for opt in group) for group in groups)


def _build_in_memory_query_rows(state: DashboardState, *, limit: int) -> List[Dict[str, Any]]:
    lim = max(1, int(limit))
    candidates: List[tuple[float, Dict[str, Any]]] = []
    for seq, event in reversed(state._iter_reasoning_with_seq()):
        text = " ".join(
            [
                str(event.source or ""),
                str(event.phase or ""),
                str(event.status or ""),
                str(event.severity or ""),
                str(event.req_id if event.req_id is not None else ""),
                str(event.snippet or ""),
            ]
        )
        if not _query_text_match(state.query_text, text):
            continue
        candidates.append(
            (
                float(seq),
                {
                    "kind": "reasoning",
                    "id": seq,
                    "source": event.source,
                    "req_id": event.req_id,
                    "status": event.status,
                    "severity": event.severity,
                    "summary": format_reasoning_snippet(
                        event.snippet,
                        max_chars=120,
                        redact=False,
                    ),
                },
            )
        )

    trace_cap = max(lim * 10, 200)
    trace_added = 0
    for trace in reversed(state.trace_records):
        for ref in reversed(trace.event_refs):
            text = " ".join(
                [
                    str(trace.trace_id or ""),
                    str(ref.source or ""),
                    str(ref.phase or ""),
                    str(ref.status or ""),
                    str(ref.severity or ""),
                    str(ref.req_id if ref.req_id is not None else ""),
                    str(ref.summary or ""),
                ]
            )
            if not _query_text_match(state.query_text, text):
                continue
            candidates.append(
                (
                    float(ref.event_index),
                    {
                        "kind": "event",
                        "id": f"{trace.trace_id}:{ref.event_index}",
                        "trace_id": trace.trace_id,
                        "req_id": ref.req_id if ref.req_id is not None else trace.req_id,
                        "source": ref.source,
                        "status": ref.status,
                        "severity": ref.severity,
                        "summary": ref.summary,
                    },
                )
            )
            trace_added += 1
            if trace_added >= trace_cap:
                break
        if trace_added >= trace_cap:
            break
    candidates.sort(key=lambda item: item[0], reverse=True)
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in candidates:
        dedupe_key = "\x1f".join(
            [
                str(row.get("kind") or ""),
                str(row.get("id") or ""),
                str(row.get("trace_id") or ""),
                str(row.get("source") or ""),
            ]
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(row)
        if len(rows) >= lim:
            break
    return rows


def _load_manifest_schema_version(session_dir: str) -> Optional[int]:
    root = str(session_dir or "").strip()
    if not root:
        return None
    manifest_path = os.path.join(root, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            decoded = json.load(fh)
    except Exception:
        return None
    if not isinstance(decoded, dict):
        return None
    try:
        return int(decoded.get("schema_version", 0) or 0)
    except (TypeError, ValueError):
        return None


def _as_int_optional(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float_optional(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_timeline_row(case_row: Mapping[str, Any], *, ordinal: int) -> TimelineRow:
    first_event_index = int(case_row.get("first_event_index", 0) or 0)
    row_id = first_event_index if first_event_index > 0 else int(ordinal)
    return TimelineRow(
        row_id=row_id,
        case_id=str(case_row.get("case_id") or ""),
        event_index=first_event_index,
        trace_id=str(case_row.get("trace_id") or ""),
        req_id=_as_int_optional(case_row.get("req_id")),
        component=str(case_row.get("component") or ""),
        source=str(case_row.get("source") or "sqlite.cases.v4"),
        ts_wall_s=_as_float_optional(case_row.get("start_ts_wall_s")),
        phase="case",
        status=str(case_row.get("status") or ""),
        severity=str(case_row.get("severity") or "info"),
        latency_ms=_as_float_optional(case_row.get("max_latency_ms")),
        confidence=_as_float_optional(case_row.get("hardness")),
        snippet=str(case_row.get("summary") or case_row.get("snippet") or ""),
        labels_csv=str(case_row.get("labels_csv") or ""),
        decision=str(case_row.get("decision") or ""),
        perception_frame_id=None,
        video_frame_id=None,
        perception_zone_hint="",
        provenance="sqlite.cases.v4",
    )


def load_timeline_rows(
    *,
    session_root: str,
    session_db_path: str,
    query_text: str,
    selected_trace_id: str,
    limit: int,
    index_mode: str,
    required_schema_version: int = TELEMETRY_SCHEMA_VERSION_V3,
) -> TimelineRowsResult:
    mode = str(index_mode or "auto")
    if mode == "off":
        return TimelineRowsResult(
            rows=[],
            total_count=0,
            source="unavailable",
            note="case explorer requires sqlite index (index_mode=off)",
            unavailable_reason="index_mode_off",
        )
    root = _expand_local_path(session_root)
    if not root:
        return TimelineRowsResult(
            rows=[],
            total_count=0,
            source="unavailable",
            note="case explorer requires --save-session (live) or --replay <session_dir>",
            unavailable_reason="session_root_missing",
        )
    schema_version = _load_manifest_schema_version(root)
    if schema_version is not None and schema_version < int(required_schema_version):
        return TimelineRowsResult(
            rows=[],
            total_count=0,
            source="unavailable",
            note=(
                f"case explorer requires schema_version>={required_schema_version}; "
                "run: uv run python -m tools.telemetry.migrate_session <session_dir>"
            ),
            unavailable_reason=f"schema_version={schema_version}",
        )
    db_path = _expand_local_path(session_db_path) if str(session_db_path or "").strip() else resolve_session_db_path(root)
    if not db_path or (not os.path.exists(db_path)):
        return TimelineRowsResult(
            rows=[],
            total_count=0,
            source="unavailable",
            note=(
                "session.db unavailable for case explorer; capture/migrate session first: "
                "uv run python -m tools.telemetry.migrate_session <session_dir>"
            ),
            unavailable_reason="session_db_missing",
        )
    parts: List[str] = ["sort:hardness", "order:desc"]
    trace_id = str(selected_trace_id or "").strip()
    if trace_id:
        parts.append(f"trace:{trace_id}")
    query = str(query_text or "").strip()
    if query:
        parts.append(query)
    case_query = " ".join(parts)
    try:
        out = query_cases_db(db_path, query=case_query, limit=max(4, int(limit)))
    except Exception as exc:
        return TimelineRowsResult(
            rows=[],
            total_count=0,
            source="unavailable",
            note=f"case sqlite query failed: {exc!r}",
            unavailable_reason="sqlite_query_failed",
        )
    case_rows = list(out.get("cases", []))
    rows = [_to_timeline_row(row, ordinal=idx + 1) for idx, row in enumerate(case_rows) if isinstance(row, dict)]
    total_count = int(out.get("total_count", len(rows)) or 0)
    query_note = (
        f"source=sqlite.cases.v4 matched={len(rows)}/{total_count} "
        f"trace={(trace_id or '*')} limit={max(4, int(limit))}"
    )
    return TimelineRowsResult(
        rows=rows,
        total_count=total_count,
        source="sqlite.cases.v4",
        note=query_note,
        unavailable_reason="",
    )


def _render_reasoning_stream(lines: List[str], state: DashboardState, args: argparse.Namespace) -> None:
    lines.append(f"{_focus_prefix(state, 'reasoning_stream')}Reasoning Stream")
    filtered = state.filtered_reasoning_events(slow_ms=float(args.reasoning_slow_ms))
    if not filtered:
        lines.append("  no reasoning events yet")
        lines.append("")
        return
    selected = state.selected_reasoning_event(slow_ms=float(args.reasoning_slow_ms))
    selected_seq = selected[0] if selected is not None else None
    max_rows = max(4, int(args.max_log_lines))
    for seq, event in filtered[-max_rows:]:
        mark = "*" if seq == selected_seq else " "
        latency = f"{event.latency_ms:.0f}ms" if event.latency_ms is not None else "-"
        req = event.req_id if event.req_id is not None else "-"
        snippet = format_reasoning_snippet(
            event.snippet,
            max_chars=int(args.reasoning_snippet_max_chars),
            redact=str(args.reasoning_redact) == "on",
        )
        lines.append(
            f" {mark} #{seq} req={req} phase={event.phase} status={event.status or '-'} "
            f"lat={latency} sev={event.severity}"
        )
        if snippet:
            lines.append(f"    {_shorten(snippet, 180)}")
    lines.append("")


def _render_trace_list(lines: List[str], state: DashboardState, args: argparse.Namespace) -> None:
    lines.append(f"{_focus_prefix(state, 'trace_list')}Trace List")
    if not state.trace_records:
        lines.append("  no traces yet")
        lines.append("")
        return
    selected = state.selected_trace()
    selected_id = selected.trace_id if selected is not None else None
    max_rows = max(4, int(args.max_log_lines))
    for trace in state.trace_records[:max_rows]:
        sel = "*" if trace.trace_id == selected_id else " "
        pin = "P" if state.trace_pinned_id == trace.trace_id else " "
        duration = f"{trace.duration_ms:.0f}ms" if trace.duration_ms is not None else "n/a"
        req = trace.req_id if trace.req_id is not None else "-"
        lines.append(
            f" {sel}{pin} {trace.trace_id} req={req} status={trace.status} "
            f"sev={trace.severity} dur={duration}"
        )
        lines.append(f"    {_shorten(trace.summary, 160)}")
    lines.append("")


def _render_trace_detail(lines: List[str], state: DashboardState, args: argparse.Namespace) -> None:
    lines.append(f"{_focus_prefix(state, 'trace_detail')}Trace Detail")
    trace = state.selected_trace()
    if trace is None:
        lines.append("  no trace selected")
        lines.append("")
        return
    pin_state = "pinned" if state.trace_pinned_id == trace.trace_id else "unpinned"
    lines.append(
        f"  trace_id={trace.trace_id} req_id={trace.req_id} status={trace.status} "
        f"severity={trace.severity} {pin_state}"
    )
    lines.append(
        f"  start={trace.start_ts_wall_s} end={trace.end_ts_wall_s} duration_ms={trace.duration_ms}"
    )
    lines.append(f"  summary={_shorten(trace.summary, 220)}")
    refs = list(trace.event_refs)[-max(4, int(args.max_log_lines)) :]
    if not refs:
        lines.append("  no events")
        lines.append("")
        return
    lines.append("  events:")
    for ref in refs:
        latency = f"{ref.latency_ms:.0f}ms" if ref.latency_ms is not None else "-"
        lines.append(
            f"    #{ref.event_index} {ref.source} phase={ref.phase or '-'} "
            f"status={ref.status or '-'} lat={latency}"
        )
    lines.append("")


def _render_quality_panel(lines: List[str], state: DashboardState) -> None:
    lines.append(f"{_focus_prefix(state, 'quality')}Quality")
    report = state.quality_report
    if report is None:
        lines.append("  no quality report loaded")
        if state.quality_gate_note:
            lines.append(f"  gate={state.quality_gate_note}")
        lines.append("")
        return
    score = report.get("score")
    grade = report.get("grade")
    lines.append(f"  grade={grade} score={score}")
    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        lines.append(
            "  "
            f"events={metrics.get('event_count')} reasoning={metrics.get('reasoning_count')} "
            f"traces={metrics.get('trace_count')} errors={metrics.get('reasoning_errors')}"
        )
        lines.append(
            "  "
            f"p50={metrics.get('latency_ms_p50')}ms p95={metrics.get('latency_ms_p95')}ms "
            f"slow={metrics.get('reasoning_slow')}"
        )
    if state.quality_gate_note:
        lines.append(f"  gate={state.quality_gate_note}")
    lines.append("")


def _render_query_panel(lines: List[str], state: DashboardState, args: argparse.Namespace) -> None:
    lines.append(f"{_focus_prefix(state, 'query')}Query")
    query_text = str(state.query_text or "")
    if not query_text:
        lines.append("  no query (set --query)")
        lines.append("")
        return
    lines.append(f"  expr={query_text}")
    if state.query_note:
        lines.append(f"  note={_shorten(state.query_note, 120)}")
    if not state.query_rows:
        lines.append("  no matches")
        lines.append("")
        return
    for row in state.query_rows[: max(1, int(args.query_limit))]:
        req = row.get("req_id")
        trace_id = row.get("trace_id")
        source = row.get("source")
        status = row.get("status")
        severity = row.get("severity")
        summary = row.get("summary")
        lines.append(
            f"  [{row.get('kind')}] id={row.get('id')} src={source or '-'} trace={trace_id or '-'} "
            f"req={req if req is not None else '-'} status={status or '-'} sev={severity or '-'}"
        )
        if summary:
            lines.append(f"    {_shorten(str(summary), 150)}")
    lines.append("")


def _render_timeline_panel(lines: List[str], state: DashboardState, args: argparse.Namespace) -> None:
    lines.append(f"{_focus_prefix(state, 'case_list')}Case Explorer")
    selected_trace = state.selected_trace()
    trace_id = selected_trace.trace_id if selected_trace is not None else ""
    lines.append(f"  source={state.timeline_source or 'unavailable'} trace={(trace_id or '*')}")
    lines.append(
        f"  rows={len(state.timeline_rows)}/{max(0, int(state.timeline_total_rows))} "
        f"selected={state.timeline_selected_row_id if state.timeline_selected_row_id is not None else 'n/a'}"
    )
    if state.timeline_note:
        lines.append(f"  note={_shorten(state.timeline_note, 140)}")
    if state.timeline_unavailable_reason:
        lines.append(f"  unavailable={_shorten(state.timeline_unavailable_reason, 140)}")
    if not state.timeline_rows:
        lines.append("  no timeline rows")
        lines.append("")
        return
    selected = state.selected_timeline_row()
    selected_id = selected.row_id if selected is not None else None
    for row in state.timeline_rows[-max(4, int(args.max_log_lines)) :]:
        marker = "*" if row.row_id == selected_id else " "
        latency_text = f"{row.latency_ms:.0f}ms" if row.latency_ms is not None else "-"
        hardness_text = f"{row.confidence:.2f}" if row.confidence is not None else "-"
        lines.append(
            f" {marker} case={row.case_id or '-'} row={row.row_id} ts={row.ts_wall_s} comp={row.component or '-'} "
            f"phase={row.phase or '-'} status={row.status or '-'} lat={latency_text} "
            f"hard={hardness_text} decision={row.decision or '-'} req={row.req_id if row.req_id is not None else '-'}"
        )
        if row.snippet:
            lines.append(f"    {_shorten(row.snippet, 180)}")
    lines.append("")


def _render_timeline_detail_panel(lines: List[str], state: DashboardState) -> None:
    lines.append(f"{_focus_prefix(state, 'case_detail')}Case Detail")
    row = state.selected_timeline_row()
    if row is None:
        lines.append("  no case selected")
        lines.append("")
        return
    lines.append(
        f"  case={row.case_id or '-'} row={row.row_id} trace={row.trace_id or '-'} req={row.req_id if row.req_id is not None else '-'} "
        f"source={row.provenance}"
    )
    lines.append(
        f"  ts={row.ts_wall_s} comp={row.component or '-'} phase={row.phase or '-'} "
        f"status={row.status or '-'} sev={row.severity or '-'} lat={row.latency_ms} hard={row.confidence}"
    )
    lines.append(
        f"  labels={row.labels_csv or '-'} decision={row.decision or '-'}"
    )
    if row.snippet:
        lines.append(f"  snippet={_shorten(row.snippet, 260)}")
    lines.append("")


def _render_annotations_panel(lines: List[str], state: DashboardState, args: argparse.Namespace) -> None:
    lines.append(f"{_focus_prefix(state, 'annotations')}Annotations")
    if not state.annotations:
        lines.append("  no bookmarks yet (press 'b')")
        lines.append("")
        return
    for row in list(state.annotations)[-max(4, int(args.max_log_lines)) :]:
        lines.append(
            f"  tag={row.get('tag') or '-'} trace={row.get('trace_id') or '-'} "
            f"req={row.get('req_id') if row.get('req_id') is not None else '-'}"
        )
        note = str(row.get("note") or "").strip()
        if note:
            lines.append(f"    {_shorten(note, 140)}")
    lines.append("")


def _panel_enabled(args: argparse.Namespace, panel: str) -> bool:
    selected = getattr(args, "panel", None)
    if not selected:
        return True
    return panel in set(selected)


def _run_curation_export(
    *,
    session_dir: str,
    profile: str,
) -> Dict[str, Any]:
    root = str(session_dir).strip()
    if not root:
        return {"ok": False, "error": "session directory unavailable: empty (set --save-session or use --replay)"}
    if not os.path.isdir(root):
        return {"ok": False, "error": f"session directory unavailable: {root}"}
    try:
        result = export_dataset_rows(
            root,
            profile=str(profile or "hard_cases"),
            include_unlabeled=False,
            write_manifest=True,
        )
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    row_count = int(result.get("row_count", 0))
    manifest_path = str(result.get("manifest_path") or "")
    manifest: Dict[str, Any] = {}
    if manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                decoded = json.load(fh)
            if isinstance(decoded, dict):
                manifest = decoded
        except Exception:
            manifest = {}
    if row_count <= 0:
        return {
            "ok": False,
            "error": "curation export produced zero rows",
            "profile": result.get("profile"),
            "row_count": row_count,
            "output_path": str(result.get("output_path") or ""),
            "manifest_path": manifest_path,
        }
    return {
        "ok": True,
        "profile": result.get("profile"),
        "row_count": row_count,
        "output_path": str(result.get("output_path") or ""),
        "manifest_path": manifest_path,
        "annotated_row_count": int(manifest.get("annotated_row_count", 0) or 0),
        "annotation_count": int(manifest.get("annotation_count", 0) or 0),
        "label_coverage_ratio": manifest.get("label_coverage_ratio"),
        "annotation_coverage_ratio": manifest.get("annotation_coverage_ratio"),
        "hard_case_ratio": manifest.get("hard_case_ratio"),
    }


def _build_viewer_summary(
    *,
    mode: str,
    state: DashboardState,
    query_text: str,
    quality_gate: str,
    curation_result: Optional[Dict[str, Any]],
    exit_code: int,
) -> Dict[str, Any]:
    ended_wall_s = time.time()
    duration_s = max(0.0, ended_wall_s - float(state.started_wall_s))
    quality_grade = None
    quality_score = None
    case_source = str(state.timeline_source or "unavailable")
    case_reason = str(state.timeline_unavailable_reason or "")
    if case_source == "sqlite.cases.v4":
        case_reason = ""
    reviewed_count = sum(1 for row in state.timeline_rows if str(row.decision or "").strip())
    if isinstance(state.quality_report, dict):
        quality_grade = state.quality_report.get("grade")
        quality_score = state.quality_report.get("score")
    return {
        "version": 4,
        "schema_version": int(TELEMETRY_SCHEMA_VERSION_V3),
        "run_id": f"{int(ended_wall_s * 1000)}-{os.getpid()}",
        "mode": str(mode or "live"),
        "started_at_wall_s": float(state.started_wall_s),
        "ended_at_wall_s": ended_wall_s,
        "session_duration_s": round(duration_s, 3),
        "query": str(query_text or ""),
        "quality_gate": str(quality_gate or "warn"),
        "quality_gate_passed": bool(state.quality_gate_passed) if state.quality_gate_passed is not None else None,
        "quality_gate_note": str(state.quality_gate_note or ""),
        "quality_grade": quality_grade,
        "quality_score": quality_score,
        "case_source": case_source,
        "case_rows_total": int(state.timeline_total_rows),
        "case_rows_visible": int(len(state.timeline_rows)),
        "case_reviewed_visible": int(reviewed_count),
        "case_unavailable_reason": case_reason,
        "event_counts": dict(state.event_counts),
        "trace_count": len(state.trace_records),
        "reasoning_count": len(state.reasoning_events),
        "dropped_events_agent": int(state.dropped_events_reported),
        "dropped_events_local": int(state.local_dropped_events),
        "warning_count": len(state.warnings),
        "curation_result": dict(curation_result) if isinstance(curation_result, dict) else None,
        "exit_code": int(exit_code),
    }


def _write_json_atomic(path: str, obj: Mapping[str, Any]) -> None:
    target = str(path)
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    temp_path = f"{target}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"), ensure_ascii=True)
    os.replace(temp_path, target)


def _write_viewer_summary(
    *,
    session_dir: str,
    summary: Dict[str, Any],
    filename: str = "viewer_summary.json",
) -> str:
    root = str(session_dir).strip()
    if not root:
        raise ValueError("session directory unavailable for viewer summary")
    os.makedirs(root, exist_ok=True)
    target = os.path.join(root, str(filename))
    _write_json_atomic(target, summary)
    return target


def _append_viewer_run(
    *,
    session_dir: str,
    summary: Mapping[str, Any],
    filename: str = "viewer_runs.jsonl",
) -> str:
    root = str(session_dir).strip()
    if not root:
        raise ValueError("session directory unavailable for viewer run history")
    os.makedirs(root, exist_ok=True)
    target = os.path.join(root, str(filename))
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(summary), separators=(",", ":"), ensure_ascii=True))
        fh.write("\n")
    return target


def _expand_local_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    return os.path.normpath(os.path.expanduser(raw))


def _paths_equivalent(path_a: str, path_b: str) -> bool:
    left = _expand_local_path(path_a)
    right = _expand_local_path(path_b)
    if not left or not right:
        return False
    left_abs = os.path.abspath(left)
    right_abs = os.path.abspath(right)
    if left_abs == right_abs:
        return True
    if os.path.exists(left_abs) and os.path.exists(right_abs):
        try:
            return os.path.samefile(left_abs, right_abs)
        except OSError:
            return False
    return False


def _default_curation_session_dir(*, now_wall_s: Optional[float] = None) -> str:
    ts = time.localtime(time.time() if now_wall_s is None else float(now_wall_s))
    stamp = time.strftime("%Y%m%d_%H%M%S", ts)
    return os.path.join("logs", "telemetry", f"session_curate_{stamp}")


def _resolve_live_save_session_dir(
    *,
    replay_mode: bool,
    save_session_dir: str,
    curate_on_exit: bool,
    now_wall_s: Optional[float] = None,
) -> tuple[str, str]:
    root = _expand_local_path(save_session_dir)
    if replay_mode or (not curate_on_exit) or root:
        return root, ""
    auto_dir = _default_curation_session_dir(now_wall_s=now_wall_s)
    return auto_dir, f"curation enabled; auto save_session={auto_dir}"


def _render(state: DashboardState, *, now_wall_s: float, args: argparse.Namespace) -> str:
    lines = []
    if _panel_enabled(args, "summary"):
        uptime = now_wall_s - state.started_wall_s
        mode = "replay" if args.replay else "live"
        lines.append(f"PALA Telemetry | {mode} | host={state.host} | up={uptime:.0f}s")
        lines.append(
            f"conn={'ok' if state.connected else 'down'} "
            f"rx={_fmt_age(now_wall_s, state.last_rx_wall_s)} "
            f"event={_fmt_age(now_wall_s, state.last_event_wall_s)} "
            f"hb={_fmt_age(now_wall_s, state.last_agent_wall_s)} "
            f"drops=agent:{state.dropped_events_reported}/local:{state.local_dropped_events}"
        )
        preset_text = state.active_preset or "-"
        lines.append(
            f"mode={preset_text} focus={state.focus_panel} "
            f"reasoning={state.reasoning_filter_mode} hotkeys={'on' if state.key_reader_enabled else 'off'}"
        )
        selected_trace = state.selected_trace()
        lines.append(
            "Traces: "
            f"count={len(state.trace_records)} selected={(selected_trace.trace_id if selected_trace else 'n/a')} "
            f"pinned={(state.trace_pinned_id or 'none')}"
        )
        if state.quality_report is not None:
            lines.append(
                "Quality: "
                f"grade={state.quality_report.get('grade')} score={state.quality_report.get('score')} "
                f"gate={state.quality_gate_note or 'n/a'}"
            )
        if state.integrity_report is not None:
            lines.append(
                "Integrity: "
                f"ok={bool(state.integrity_report.get('ok'))} "
                f"missing={len(state.integrity_report.get('missing') or [])} "
                f"mismatch={len(state.integrity_report.get('mismatch') or [])}"
            )
        if state.query_text:
            lines.append(f"Query: '{state.query_text}' rows_shown={len(state.query_rows)}")
        lines.append(
            "Cases: "
            f"source={state.timeline_source} rows={len(state.timeline_rows)}/{state.timeline_total_rows} "
            f"selected={(state.timeline_selected_row_id if state.timeline_selected_row_id is not None else 'n/a')}"
        )
        if state.connection_note:
            lines.append(f"Note: {_shorten(state.connection_note, 120)}")
        lines.append("")

    if _panel_enabled(args, "reasoning_stream"):
        _render_reasoning_stream(lines, state, args)

    if _panel_enabled(args, "trace_list"):
        _render_trace_list(lines, state, args)

    if _panel_enabled(args, "trace_detail"):
        _render_trace_detail(lines, state, args)

    if _panel_enabled(args, "quality"):
        _render_quality_panel(lines, state)

    if _panel_enabled(args, "query"):
        _render_query_panel(lines, state, args)

    if _panel_enabled(args, "case_list"):
        _render_timeline_panel(lines, state, args)

    if _panel_enabled(args, "case_detail"):
        _render_timeline_detail_panel(lines, state)

    if _panel_enabled(args, "annotations"):
        _render_annotations_panel(lines, state, args)

    if _panel_enabled(args, "video"):
        lines.append(f"{_focus_prefix(state, 'video')}Video")
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

    if state.reasoning_show_help:
        lines.append("Hotkeys")
        lines.append("  ?: toggle this help")
        lines.append("  h/l: move focus panel")
        lines.append("  j/k: previous/next reasoning event (or case when case panel is focused)")
        lines.append("  u/i: previous/next trace")
        lines.append("  f: cycle reasoning filter (all/errors/slow)")
        lines.append("  r: toggle reasoning redaction")
        lines.append("  o: focus case detail panel")
        lines.append("  p: pin/unpin selected trace")
        lines.append("  b: bookmark selected trace/event")
        lines.append("  a/x/n/m: review case (accept/reject/needs_context/label)")
        lines.append("  Ctrl-C: exit")
        lines.append("")
    lines.append(
        "Cmd: [? help] [h/l focus] [j/k reasoning|case] [u/i trace] [o detail] [a/x/n/m review] [p pin] [b bookmark] [f filter] [r redact] [Ctrl-C exit]"
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mac-side telemetry dashboard for PALA.")
    hidden = argparse.SUPPRESS

    # Primary UX surface (keep this compact).
    parser.add_argument(
        "--mode",
        choices=["live", "replay", "curate"],
        default="live",
        help="Run mode: live stream, replay a session, or curate/export dataset rows.",
    )
    parser.add_argument("--jetson-host", default="jetson", help="SSH host alias for Jetson.")
    parser.add_argument("--jetson-dir", default="~/pala", help="Project directory on Jetson.")
    parser.add_argument("--replay", default="", help="Replay a local telemetry session directory.")
    parser.add_argument("--save-session", default="", help="Write local capture bundle directory.")
    parser.add_argument("--query", default="", help="Indexed telemetry query expression.")
    parser.add_argument("--index-mode", choices=["auto", "off", "sqlite"], default="auto", help=hidden)
    parser.add_argument("--quality-gate", choices=["off", "warn", "strict"], default="warn")
    parser.add_argument("--no-video", action="store_true", help="Disable video stream rendering.")
    parser.add_argument("--curate-on-exit", action="store_true", help="Export dataset rows when viewer exits.")
    parser.add_argument(
        "--curate-profile",
        choices=["fast", "strict", "hard_cases"],
        default="hard_cases",
        help="Dataset profile used by --curate-on-exit.",
    )

    # Advanced/runtime controls.
    parser.add_argument("--from-start", action="store_true", help=hidden)
    parser.add_argument("--agent-capture-dir", default="", help=hidden)
    parser.add_argument("--capture-frames", choices=["off", "keyframes", "all"], default="off", help=hidden)
    parser.add_argument("--capture-max-seconds", type=float, default=0.0, help=hidden)
    parser.add_argument("--stale-timeout-s", type=float, default=10.0, help=hidden)
    parser.add_argument("--bookmark-tag", default="bookmark", help=hidden)
    parser.add_argument("--focus-panel", default="", help=hidden)
    parser.add_argument("--perception-log", default="logs/perception.jsonl", help=hidden)
    parser.add_argument("--actions-log", default="logs/actions.jsonl", help=hidden)
    parser.add_argument("--memory-log", default="logs/orchestrator_memory.jsonl", help=hidden)
    parser.add_argument("--timeline-log", default="logs/orchestrator_timeline.jsonl", help=hidden)
    parser.add_argument("--behavior-env-log", default="logs/behavior_env.jsonl", help=hidden)
    parser.add_argument("--behavior-planner-log", default="logs/behavior_planner.jsonl", help=hidden)
    parser.add_argument("--behavior-reasoning-log", default="logs/behavior_reasoning.jsonl", help=hidden)
    parser.add_argument("--poll-ms", type=int, default=200, help=hidden)
    parser.add_argument("--heartbeat-s", type=float, default=1.0, help=hidden)
    parser.add_argument("--queue-size", type=int, default=1024, help=hidden)
    parser.add_argument("--warning-throttle-s", type=float, default=2.0, help=hidden)
    parser.add_argument("--worker-restart-delay-s", type=float, default=1.0, help=hidden)
    parser.add_argument("--no-tegrastats", action="store_true", help=hidden)
    parser.add_argument("--tegrastats-interval-ms", type=int, default=1000, help=hidden)
    parser.add_argument("--no-journal", action="store_true", help=hidden)
    parser.add_argument("--journal-filter", default=r"(deepstream|nvinfer|gstreamer|gst|error|timeout|engine)", help=hidden)
    parser.add_argument("--video-source", choices=["dummy", "gst", "tap"], default="tap", help=hidden)
    parser.add_argument("--video-device", default="/dev/video0", help=hidden)
    parser.add_argument("--video-width", type=int, default=640, help=hidden)
    parser.add_argument("--video-height", type=int, default=360, help=hidden)
    parser.add_argument("--video-capture-fps", type=int, default=30, help=hidden)
    parser.add_argument("--video-fps", type=float, default=6.0, help=hidden)
    parser.add_argument("--video-max-width", type=int, default=640, help=hidden)
    parser.add_argument("--video-max-height", type=int, default=360, help=hidden)
    parser.add_argument("--video-jpeg-quality", type=int, default=70, help=hidden)
    parser.add_argument("--video-max-bytes", type=int, default=700_000, help=hidden)
    parser.add_argument("--video-wait-warn-s", type=float, default=5.0, help=hidden)
    parser.add_argument("--video-tap-jpeg", default="logs/telemetry/preview/latest.jpg", help=hidden)
    parser.add_argument("--video-tap-meta", default="logs/telemetry/preview/latest.json", help=hidden)
    parser.add_argument("--video-pipeline", default="", help=hidden)
    parser.add_argument("--max-frame-bytes", type=int, default=2_000_000, help=hidden)
    parser.add_argument("--no-video-window", action="store_true", help=hidden)
    parser.add_argument("--video-window-scale", type=float, default=1.0, help=hidden)
    parser.add_argument("--no-lamp-panel", action="store_true", help=hidden)
    parser.add_argument("--lamp-panel-width", type=int, default=260, help=hidden)
    parser.add_argument("--refresh-hz", type=float, default=4.0, help=hidden)
    parser.add_argument("--reconnect-delay-s", type=float, default=2.0, help=hidden)
    parser.add_argument("--reconnect-backoff", type=float, default=1.6, help=hidden)
    parser.add_argument("--reconnect-max-delay-s", type=float, default=15.0, help=hidden)
    parser.add_argument("--ssh-connect-timeout-s", type=float, default=5.0, help=hidden)
    parser.add_argument("--capture-manifest-version", type=int, default=TELEMETRY_SCHEMA_VERSION_V3, help=hidden)
    parser.add_argument("--index-live-every", type=int, default=0, help=hidden)
    parser.add_argument("--replay-speed", type=float, default=1.0, help=hidden)
    parser.add_argument("--replay-no-timing", action="store_true", help=hidden)
    parser.add_argument("--max-log-lines", type=int, default=8, help=hidden)
    parser.add_argument("--reasoning-snippet-max-chars", type=int, default=180, help=hidden)
    parser.add_argument("--reasoning-redact", choices=["on", "off"], default="on", help=hidden)
    parser.add_argument("--reasoning-slow-ms", type=float, default=2000.0, help=hidden)
    parser.add_argument("--trace-match-window-s", type=float, default=2.0, help=hidden)
    parser.add_argument("--trace-max-events", type=int, default=1000, help=hidden)
    parser.add_argument("--query-limit", type=int, default=10, help=hidden)
    parser.set_defaults(pack=[], panel=[])
    return parser


def _apply_mode_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> List[str]:
    notes: List[str] = []
    mode = str(getattr(args, "mode", "live") or "live").strip().lower()
    replay = str(getattr(args, "replay", "") or "").strip()
    save_session = str(getattr(args, "save_session", "") or "").strip()
    cfg = MODE_CONFIGS.get(mode)
    if cfg is None:
        parser.error(f"unknown mode: {mode}")
        return notes

    if mode == "live":
        if replay:
            parser.error("--mode live does not accept --replay (use --mode replay or --mode curate)")
    elif mode == "replay":
        if not replay:
            parser.error("--mode replay requires --replay <session_dir>")
    elif mode == "curate":
        if (not replay) and (not save_session):
            parser.error("--mode curate requires --replay <session_dir> or --save-session <session_dir>")
        args.curate_on_exit = True
    args.pack = [cfg.pack]
    args.panel = list(cfg.panels)
    if (not str(args.query or "").strip()) and cfg.default_query:
        args.query = cfg.default_query
    if str(args.quality_gate or "warn") == "warn" and cfg.quality_gate != "warn":
        args.quality_gate = cfg.quality_gate
    if str(args.index_mode or "auto") == "auto" and cfg.index_mode != "auto":
        args.index_mode = cfg.index_mode
    if cfg.no_video:
        args.no_video = True
    return notes


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    notes = _apply_mode_defaults(args, parser)

    stop = threading.Event()
    replay_dir = _expand_local_path(str(args.replay or ""))
    replay_mode = bool(replay_dir)
    if replay_dir:
        args.replay = replay_dir
    map_note: Optional[str] = None
    if not replay_mode:
        mapped_dir, map_note = _normalize_jetson_dir(args.jetson_dir)
        args.jetson_dir = mapped_dir

    def _stop_handler(_sig, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    state = DashboardState(host=args.jetson_host, max_frame_bytes=max(0, int(args.max_frame_bytes)))
    state.active_preset = str(args.mode or "live")
    state.configure_panels(args.panel, focus_panel=str(args.focus_panel or ""))
    state.query_text = str(args.query or "").strip()
    if "trace_list" in args.panel:
        state.focus_panel = "trace_list"
    elif "reasoning_stream" in args.panel:
        state.focus_panel = "reasoning_stream"
    for note in notes:
        state.logs.append(note)
    if map_note:
        state.logs.append(map_note)
    in_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(256, int(args.queue_size)))
    trace_builder = TraceGraphBuilder(
        match_window_s=max(0.1, float(args.trace_match_window_s)),
        max_events=max(128, int(args.trace_max_events)),
    )
    key_reader = _KeyboardReader()
    state.key_reader_enabled = key_reader.start()
    if not state.key_reader_enabled:
        state.logs.append("hotkeys unavailable (non-tty/platform)")

    proc: Optional[subprocess.Popen[str]] = None
    proc_started_wall_s: Optional[float] = None
    reader_thread: Optional[threading.Thread] = None
    reader_local_drops = _Counter()
    replay_thread: Optional[threading.Thread] = None
    replay_reader: Optional[SessionReplayReader] = None
    curation_result: Optional[Dict[str, Any]] = None
    using_preloaded_trace_index = False
    next_connect_time = 0.0
    reconnect_attempt = 0
    reconnect_base_s = max(0.5, float(args.reconnect_delay_s))
    reconnect_backoff = max(1.0, float(args.reconnect_backoff))
    reconnect_max_s = max(reconnect_base_s, float(args.reconnect_max_delay_s))
    stale_timeout_s = max(2.0, float(args.stale_timeout_s))
    refresh_s = 1.0 / max(1.0, float(args.refresh_hz))
    last_draw = 0.0
    last_query_refresh_s = 0.0
    last_timeline_refresh_s = 0.0
    session_db_path = ""
    capture_writer: Optional[SessionCaptureWriter] = None
    save_session_dir, save_session_note = _resolve_live_save_session_dir(
        replay_mode=replay_mode,
        save_session_dir=str(args.save_session or ""),
        curate_on_exit=bool(args.curate_on_exit),
    )
    if save_session_dir:
        args.save_session = save_session_dir
    if replay_mode and save_session_dir and _paths_equivalent(save_session_dir, replay_dir):
        parser.error("--save-session must differ from --replay to avoid overwriting replay data")
    if save_session_note:
        state.logs.append(save_session_note)
    if save_session_dir:
        state.annotation_session_dir = save_session_dir
        for row in load_annotations(save_session_dir, limit=120):
            state.add_annotation(dict(row))
        state.integrity_report = verify_integrity_report(save_session_dir)
        session_db_path = resolve_session_db_path(save_session_dir)
        cfg = CaptureConfig(
            directory=save_session_dir,
            frames_mode=args.capture_frames,
            max_seconds=max(0.0, float(args.capture_max_seconds)),
            manifest_version=int(args.capture_manifest_version),
            trace_match_window_s=max(0.1, float(args.trace_match_window_s)),
            index_live_every=max(0, int(args.index_live_every)),
            metadata={
                "mode": "replay" if replay_mode else "live",
                "packs": list(args.pack),
                "mode_profile": state.active_preset,
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

    def _apply_quality_gate() -> None:
        passed, note = evaluate_quality_gate(state.quality_report, str(args.quality_gate))
        state.quality_gate_passed = passed
        state.quality_gate_note = note

    def _resolve_lookup_root() -> str:
        if session_db_path and os.path.exists(session_db_path):
            return session_db_path
        if replay_mode and os.path.isdir(replay_dir):
            return replay_dir
        if save_session_dir and os.path.isdir(save_session_dir):
            return save_session_dir
        return ""

    def _resolve_timeline_session_root() -> str:
        if replay_mode and os.path.isdir(replay_dir):
            return replay_dir
        if save_session_dir and os.path.isdir(save_session_dir):
            return save_session_dir
        return ""

    _apply_quality_gate()
    video_window = _init_video_window(args, state)

    if replay_mode:
        replay_dir = _expand_local_path(str(args.replay))
        if not os.path.isdir(replay_dir):
            parser.error(f"replay directory not found: {replay_dir}")
        state.annotation_session_dir = replay_dir
        for row in load_annotations(replay_dir, limit=120):
            state.add_annotation(dict(row))
        state.connected = True
        state.connection_note = f"replay from {replay_dir}"
        replay_reader = SessionReplayReader(replay_dir)
        session_db_path = resolve_session_db_path(replay_dir)
        state.quality_report = load_quality_report(replay_dir)
        state.integrity_report = replay_reader.integrity
        if isinstance(state.integrity_report, dict) and (not bool(state.integrity_report.get("ok"))):
            state.warnings.append(
                "integrity check failed: "
                f"missing={len(state.integrity_report.get('missing') or [])} "
                f"mismatch={len(state.integrity_report.get('mismatch') or [])}"
            )
        _apply_quality_gate()
        try:
            trace_index_path = resolve_trace_index_path(replay_dir, replay_reader.manifest)
            if os.path.exists(trace_index_path):
                preloaded_traces = load_trace_index(trace_index_path)
                state.set_traces(preloaded_traces)
                if preloaded_traces:
                    using_preloaded_trace_index = True
                    state.logs.append(f"preloaded {len(preloaded_traces)} traces from trace index")
        except Exception as exc:
            state.warnings.append(f"trace_index_load_failed: {exc!r}")
        if state.query_text:
            if args.index_mode == "off":
                state.query_note = "query index disabled (--index-mode off)"
                state.query_rows = _build_in_memory_query_rows(state, limit=int(args.query_limit))
            else:
                try:
                    query_out = query_session_db(
                        session_db_path if os.path.exists(session_db_path) else replay_dir,
                        query=state.query_text,
                        limit=int(args.query_limit),
                    )
                    event_rows = [
                        {
                            "kind": "event",
                            "id": row.get("seq"),
                            "trace_id": row.get("trace_id"),
                            "req_id": row.get("req_id"),
                            "source": row.get("source"),
                            "status": row.get("status"),
                            "severity": row.get("severity"),
                            "summary": row.get("snippet"),
                        }
                        for row in query_out.get("events", [])
                    ]
                    joined_rows = [
                        {
                            "kind": "joined",
                            "id": row.get("row_id"),
                            "trace_id": row.get("trace_id"),
                            "req_id": row.get("req_id"),
                            "source": row.get("component") or row.get("source"),
                            "status": row.get("status"),
                            "severity": row.get("severity"),
                            "summary": row.get("snippet") or row.get("output_preview"),
                        }
                        for row in query_out.get("joined", [])
                    ]
                    state.query_rows = (event_rows + joined_rows)[: max(1, int(args.query_limit))]
                    state.query_note = (
                        f"sqlite events={len(query_out.get('events', []))} "
                        f"joined={len(query_out.get('joined', []))}"
                    )
                except Exception as exc:
                    state.query_note = f"sqlite query failed: {exc!r}"
                    state.query_rows = _build_in_memory_query_rows(state, limit=int(args.query_limit))
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
        state.local_dropped_events = reader_local_drops.value()
        for key in key_reader.poll():
            if key == "?":
                state.reasoning_show_help = not state.reasoning_show_help
                continue
            if key == "h":
                state.cycle_focus_panel(delta=-1)
                continue
            if key == "l":
                state.cycle_focus_panel(delta=1)
                continue
            if key == "j":
                if state.focus_panel in {"case_list", "case_detail"}:
                    state.move_timeline_selection(-1)
                else:
                    state.move_reasoning_selection(delta=-1, slow_ms=float(args.reasoning_slow_ms))
                continue
            if key == "k":
                if state.focus_panel in {"case_list", "case_detail"}:
                    state.move_timeline_selection(1)
                else:
                    state.move_reasoning_selection(delta=1, slow_ms=float(args.reasoning_slow_ms))
                continue
            if key == "u":
                state.move_trace_selection(-1)
                continue
            if key == "i":
                state.move_trace_selection(1)
                continue
            if key == "o":
                if "case_detail" in state.active_panels:
                    state.focus_panel = "case_detail"
                elif "trace_detail" in state.active_panels:
                    state.focus_panel = "trace_detail"
                continue
            if key in {"a", "x", "n", "m"}:
                decision = {"a": "accept", "x": "reject", "n": "needs_context", "m": "label"}[key]
                if session_db_path and os.path.exists(session_db_path):
                    selected_case = state.selected_timeline_row()
                    if selected_case is not None and selected_case.case_id:
                        try:
                            review_case(
                                session_db_path,
                                case_id=selected_case.case_id,
                                decision=decision,
                                note="",
                                reviewer="viewer",
                            )
                            state.logs.append(f"case review: {selected_case.case_id} -> {decision}")
                        except Exception as exc:
                            state.warnings.append(f"case review failed: {exc!r}")
                continue
            if key == "p":
                state.toggle_trace_pin()
                continue
            if key == "b":
                selected_trace = state.selected_trace()
                selected_reasoning = state.selected_reasoning_event(slow_ms=float(args.reasoning_slow_ms))
                annotation = {
                    "tag": str(args.bookmark_tag or "bookmark"),
                    "trace_id": selected_trace.trace_id if selected_trace is not None else "",
                    "req_id": selected_trace.req_id if selected_trace is not None else None,
                    "note": (
                        selected_reasoning[1].snippet
                        if selected_reasoning is not None and selected_reasoning[1].snippet
                        else "bookmark"
                    ),
                    "phase": selected_reasoning[1].phase if selected_reasoning is not None else "",
                    "status": selected_reasoning[1].status if selected_reasoning is not None else "",
                    "source": selected_reasoning[1].source if selected_reasoning is not None else "",
                }
                if state.annotation_session_dir:
                    try:
                        row = append_annotation(state.annotation_session_dir, annotation)
                        state.add_annotation(row)
                        state.logs.append("annotation saved")
                    except Exception as exc:
                        state.warnings.append(f"annotation write failed: {exc!r}")
                else:
                    annotation["note"] = f"{annotation.get('note')} (ephemeral)"
                    state.add_annotation(annotation)
                    state.logs.append("annotation stored in-memory (no session dir)")
                continue
            if key == "f":
                _cycle_reasoning_filter(state)
                continue
            if key == "r":
                args.reasoning_redact = "off" if args.reasoning_redact == "on" else "on"
                state.logs.append(f"reasoning redaction={args.reasoning_redact}")
                continue

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
                    kwargs={"stop": stop, "proc": proc, "out_q": in_q, "local_drops": reader_local_drops},
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
            state.apply(msg)
            if (not using_preloaded_trace_index) and trace_builder.ingest(msg):
                state.set_traces(trace_builder.traces())
            if capture_writer is not None:
                keep_writing = capture_writer.write(msg)
                if not keep_writing:
                    capture_writer.close()
                    capture_writer = None
                    state.logs.append("local capture stopped: max_seconds_elapsed")
                    if save_session_dir:
                        state.quality_report = load_quality_report(save_session_dir)
                        state.integrity_report = verify_integrity_report(save_session_dir)
                        for row in load_annotations(save_session_dir, limit=120):
                            state.add_annotation(dict(row))
                        _apply_quality_gate()

        if state.query_text and (now - last_query_refresh_s) >= 1.0:
            last_query_refresh_s = now
            if args.index_mode == "off":
                state.query_rows = _build_in_memory_query_rows(state, limit=int(args.query_limit))
                state.query_note = "memory query (reasoning + trace events; index disabled)"
            else:
                lookup_root = _resolve_lookup_root()

                if lookup_root:
                    try:
                        query_out = query_session_db(
                            lookup_root,
                            query=state.query_text,
                            limit=int(args.query_limit),
                        )
                        event_rows = [
                            {
                                "kind": "event",
                                "id": row.get("seq"),
                                "trace_id": row.get("trace_id"),
                                "req_id": row.get("req_id"),
                                "source": row.get("source"),
                                "status": row.get("status"),
                                "severity": row.get("severity"),
                                "summary": row.get("snippet"),
                            }
                            for row in query_out.get("events", [])
                        ]
                        joined_rows = [
                            {
                                "kind": "joined",
                                "id": row.get("row_id"),
                                "trace_id": row.get("trace_id"),
                                "req_id": row.get("req_id"),
                                "source": row.get("component") or row.get("source"),
                                "status": row.get("status"),
                                "severity": row.get("severity"),
                                "summary": row.get("snippet") or row.get("output_preview"),
                            }
                            for row in query_out.get("joined", [])
                        ]
                        state.query_rows = (event_rows + joined_rows)[: max(1, int(args.query_limit))]
                        state.query_note = (
                            f"sqlite events={len(query_out.get('events', []))} "
                            f"joined={len(query_out.get('joined', []))}"
                        )
                    except Exception as exc:
                        state.query_rows = _build_in_memory_query_rows(state, limit=int(args.query_limit))
                        state.query_note = f"sqlite query fallback: {exc!r}"
                else:
                    state.query_rows = _build_in_memory_query_rows(state, limit=int(args.query_limit))
                    state.query_note = "memory query (reasoning + trace events; session db unavailable)"

        if (now - last_timeline_refresh_s) >= 1.0:
            selected = state.selected_trace()
            current_trace_id = selected.trace_id if selected is not None else ""
            timeline_limit = max(8, int(args.query_limit) * 4)
            timeline_out = load_timeline_rows(
                session_root=_resolve_timeline_session_root(),
                session_db_path=session_db_path,
                query_text=state.query_text,
                selected_trace_id=current_trace_id,
                limit=timeline_limit,
                index_mode=str(args.index_mode or "auto"),
                required_schema_version=int(TELEMETRY_SCHEMA_VERSION_V3),
            )
            state.set_timeline_rows(
                timeline_out.rows,
                total_count=timeline_out.total_count,
                source=timeline_out.source,
                note=timeline_out.note,
                unavailable_reason=timeline_out.unavailable_reason,
            )
            last_timeline_refresh_s = now

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
    if save_session_dir:
        loaded_quality = load_quality_report(save_session_dir)
        if loaded_quality is not None:
            state.quality_report = loaded_quality
            _apply_quality_gate()
        state.integrity_report = verify_integrity_report(save_session_dir)
    if args.curate_on_exit:
        curation_root = replay_dir if replay_mode else save_session_dir
        curation_result = _run_curation_export(session_dir=curation_root, profile=str(args.curate_profile))
        if bool(curation_result.get("ok")):
            state.logs.append(
                "curation export ok: "
                f"profile={curation_result.get('profile')} rows={curation_result.get('row_count')} "
                f"annotated={curation_result.get('annotated_row_count')} "
                f"path={curation_result.get('output_path')}"
            )
        else:
            state.warnings.append(f"curation export failed: {curation_result.get('error')}")
    if replay_reader is not None and replay_reader.decode_errors > 0:
        state.warnings.append(f"replay decode errors skipped={replay_reader.decode_errors}")
    key_reader.stop()

    if video_window is not None:
        try:
            video_window.root.destroy()
        except Exception:
            pass

    print("\n")
    if args.curate_on_exit and curation_result is not None:
        if bool(curation_result.get("ok")):
            print(
                "curation_export: "
                f"profile={curation_result.get('profile')} rows={curation_result.get('row_count')} "
                f"annotated={curation_result.get('annotated_row_count')} "
                f"output={curation_result.get('output_path')}"
            )
        else:
            print(f"curation_export_error: {curation_result.get('error')}")
    final_exit = 0
    if args.curate_on_exit and (curation_result is None or (not bool(curation_result.get("ok")))):
        final_exit = 3
    elif str(args.quality_gate) == "strict" and state.quality_gate_passed is not True:
        final_exit = 2

    summary_root = ""
    if save_session_dir:
        summary_root = save_session_dir
    elif replay_mode and args.curate_on_exit:
        summary_root = replay_dir
    if summary_root:
        try:
            summary = _build_viewer_summary(
                mode=str(args.mode or "live"),
                state=state,
                query_text=str(args.query or ""),
                quality_gate=str(args.quality_gate or "warn"),
                curation_result=curation_result,
                exit_code=final_exit,
            )
        except Exception as exc:
            print(f"viewer_summary_error: {exc!r}")
        else:
            try:
                summary_path = _write_viewer_summary(session_dir=summary_root, summary=summary)
                print(f"viewer_summary: {summary_path}")
            except Exception as exc:
                print(f"viewer_summary_error: {exc!r}")
            try:
                runs_path = _append_viewer_run(session_dir=summary_root, summary=summary)
                print(f"viewer_runs: {runs_path}")
            except Exception as exc:
                print(f"viewer_runs_error: {exc!r}")

    return final_exit


if __name__ == "__main__":
    raise SystemExit(main())
