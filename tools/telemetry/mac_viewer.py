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
from typing import Any, Deque, Dict, List, Optional

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
from tools.telemetry.doctor import build_doctor_report, load_doctor_report, write_doctor_report
from tools.telemetry.filters import matches_field_filters, parse_field_filters
from tools.telemetry.incident import build_incident_report, load_incident_report, write_incident_markdown, write_incident_report
from tools.telemetry.insights import build_improvement_report, load_improvement_report, write_improvement_report
from tools.telemetry.packs import apply_pack_overrides, list_packs, resolve_packs
from tools.telemetry.protocol import decode_message
from tools.telemetry.lamp_viz import draw_lamp_panel
from tools.telemetry.quality import evaluate_quality_gate, load_quality_report
from tools.telemetry.reasoning import ReasoningEvent, format_reasoning_snippet, normalize_reasoning_message
from tools.telemetry.replay import SessionReplayReader
from tools.telemetry.scoreboard import DEFAULT_SCOREBOARD_PATH, load_scoreboard
from tools.telemetry.schema_v3 import TELEMETRY_SCHEMA_VERSION_V3
from tools.telemetry.storage_sqlite import build_session_db, query_session_db, resolve_session_db_path
from tools.telemetry.story import build_reasoning_story, build_trace_story
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
    quality_report: Optional[Dict[str, Any]] = None
    improvement_report: Optional[Dict[str, Any]] = None
    doctor_report: Optional[Dict[str, Any]] = None
    incident_report: Optional[Dict[str, Any]] = None
    quality_gate_note: str = ""
    quality_gate_passed: Optional[bool] = None
    doctor_gate_note: str = ""
    doctor_gate_passed: Optional[bool] = None
    active_alerts: List[str] = field(default_factory=list)
    recent_event_samples: Deque[tuple[float, str]] = field(default_factory=lambda: collections.deque(maxlen=4000))
    recent_query_events: Deque[Dict[str, Any]] = field(default_factory=lambda: collections.deque(maxlen=1500))
    query_event_counter: int = 0
    alert_history: Deque[str] = field(default_factory=lambda: collections.deque(maxlen=24))
    scoreboard: Optional[Dict[str, Any]] = None
    query_slice_exports: int = 0

    def configure_panels(self, panels: List[str], *, focus_panel: str = "") -> None:
        self.active_panels = list(panels)
        if focus_panel and focus_panel in self.active_panels:
            self.focus_panel = focus_panel
        elif self.focus_panel not in self.active_panels:
            self.focus_panel = self.active_panels[0] if self.active_panels else "summary"

    def apply(self, msg: Dict[str, Any]) -> None:
        source = str(msg.get("source", "unknown"))
        now = time.time()
        self.last_event_wall_s = now
        self.event_counts[source] = self.event_counts.get(source, 0) + 1
        self.recent_event_samples.append((now, source))

        payload = msg.get("payload", {})
        if not isinstance(payload, dict):
            return
        self._ingest_reasoning_event(msg)
        self._record_query_event(msg, payload)

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

    def event_rates(self, *, now_wall_s: float, window_s: float) -> tuple[float, Dict[str, float]]:
        window = max(0.5, float(window_s))
        cutoff = now_wall_s - window
        per_source: Dict[str, int] = {}
        total = 0
        for ts, source in reversed(self.recent_event_samples):
            if ts < cutoff:
                break
            total += 1
            per_source[source] = per_source.get(source, 0) + 1
        rates = {source: (count / window) for source, count in per_source.items()}
        return (total / window), rates

    def _record_query_event(self, msg: Dict[str, Any], payload: Dict[str, Any]) -> None:
        source = str(msg.get("source", "unknown"))
        if source == "video_frame":
            return
        ts_wall_s_raw = msg.get("ts_wall_s")
        ts_wall_s = float(ts_wall_s_raw) if isinstance(ts_wall_s_raw, (int, float)) else time.time()
        level = str(msg.get("level", "") or "").lower()
        severity = "info"
        if level in {"warning", "warn"}:
            severity = "warning"
        elif level == "error":
            severity = "error"
        if "error" in payload:
            severity = "error"

        req_id: Optional[int] = None
        phase = ""
        status = ""
        snippet = ""
        trace_id = None

        if source == "timeline_log":
            data = payload.get("data")
            if isinstance(data, dict):
                phase = str(data.get("type") or "")
                dp = data.get("payload")
                if isinstance(dp, dict):
                    status = str(dp.get("status") or "")
                    rid = dp.get("request_id")
                    if rid is None:
                        rid = dp.get("req_id")
                    if rid is None:
                        rid = dp.get("id")
                    if isinstance(rid, int):
                        req_id = rid
                    snippet = str(dp.get("reasoning") or dp.get("message") or dp.get("detail") or "")
        elif source == "actions_log":
            data = payload.get("data")
            if isinstance(data, dict):
                phase = "action_plan"
                status = str(data.get("status") or "ok")
                rid = data.get("request_id")
                if rid is None:
                    rid = data.get("req_id")
                if isinstance(rid, int):
                    req_id = rid
                snippet = str(data.get("explanation") or data.get("primitive") or "")
        elif source == "journal":
            line = payload.get("line")
            if isinstance(line, str):
                snippet = line
            phase = "journal"
        elif source == "agent":
            status = "error" if "error" in payload else "ok"
            snippet = str(payload.get("error") or payload.get("detail") or "")
        else:
            snippet = str(payload.get("line") or payload.get("status") or payload.get("message") or "")

        self.query_event_counter += 1
        self.recent_query_events.append(
            {
                "kind": "event",
                "id": self.query_event_counter,
                "ts_wall_s": ts_wall_s,
                "source": source,
                "trace_id": trace_id,
                "req_id": req_id,
                "phase": phase,
                "status": status,
                "severity": severity,
                "summary": format_reasoning_snippet(snippet, max_chars=220, redact=False),
            }
        )


@dataclass
class _VideoWindow:
    root: Any
    container: Any
    label: Any
    photo: Any = None
    last_frame_id: int = -1


PANEL_PRESETS: Dict[str, List[str]] = {
    "1": [
        "summary",
        "alerts",
        "quality",
        "doctor",
        "incident",
        "insights",
        "story",
        "query",
        "scoreboard",
        "trace_list",
        "trace_detail",
        "reasoning_stream",
        "request_detail",
        "reasoning_health",
        "video",
    ],
    "2": [
        "summary",
        "alerts",
        "throughput",
        "quality",
        "doctor",
        "incident",
        "insights",
        "story",
        "query",
        "scoreboard",
        "trace_list",
        "trace_detail",
        "reasoning_stream",
        "request_detail",
        "logs",
        "warnings",
        "transport",
    ],
    "3": [
        "summary",
        "alerts",
        "throughput",
        "doctor",
        "incident",
        "insights",
        "story",
        "scoreboard",
        "trace_list",
        "trace_detail",
        "video",
        "perception",
        "action",
        "system",
        "events",
    ],
}


def _apply_panel_preset(state: DashboardState, args: argparse.Namespace, key: str) -> None:
    panels = PANEL_PRESETS.get(str(key))
    if not panels:
        return
    args.panel = list(panels)
    state.configure_panels(args.panel, focus_panel=state.focus_panel)
    state.logs.append(f"panel preset {key} applied")


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
        "--trace-match-window-s",
        str(float(args.trace_match_window_s)),
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
                "--capture-runbook",
                str(args.runbook),
                "--capture-scoreboard-path",
                str(args.scoreboard_path),
            ]
        )
        for tag in args.scenario_tag:
            agent_args.extend(["--capture-scenario-tag", str(tag)])
        for tag in args.goal_tag:
            agent_args.extend(["--capture-goal-tag", str(tag)])
        for session in args.golden_session:
            agent_args.extend(["--capture-golden-session", str(session)])
        if args.no_scoreboard_update:
            agent_args.append("--no-capture-scoreboard")

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


def _focus_prefix(state: DashboardState, panel: str) -> str:
    if state.focus_panel == panel:
        return "> "
    return "  "


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = int(round((max(0.0, min(100.0, pct)) / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def _split_query_tokens(query: str) -> List[str]:
    try:
        return [tok for tok in shlex.split(str(query or "")) if tok]
    except ValueError:
        return [tok for tok in str(query or "").split() if tok]


def _parse_query_since_s(query: str) -> Optional[float]:
    values: List[float] = []
    for token in _split_query_tokens(query):
        if ":" not in token:
            continue
        key, raw = token.split(":", 1)
        if key.strip().lower() != "since":
            continue
        text = raw.strip().lower()
        if not text:
            continue
        scale = 1.0
        if text.endswith("ms"):
            scale = 0.001
            text = text[:-2]
        elif text.endswith("s"):
            scale = 1.0
            text = text[:-1]
        elif text.endswith("m"):
            scale = 60.0
            text = text[:-1]
        elif text.endswith("h"):
            scale = 3600.0
            text = text[:-1]
        try:
            val = float(text)
        except ValueError:
            continue
        if val > 0.0:
            values.append(val * scale)
    if not values:
        return None
    return min(values)


def _parse_query_keyed_values(query: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {
        "source": [],
        "severity": [],
        "status": [],
        "phase": [],
        "req": [],
        "trace": [],
        "kind": [],
        "since": [],
    }
    for token in _split_query_tokens(query):
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        key_l = key.strip().lower()
        value = value.strip()
        if key_l in out and value:
            out[key_l].append(value)
    return out


def _query_text_terms(query: str) -> List[str]:
    keyed = {"source", "severity", "status", "phase", "req", "trace", "since", "kind"}
    out: List[str] = []
    for token in _split_query_tokens(query):
        if ":" in token:
            key, value = token.split(":", 1)
            if key.strip().lower() in keyed:
                if key.strip().lower() not in {"since"} and not value.strip():
                    continue
                if key.strip().lower() in {"source", "severity", "status", "phase", "req", "trace", "kind", "since"}:
                    continue
        out.append(token)
    return out


def _query_text_match(query: str, text: str) -> bool:
    tokens = _query_text_terms(query)
    if not tokens:
        return True
    hay = str(text or "").lower()
    return all(token.lower() in hay for token in tokens)


def _build_in_memory_query_rows(state: DashboardState, *, limit: int, now_wall_s: float) -> List[Dict[str, Any]]:
    lim = max(1, int(limit))
    since_s = _parse_query_since_s(state.query_text)
    keyed = _parse_query_keyed_values(state.query_text)
    kind_values = {str(v).lower() for v in keyed.get("kind", []) if str(v).strip()}
    want_events = (not kind_values) or bool(kind_values & {"event", "events"})
    want_reasoning = (not kind_values) or bool(kind_values & {"reasoning", "reason"})
    want_traces = (not kind_values) or bool(kind_values & {"trace", "traces"})
    req_values = {int(v) for v in keyed.get("req", []) if str(v).isdigit()}
    status_values = {str(v).lower() for v in keyed.get("status", [])}
    severity_values = {str(v).lower() for v in keyed.get("severity", [])}
    phase_values = {str(v).lower() for v in keyed.get("phase", [])}
    source_values = {str(v).lower() for v in keyed.get("source", [])}
    trace_values = [str(v).lower() for v in keyed.get("trace", []) if str(v).strip()]
    rows: List[Dict[str, Any]] = []

    if want_events:
        for item in reversed(state.recent_query_events):
            ts = item.get("ts_wall_s")
            if since_s is not None and isinstance(ts, (int, float)) and (now_wall_s - float(ts)) > since_s:
                continue
            req = item.get("req_id")
            if req_values and (not isinstance(req, int) or req not in req_values):
                continue
            status = str(item.get("status") or "").lower()
            if status_values and status not in status_values:
                continue
            sev = str(item.get("severity") or "").lower()
            if severity_values and sev not in severity_values:
                continue
            phase = str(item.get("phase") or "").lower()
            if phase_values and phase not in phase_values:
                continue
            src = str(item.get("source") or "").lower()
            if source_values and src not in source_values:
                continue
            trace_id = str(item.get("trace_id") or "").lower()
            if trace_values and not any(value in trace_id for value in trace_values):
                continue
            text = " ".join(
                [
                    str(item.get("source") or ""),
                    str(item.get("phase") or ""),
                    str(item.get("status") or ""),
                    str(item.get("severity") or ""),
                    str(item.get("req_id") if item.get("req_id") is not None else ""),
                    str(item.get("summary") or ""),
                ]
            )
            if not _query_text_match(state.query_text, text):
                continue
            rows.append(dict(item))

    if want_reasoning:
        for seq, event in reversed(state._iter_reasoning_with_seq()):
            if since_s is not None and event.ts_wall_s is not None and (now_wall_s - float(event.ts_wall_s)) > since_s:
                continue
            if req_values and (event.req_id is None or int(event.req_id) not in req_values):
                continue
            if status_values and str(event.status or "").lower() not in status_values:
                continue
            if severity_values and str(event.severity or "").lower() not in severity_values:
                continue
            if phase_values and str(event.phase or "").lower() not in phase_values:
                continue
            if source_values and str(event.source or "").lower() not in source_values:
                continue
            if trace_values:
                continue
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
            rows.append(
                {
                    "kind": "reasoning",
                    "id": seq,
                    "ts_wall_s": event.ts_wall_s,
                    "source": event.source,
                    "trace_id": None,
                    "req_id": event.req_id,
                    "phase": event.phase,
                    "status": event.status,
                    "severity": event.severity,
                    "summary": format_reasoning_snippet(
                        event.snippet,
                        max_chars=120,
                        redact=False,
                    ),
                }
            )

    if want_traces:
        for trace in state.trace_records:
            ts = trace.end_ts_wall_s if trace.end_ts_wall_s is not None else trace.start_ts_wall_s
            if since_s is not None and ts is not None and (now_wall_s - float(ts)) > since_s:
                continue
            if req_values and (trace.req_id is None or int(trace.req_id) not in req_values):
                continue
            if status_values and str(trace.status or "").lower() not in status_values:
                continue
            if severity_values and str(trace.severity or "").lower() not in severity_values:
                continue
            if source_values and "trace" not in source_values:
                continue
            trace_id_text = str(trace.trace_id or "")
            if trace_values and not any(value in trace_id_text.lower() for value in trace_values):
                continue
            if phase_values:
                continue
            text = " ".join(
                [
                    trace_id_text,
                    str(trace.req_id if trace.req_id is not None else ""),
                    str(trace.status or ""),
                    str(trace.severity or ""),
                    str(trace.summary or ""),
                ]
            )
            if not _query_text_match(state.query_text, text):
                continue
            rows.append(
                {
                    "kind": "trace",
                    "id": trace.trace_id,
                    "ts_wall_s": ts,
                    "source": "trace",
                    "trace_id": trace.trace_id,
                    "req_id": trace.req_id,
                    "phase": "",
                    "status": trace.status,
                    "severity": trace.severity,
                    "summary": trace.summary,
                }
            )
    kind_rank = {"event": 0, "reasoning": 1, "trace": 2}
    rows.sort(
        key=lambda item: (
            -float(item.get("ts_wall_s") or 0.0),
            kind_rank.get(str(item.get("kind")), 99),
            str(item.get("id")),
        )
    )
    if len(rows) > lim:
        rows = rows[:lim]
    return rows


def _rows_from_query_out(query_out: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in query_out.get("events", []):
        rows.append(
            {
                "kind": "event",
                "id": row.get("seq"),
                "ts_wall_s": row.get("ts_wall_s"),
                "source": row.get("source"),
                "trace_id": row.get("trace_id"),
                "req_id": row.get("req_id"),
                "status": row.get("status"),
                "severity": row.get("severity"),
                "summary": row.get("snippet"),
            }
        )
    for row in query_out.get("reasoning", []):
        rows.append(
            {
                "kind": "reasoning",
                "id": row.get("event_index"),
                "ts_wall_s": row.get("ts_wall_s"),
                "source": "reasoning",
                "trace_id": None,
                "req_id": row.get("req_id"),
                "status": row.get("status"),
                "severity": row.get("severity"),
                "summary": row.get("snippet"),
            }
        )
    for row in query_out.get("traces", []):
        rows.append(
            {
                "kind": "trace",
                "id": row.get("trace_id"),
                "ts_wall_s": None,
                "source": "trace",
                "trace_id": row.get("trace_id"),
                "req_id": row.get("req_id"),
                "status": row.get("status"),
                "severity": row.get("severity"),
                "summary": row.get("summary"),
            }
        )
    kind_rank = {"event": 0, "reasoning": 1, "trace": 2}
    rows.sort(
        key=lambda item: (
            -float(item.get("ts_wall_s") or 0.0),
            kind_rank.get(str(item.get("kind")), 99),
            str(item.get("id")),
        )
    )
    return rows[: max(1, int(limit))]


def _query_note_from_query_out(query_out: Dict[str, Any]) -> str:
    counts = query_out.get("counts")
    if isinstance(counts, dict):
        return (
            f"sqlite matches events={counts.get('events', 0)} "
            f"reasoning={counts.get('reasoning', 0)} traces={counts.get('traces', 0)}"
        )
    return (
        f"sqlite matches events={len(query_out.get('events', []))} "
        f"reasoning={len(query_out.get('reasoning', []))} traces={len(query_out.get('traces', []))}"
    )


def _write_query_export(path: str, *, query: str, note: str, rows: List[Dict[str, Any]]) -> None:
    payload = {
        "exported_at_wall_s": time.time(),
        "query": str(query),
        "note": str(note),
        "row_count": len(rows),
        "rows": rows,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"), ensure_ascii=True)


def _write_query_slice_export(path: str, *, query: str, rows: List[Dict[str, Any]]) -> int:
    target = str(path).strip()
    if not target:
        raise ValueError("query slice export path is empty")
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    written = 0
    with open(target, "w", encoding="utf-8") as fh:
        for row in rows:
            record = {
                "label": "query_slice",
                "query": str(query),
                "kind": row.get("kind"),
                "trace_id": row.get("trace_id"),
                "req_id": row.get("req_id"),
                "source": row.get("source"),
                "status": row.get("status"),
                "severity": row.get("severity"),
                "summary": row.get("summary"),
                "row": row,
            }
            fh.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True))
            fh.write("\n")
            written += 1
    return written


def _apply_alert_policy(args: argparse.Namespace, *, defaults: Optional[argparse.Namespace] = None) -> None:
    policy = str(getattr(args, "alert_policy", "custom") or "custom").lower()
    if policy == "custom":
        return
    presets: Dict[str, Dict[str, float]] = {
        "demo": {
            "alert_stale_s": 4.0,
            "alert_heartbeat_s": 2.0,
            "alert_video_idle_s": 3.0,
            "alert_dropped_events": 1.0,
            "alert_warning_count": 3.0,
            "alert_trace_grace_s": 10.0,
            "alert_min_events_per_s": 0.3,
        },
        "training": {
            "alert_stale_s": 12.0,
            "alert_heartbeat_s": 6.0,
            "alert_video_idle_s": 8.0,
            "alert_dropped_events": 4.0,
            "alert_warning_count": 6.0,
            "alert_trace_grace_s": 25.0,
            "alert_min_events_per_s": 0.0,
        },
        "debug": {
            "alert_stale_s": 30.0,
            "alert_heartbeat_s": 15.0,
            "alert_video_idle_s": 20.0,
            "alert_dropped_events": 15.0,
            "alert_warning_count": 12.0,
            "alert_trace_grace_s": 60.0,
            "alert_min_events_per_s": 0.0,
        },
    }
    selected = presets.get(policy)
    if not selected:
        return
    for key, value in selected.items():
        if defaults is not None and getattr(args, key, None) != getattr(defaults, key, None):
            continue
        current = getattr(args, key, None)
        if isinstance(current, int):
            setattr(args, key, int(value))
        else:
            setattr(args, key, float(value))


def _collect_alerts(state: DashboardState, args: argparse.Namespace, now_wall_s: float) -> List[str]:
    alerts: List[str] = []
    if state.quality_gate_passed is False:
        alerts.append(f"quality_gate: {state.quality_gate_note}")
    if state.doctor_gate_passed is False:
        alerts.append(f"doctor_gate: {state.doctor_gate_note}")
    stale_age = None if state.last_rx_wall_s is None else (now_wall_s - float(state.last_rx_wall_s))
    if state.connected and stale_age is not None and stale_age > float(args.alert_stale_s):
        alerts.append(f"stream_stale: last_rx_age={stale_age:.1f}s")
    hb_age = None if state.last_agent_wall_s is None else (now_wall_s - float(state.last_agent_wall_s))
    if state.connected and hb_age is not None and hb_age > float(args.alert_heartbeat_s):
        alerts.append(f"agent_heartbeat_stale: age={hb_age:.1f}s")
    if (not args.no_video) and state.last_video_wall_s is not None:
        video_age = now_wall_s - float(state.last_video_wall_s)
        if video_age > float(args.alert_video_idle_s):
            alerts.append(f"video_idle: age={video_age:.1f}s")
    if int(state.dropped_events_reported) >= int(args.alert_dropped_events):
        alerts.append(f"dropped_events: {state.dropped_events_reported}")
    if len(state.warnings) >= int(args.alert_warning_count):
        alerts.append(f"warning_burst: warnings={len(state.warnings)}")
    recent_eps, _ = state.event_rates(now_wall_s=now_wall_s, window_s=float(args.rate_window_s))
    if state.connected and float(args.alert_min_events_per_s) > 0.0 and recent_eps < float(args.alert_min_events_per_s):
        alerts.append(f"throughput_low: eps={recent_eps:.2f} (<{float(args.alert_min_events_per_s):.2f})")
    if state.connected and (now_wall_s - state.started_wall_s) > float(args.alert_trace_grace_s) and not state.trace_records:
        alerts.append("trace_gap: no traces observed yet")
    return alerts


def _evaluate_doctor_gate(report: Optional[Dict[str, Any]], mode: str) -> tuple[Optional[bool], str]:
    gate = str(mode or "off")
    if gate == "off":
        return None, "doctor gate disabled"
    readiness = report.get("readiness") if isinstance(report, dict) else None
    readiness = readiness if isinstance(readiness, dict) else {}
    grade = str(readiness.get("grade") or "unknown").lower()
    score = readiness.get("score")
    label = f"doctor grade={grade} score={score}"
    if gate == "warn":
        return grade != "fail", label
    if gate == "strict":
        return grade == "pass", label
    return None, f"unknown doctor gate mode: {gate}"


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


def _render_request_detail(lines: List[str], state: DashboardState, args: argparse.Namespace) -> None:
    lines.append(f"{_focus_prefix(state, 'request_detail')}Request Detail")
    selected = state.selected_reasoning_event(slow_ms=float(args.reasoning_slow_ms))
    if selected is None:
        lines.append("  no selection")
        lines.append("")
        return
    seq, event = selected
    lines.append(
        f"  seq={seq} source={event.source} req_id={event.req_id} phase={event.phase} status={event.status or '-'}"
    )
    lines.append(
        "  "
        f"latency_ms={event.latency_ms} primitive={event.primitive} confidence={event.confidence} "
        f"target_zone={event.target_zone}"
    )
    lines.append(f"  model={event.model} provider={event.provider} severity={event.severity}")
    snippet = format_reasoning_snippet(
        event.snippet,
        max_chars=int(args.reasoning_snippet_max_chars),
        redact=str(args.reasoning_redact) == "on",
    )
    if snippet:
        lines.append(f"  snippet={_shorten(snippet, 240)}")
    lines.append("")


def _render_reasoning_health(lines: List[str], state: DashboardState, args: argparse.Namespace, now_wall_s: float) -> None:
    lines.append(f"{_focus_prefix(state, 'reasoning_health')}Reasoning Health")
    window_s = max(1.0, float(args.reasoning_kpi_window_s))
    recent: List[ReasoningEvent] = []
    for _, event in state.filtered_reasoning_events(slow_ms=float(args.reasoning_slow_ms)):
        if event.ts_wall_s is None:
            continue
        if (now_wall_s - float(event.ts_wall_s)) <= window_s:
            recent.append(event)

    if not recent:
        lines.append(f"  no events in last {window_s:.0f}s")
        lines.append("")
        return
    total = len(recent)
    error_count = sum(1 for ev in recent if ev.severity == "error")
    parse_fail = sum(1 for ev in recent if "parse_fail" in ev.phase)
    no_content = sum(1 for ev in recent if (ev.status or "").lower() == "no_content")
    stale = sum(1 for ev in recent if "stale" in ev.phase or "stale" in (ev.status or "").lower())
    latencies = [float(ev.latency_ms) for ev in recent if ev.latency_ms is not None]
    p50 = _percentile(latencies, 50.0)
    p95 = _percentile(latencies, 95.0)
    ok_rate = (100.0 * max(0, total - error_count) / total) if total > 0 else 0.0
    lines.append(f"  window={window_s:.0f}s events={total} ok_rate={ok_rate:.1f}% errors={error_count}")
    lines.append(f"  parse_fail={parse_fail} no_content={no_content} stale_markers={stale}")
    lines.append(f"  latency_ms p50={p50:.0f} p95={p95:.0f}" if p50 is not None and p95 is not None else "  latency_ms n/a")
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


def _render_doctor_panel(lines: List[str], state: DashboardState) -> None:
    lines.append(f"{_focus_prefix(state, 'doctor')}Doctor")
    report = state.doctor_report
    if report is None:
        lines.append("  no doctor report loaded")
        if state.doctor_gate_note:
            lines.append(f"  gate={state.doctor_gate_note}")
        lines.append("")
        return
    readiness = report.get("readiness")
    readiness = readiness if isinstance(readiness, dict) else {}
    lines.append(f"  grade={readiness.get('grade')} score={readiness.get('score')}")
    summary = report.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    lines.append(
        "  "
        f"errors={summary.get('error_count', 0)} warnings={summary.get('warning_count', 0)}"
    )
    checks = report.get("checks")
    checks = checks if isinstance(checks, dict) else {}
    lines.append(
        "  "
        f"events(lines)={checks.get('event_count_lines')} indexed={checks.get('event_count_indexed')} "
        f"invalid_json={checks.get('invalid_json_count')}"
    )
    if state.doctor_gate_note:
        lines.append(f"  gate={state.doctor_gate_note}")
    issues = report.get("issues")
    if isinstance(issues, list) and issues:
        lines.append("  top_issues:")
        for item in issues[:2]:
            if not isinstance(item, dict):
                continue
            lines.append(f"    [{item.get('severity')}] {item.get('code')}")
    lines.append("")


def _render_incident_panel(lines: List[str], state: DashboardState, args: argparse.Namespace) -> None:
    lines.append(f"{_focus_prefix(state, 'incident')}Incident")
    report = state.incident_report
    if report is None:
        lines.append("  no incident report loaded")
        lines.append("")
        return
    lines.append(f"  severity={report.get('severity')} title={_shorten(str(report.get('title') or ''), 96)}")
    summary = report.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    lines.append(
        "  "
        f"parse_fail={summary.get('parse_fail_count')} timeout={summary.get('timeout_count')} "
        f"traces={summary.get('trace_count')} issues={summary.get('error_issue_count', 0)}/{summary.get('warning_issue_count', 0)}"
    )
    issues = report.get("issues")
    if isinstance(issues, list) and issues:
        lines.append("  issues:")
        for item in issues[: max(1, int(args.max_log_lines // 2))]:
            if not isinstance(item, dict):
                continue
            lines.append(f"    [{item.get('severity')}] {item.get('code')}")
    recs = report.get("recommendations")
    if isinstance(recs, list) and recs:
        lines.append("  recommendations:")
        for rec in recs[:2]:
            lines.append(f"    {_shorten(str(rec), 140)}")
    lines.append("")


def _render_insights_panel(lines: List[str], state: DashboardState, args: argparse.Namespace) -> None:
    lines.append(f"{_focus_prefix(state, 'insights')}Insights")
    report = state.improvement_report
    if report is None:
        lines.append("  no improvement report loaded")
        lines.append("")
        return
    summary = report.get("summary")
    if isinstance(summary, dict):
        lines.append(
            "  "
            f"reasoning={summary.get('reasoning_count')} traces={summary.get('trace_count')} "
            f"parse_fail={summary.get('parse_fail_count')} timeout={summary.get('timeout_count')} slow={summary.get('slow_count')}"
        )
    recs = report.get("recommendations")
    if not isinstance(recs, list) or not recs:
        lines.append("  no recommendations")
        lines.append("")
        return
    max_rows = max(1, int(args.insight_max_recommendations))
    for item in recs[:max_rows]:
        if not isinstance(item, dict):
            continue
        prio = str(item.get("priority") or "info")
        title = str(item.get("title") or "recommendation")
        why = str(item.get("why") or "")
        action = str(item.get("action") or "")
        lines.append(f"  [{prio}] {title}")
        if why:
            lines.append(f"    why: {_shorten(why, 140)}")
        if action:
            lines.append(f"    next: {_shorten(action, 140)}")
    golden = report.get("golden_comparison")
    if isinstance(golden, dict):
        delta = golden.get("delta")
        baseline = golden.get("baseline")
        lines.append(
            f"  golden_sessions={golden.get('golden_session_count')} "
            f"baseline_quality={(baseline or {}).get('quality_score')}"
        )
        if isinstance(delta, dict):
            lines.append(
                "  "
                f"delta quality={delta.get('quality_score')} "
                f"parse_fail={delta.get('parse_fail_rate')} timeout={delta.get('timeout_rate')}"
            )
    fingerprints = report.get("failure_fingerprints")
    if isinstance(fingerprints, list) and fingerprints:
        lines.append("  fingerprints:")
        for row in fingerprints[:2]:
            if not isinstance(row, dict):
                continue
            lines.append(f"    {row.get('fingerprint')}: {row.get('count')}")
    scenario_tags = report.get("scenario_tags")
    if isinstance(scenario_tags, list) and scenario_tags:
        lines.append(f"  scenario_tags={','.join(str(x) for x in scenario_tags[:6])}")
    lines.append("")


def _render_story_panel(lines: List[str], state: DashboardState, args: argparse.Namespace) -> None:
    lines.append(f"{_focus_prefix(state, 'story')}Story")
    max_rows = max(3, int(args.story_max_rows))
    story_lines = build_trace_story(state.selected_trace(), max_events=max_rows)
    for text in story_lines[: max_rows + 4]:
        lines.append(f"  {_shorten(text, 160)}")
    if state.query_rows:
        lines.append("  query_slice:")
        for text in build_reasoning_story(state.query_rows, max_rows=max_rows)[:max_rows]:
            lines.append(f"    {_shorten(text, 148)}")
    lines.append("")


def _render_scoreboard_panel(lines: List[str], state: DashboardState, args: argparse.Namespace) -> None:
    lines.append(f"{_focus_prefix(state, 'scoreboard')}Scoreboard")
    board = state.scoreboard
    if not isinstance(board, dict):
        lines.append("  no scoreboard loaded")
        lines.append("")
        return
    sessions = board.get("sessions")
    trend = board.get("trend")
    if isinstance(trend, dict):
        lines.append(
            f"  trend quality_delta={trend.get('quality_delta')} "
            f"parse_fail_delta={trend.get('parse_fail_delta')} timeout_delta={trend.get('timeout_delta')}"
        )
    if not isinstance(sessions, list) or not sessions:
        lines.append("  no sessions")
        lines.append("")
        return
    max_rows = max(2, int(args.max_log_lines))
    for row in list(sessions)[-max_rows:]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "  "
            f"{row.get('session_name')} q={row.get('quality_score')} "
            f"pf={row.get('parse_fail_rate')} to={row.get('timeout_rate')} lbl={row.get('weak_label_count')}"
        )
    lines.append("")


def _render_alerts_panel(lines: List[str], state: DashboardState, args: argparse.Namespace) -> None:
    lines.append(f"{_focus_prefix(state, 'alerts')}Alerts")
    if not state.active_alerts:
        lines.append("  none")
    else:
        max_rows = max(4, int(args.max_log_lines))
        for item in state.active_alerts[:max_rows]:
            lines.append(f"  ! {_shorten(item, 160)}")
    if state.alert_history:
        lines.append("  recent:")
        for item in list(state.alert_history)[-max(2, int(args.max_log_lines // 2)) :]:
            lines.append(f"    {_shorten(item, 150)}")
    lines.append("")


def _render_throughput_panel(lines: List[str], state: DashboardState, args: argparse.Namespace, now_wall_s: float) -> None:
    lines.append(f"{_focus_prefix(state, 'throughput')}Throughput")
    uptime_s = max(0.1, now_wall_s - state.started_wall_s)
    lifetime_total = sum(int(v) for v in state.event_counts.values())
    lifetime_eps = lifetime_total / uptime_s
    recent_eps, rates = state.event_rates(now_wall_s=now_wall_s, window_s=float(args.rate_window_s))
    lines.append(
        f"  lifetime_events={lifetime_total} lifetime_eps={lifetime_eps:.2f} "
        f"recent_eps={recent_eps:.2f} window={float(args.rate_window_s):.1f}s"
    )
    if not rates:
        lines.append("  no recent events")
        lines.append("")
        return
    ranked = sorted(rates.items(), key=lambda item: (-item[1], item[0]))
    for source, eps in ranked[: max(4, int(args.max_log_lines))]:
        lines.append(f"  {source}: {eps:.2f} eps")
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
            f"  [{row.get('kind')}] id={row.get('id')} src={source or '-'} "
            f"trace={trace_id or '-'} req={req if req is not None else '-'} "
            f"status={status or '-'} sev={severity or '-'}"
        )
        if summary:
            lines.append(f"    {_shorten(str(summary), 150)}")
    lines.append("")


def _panel_enabled(args: argparse.Namespace, panel: str) -> bool:
    selected = getattr(args, "panel", None)
    if not selected:
        return True
    return panel in set(selected)


def _can_preload_trace_index(*, field_filters: List[Any]) -> bool:
    return len(field_filters) == 0


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
        lines.append(
            f"Reasoning filter={state.reasoning_filter_mode} | focus={state.focus_panel} | "
            f"hotkeys={'on' if state.key_reader_enabled else 'off'}"
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
        if state.improvement_report is not None:
            recs = state.improvement_report.get("recommendations")
            rec_count = len(recs) if isinstance(recs, list) else 0
            lines.append(f"Insights: recommendations={rec_count}")
        if state.doctor_report is not None:
            readiness = state.doctor_report.get("readiness")
            readiness = readiness if isinstance(readiness, dict) else {}
            lines.append(
                "Doctor: "
                f"grade={readiness.get('grade')} score={readiness.get('score')} gate={state.doctor_gate_note or 'n/a'}"
            )
        if state.incident_report is not None:
            lines.append(
                "Incident: "
                f"severity={state.incident_report.get('severity')} title={_shorten(str(state.incident_report.get('title') or ''), 72)}"
            )
        if state.query_text:
            lines.append(f"Query: '{state.query_text}' matches={len(state.query_rows)}")
        if state.query_slice_exports > 0:
            lines.append(f"Query slices exported: {state.query_slice_exports}")
        lines.append(f"Alerts: {len(state.active_alerts)}")
        recent_eps, _ = state.event_rates(now_wall_s=now_wall_s, window_s=float(args.rate_window_s))
        lines.append(f"Recent throughput: {recent_eps:.2f} eps over {float(args.rate_window_s):.1f}s")
        if isinstance(state.scoreboard, dict):
            trend = state.scoreboard.get("trend")
            if isinstance(trend, dict):
                lines.append(
                    "Scoreboard trend: "
                    f"quality_delta={trend.get('quality_delta')} parse_fail_delta={trend.get('parse_fail_delta')}"
                )
        lines.append("")

    if _panel_enabled(args, "reasoning_stream"):
        _render_reasoning_stream(lines, state, args)

    if _panel_enabled(args, "request_detail"):
        _render_request_detail(lines, state, args)

    if _panel_enabled(args, "reasoning_health"):
        _render_reasoning_health(lines, state, args, now_wall_s)

    if _panel_enabled(args, "trace_list"):
        _render_trace_list(lines, state, args)

    if _panel_enabled(args, "trace_detail"):
        _render_trace_detail(lines, state, args)

    if _panel_enabled(args, "alerts"):
        _render_alerts_panel(lines, state, args)

    if _panel_enabled(args, "throughput"):
        _render_throughput_panel(lines, state, args, now_wall_s)

    if _panel_enabled(args, "quality"):
        _render_quality_panel(lines, state)

    if _panel_enabled(args, "doctor"):
        _render_doctor_panel(lines, state)

    if _panel_enabled(args, "incident"):
        _render_incident_panel(lines, state, args)

    if _panel_enabled(args, "insights"):
        _render_insights_panel(lines, state, args)

    if _panel_enabled(args, "story"):
        _render_story_panel(lines, state, args)

    if _panel_enabled(args, "scoreboard"):
        _render_scoreboard_panel(lines, state, args)

    if _panel_enabled(args, "query"):
        _render_query_panel(lines, state, args)

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

    if _panel_enabled(args, "perception"):
        lines.append(f"{_focus_prefix(state, 'perception')}Perception")
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
        lines.append(f"{_focus_prefix(state, 'action')}Action")
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
        lines.append(f"{_focus_prefix(state, 'command')}Command")
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
        lines.append(f"{_focus_prefix(state, 'system')}System (tegrastats)")
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
        lines.append(f"{_focus_prefix(state, 'memory')}Memory")
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
        lines.append(f"{_focus_prefix(state, 'timeline')}Timeline")
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
        lines.append(f"{_focus_prefix(state, 'transport')}Transport")
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
        lines.append(f"{_focus_prefix(state, 'logs')}Recent Logs")
        if state.logs:
            for item in list(state.logs)[-args.max_log_lines :]:
                lines.append(f"  {_shorten(item, 160)}")
        else:
            lines.append("  no journal matches yet")
        lines.append("")

    if _panel_enabled(args, "warnings") and state.warnings:
        lines.append(f"{_focus_prefix(state, 'warnings')}Warnings")
        for item in state.warnings:
            lines.append(f"  {_shorten(item, 160)}")
        lines.append("")

    if _panel_enabled(args, "events"):
        lines.append(f"{_focus_prefix(state, 'events')}Event Counts")
        if not state.event_counts:
            lines.append("  no events yet")
        else:
            for source in sorted(state.event_counts):
                lines.append(f"  {source}: {state.event_counts[source]}")
        lines.append("")

    if state.reasoning_show_help:
        lines.append("Hotkeys")
        lines.append("  ?: toggle this help")
        lines.append("  h/l: move focus panel")
        lines.append("  j/k: previous/next reasoning event")
        lines.append("  u/i: previous/next trace")
        lines.append("  f: cycle reasoning filter (all/errors/slow)")
        lines.append("  r: toggle reasoning redaction")
        lines.append("  x: export query slice now")
        lines.append("  o: focus trace detail panel")
        lines.append("  p: pin/unpin selected trace")
        lines.append("  1/2/3: apply panel preset")
        lines.append("  Ctrl-C: exit")
        lines.append("")
    lines.append(
        "Cmd: [? help] [h/l focus] [j/k reasoning] [u/i trace] [o detail] [p pin] [f filter] [r redact] [x export] [1/2/3 presets] [Ctrl-C exit]"
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mac-side telemetry dashboard for PALA.")
    parser.add_argument(
        "--ui-mode",
        choices=["reasoning", "classic"],
        default="reasoning",
        help="Viewer layout mode. reasoning enables reasoning-first panels and hotkeys.",
    )
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
        choices=[
            "summary",
            "alerts",
            "throughput",
            "quality",
            "doctor",
            "incident",
            "insights",
            "story",
            "query",
            "scoreboard",
            "trace_list",
            "trace_detail",
            "reasoning_stream",
            "request_detail",
            "reasoning_health",
            "video",
            "perception",
            "action",
            "command",
            "system",
            "memory",
            "timeline",
            "transport",
            "logs",
            "warnings",
            "events",
        ],
        help="Restrict visible dashboard panels (repeatable). Default derives from active packs.",
    )
    parser.add_argument("--focus-panel", default="", help="Initial focused panel for keyboard navigation.")
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
    parser.add_argument("--capture-manifest-version", type=int, default=TELEMETRY_SCHEMA_VERSION_V3)
    parser.add_argument("--save-session", default="", help="Local capture directory on Mac for viewer-side session bundle.")
    parser.add_argument("--replay", default="", help="Replay an existing local capture directory instead of SSH live mode.")
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--replay-no-timing", action="store_true")
    parser.add_argument("--max-log-lines", type=int, default=8)
    parser.add_argument("--reasoning-snippet-max-chars", type=int, default=180)
    parser.add_argument("--reasoning-redact", choices=["on", "off"], default="on")
    parser.add_argument("--reasoning-kpi-window-s", type=float, default=120.0)
    parser.add_argument("--reasoning-slow-ms", type=float, default=2000.0)
    parser.add_argument("--trace-match-window-s", type=float, default=2.0)
    parser.add_argument("--trace-max-events", type=int, default=1000)
    parser.add_argument("--index-mode", choices=["auto", "off", "sqlite"], default="auto")
    parser.add_argument("--query", default="", help="Optional query against indexed telemetry (for replay/session DB).")
    parser.add_argument("--query-limit", type=int, default=10, help="Max rows shown in query panel.")
    parser.add_argument("--query-export", default="", help="Optional path to write current query results JSON.")
    parser.add_argument("--query-export-interval-s", type=float, default=5.0, help="Minimum seconds between query export writes.")
    parser.add_argument("--query-slice-export", default="", help="Optional JSONL path for on-demand query slice exports.")
    parser.add_argument("--insight-max-recommendations", type=int, default=4, help="Max recommendations shown in Insights panel.")
    parser.add_argument("--story-max-rows", type=int, default=8, help="Max story rows shown in Story panel.")
    parser.add_argument("--golden-session", action="append", default=[], help="Golden session path for baseline comparison.")
    parser.add_argument("--scenario-tag", action="append", default=[], help="Scenario tag for insight/scoreboard metadata.")
    parser.add_argument("--goal-tag", action="append", default=[], help="Goal tag for insight/scoreboard metadata.")
    parser.add_argument("--runbook", default="", help="Runbook/context note for this run.")
    parser.add_argument("--scoreboard-path", default=DEFAULT_SCOREBOARD_PATH, help="Scoreboard JSON path.")
    parser.add_argument("--scoreboard-refresh-s", type=float, default=10.0, help="Seconds between scoreboard reloads.")
    parser.add_argument("--no-scoreboard-update", action="store_true", help="Disable scoreboard updates for local captures.")
    parser.add_argument("--rate-window-s", type=float, default=10.0, help="Window in seconds for recent throughput metrics.")
    parser.add_argument("--index-refresh-s", type=float, default=15.0, help="Rebuild local session.db periodically during live capture.")
    parser.add_argument("--alert-policy", choices=["demo", "training", "debug", "custom"], default="custom", help="Preset alert thresholds.")
    parser.add_argument("--alert-stale-s", type=float, default=10.0, help="Alert when stream RX age exceeds this in seconds.")
    parser.add_argument("--alert-heartbeat-s", type=float, default=5.0, help="Alert when heartbeat age exceeds this in seconds.")
    parser.add_argument("--alert-video-idle-s", type=float, default=6.0, help="Alert when video age exceeds this in seconds.")
    parser.add_argument("--alert-dropped-events", type=int, default=1, help="Alert when dropped agent events >= this.")
    parser.add_argument("--alert-warning-count", type=int, default=4, help="Alert when warnings deque reaches this size.")
    parser.add_argument("--alert-trace-grace-s", type=float, default=20.0, help="Alert if no traces observed after this uptime.")
    parser.add_argument("--alert-min-events-per-s", type=float, default=0.0, help="Alert when recent events/sec drops below this (>0 enables).")
    parser.add_argument(
        "--quality-gate",
        choices=["off", "warn", "strict"],
        default="off",
        help="Apply quality gate status to loaded replay/capture quality report.",
    )
    parser.add_argument(
        "--doctor-gate",
        choices=["off", "warn", "strict"],
        default="off",
        help="Apply doctor readiness gate to loaded replay/capture doctor report.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    defaults = parser.parse_args([])
    args = parser.parse_args()
    _apply_alert_policy(args, defaults=defaults)
    if args.list_packs:
        for pack in list_packs():
            print(f"{pack.name}: {pack.description}")
        return 0
    if not args.pack:
        args.pack = ["reasoning_live"] if args.ui_mode == "reasoning" else ["runtime_core"]
    try:
        resolved_packs = resolve_packs(args.pack)
        resolved_packs = apply_pack_overrides(resolved_packs, args.pack_override)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.panel:
        if args.ui_mode == "reasoning":
            args.panel = [
                "summary",
                "alerts",
                "quality",
                "doctor",
                "incident",
                "insights",
                "story",
                "query",
                "scoreboard",
                "trace_list",
                "trace_detail",
                "reasoning_stream",
                "request_detail",
                "reasoning_health",
                "video",
                "transport",
            ]
        else:
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
    state.configure_panels(args.panel, focus_panel=str(args.focus_panel or ""))
    state.query_text = str(args.query or "").strip()
    if args.ui_mode == "reasoning":
        if "trace_list" in args.panel:
            state.focus_panel = "trace_list"
        elif "reasoning_stream" in args.panel:
            state.focus_panel = "reasoning_stream"
    if map_note:
        state.logs.append(map_note)
    if str(args.scoreboard_path or "").strip():
        try:
            state.scoreboard = load_scoreboard(str(args.scoreboard_path))
        except Exception as exc:
            state.warnings.append(f"scoreboard_load_failed: {exc!r}")
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
    replay_thread: Optional[threading.Thread] = None
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
    last_query_export_s = 0.0
    last_query_export_error_s = 0.0
    last_query_slice_export_s = 0.0
    last_query_slice_note_s = 0.0
    last_index_refresh_s = 0.0
    last_index_refresh_note_s = 0.0
    last_index_error_s = 0.0
    last_insight_reload_s = 0.0
    last_scoreboard_reload_s = 0.0
    last_scoreboard_error_s = 0.0
    last_alert_set: set[str] = set()
    session_db_path = ""
    capture_writer: Optional[SessionCaptureWriter] = None
    save_session_dir = str(args.save_session or "").strip()
    if save_session_dir:
        session_db_path = resolve_session_db_path(save_session_dir)
        cfg = CaptureConfig(
            directory=save_session_dir,
            frames_mode=args.capture_frames,
            max_seconds=max(0.0, float(args.capture_max_seconds)),
            manifest_version=int(args.capture_manifest_version),
            trace_match_window_s=max(0.1, float(args.trace_match_window_s)),
            scenario_tags=list(args.scenario_tag or []),
            goal_tags=list(args.goal_tag or []),
            runbook=str(args.runbook or ""),
            golden_sessions=list(args.golden_session or []),
            scoreboard_path=str(args.scoreboard_path or DEFAULT_SCOREBOARD_PATH),
            scoreboard_update=not bool(args.no_scoreboard_update),
            metadata={
                "mode": "replay" if replay_mode else "live",
                "packs": list(args.pack),
                "field_filters": list(args.field_filter),
                "scenario_tags": list(args.scenario_tag or []),
                "goal_tags": list(args.goal_tag or []),
                "runbook": str(args.runbook or ""),
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

    def _apply_doctor_gate() -> None:
        passed, note = _evaluate_doctor_gate(state.doctor_report, str(args.doctor_gate))
        state.doctor_gate_passed = passed
        state.doctor_gate_note = note

    _apply_quality_gate()
    _apply_doctor_gate()
    video_window = _init_video_window(args, state)

    if replay_mode:
        replay_dir = str(args.replay).strip()
        if not os.path.isdir(replay_dir):
            parser.error(f"replay directory not found: {replay_dir}")
        state.connected = True
        state.connection_note = f"replay from {replay_dir}"
        replay_reader = SessionReplayReader(replay_dir)
        session_db_path = resolve_session_db_path(replay_dir)
        if args.index_mode in {"auto", "sqlite"} and not os.path.exists(session_db_path):
            try:
                build_summary = build_session_db(replay_dir, replace=False)
                state.logs.append(
                    f"built replay index: events={build_summary.get('event_count')} traces={build_summary.get('trace_count')}"
                )
            except Exception as exc:
                state.warnings.append(f"replay_index_build_failed: {exc!r}")
                if args.index_mode == "sqlite":
                    state.warnings.append("sqlite mode requested but session.db is unavailable")
        state.quality_report = load_quality_report(replay_dir)
        state.doctor_report = load_doctor_report(replay_dir)
        state.improvement_report = load_improvement_report(replay_dir)
        state.incident_report = load_incident_report(replay_dir)
        wants_runtime_insights = bool(args.golden_session or args.scenario_tag or args.goal_tag or str(args.runbook or "").strip())
        if state.improvement_report is None or wants_runtime_insights:
            try:
                generated = build_improvement_report(
                    replay_dir,
                    golden_sessions=list(args.golden_session or []),
                    scenario_tags=list(args.scenario_tag or []),
                    goal_tags=list(args.goal_tag or []),
                    runbook=str(args.runbook or ""),
                )
                if state.improvement_report is None:
                    write_improvement_report(replay_dir, generated)
                state.improvement_report = generated
                state.logs.append("generated runtime improvement report for replay session")
            except Exception as exc:
                state.warnings.append(f"improvement_report_build_failed: {exc!r}")
        if state.doctor_report is None:
            try:
                generated_doc = build_doctor_report(
                    replay_dir,
                    quality_report=state.quality_report,
                    improvement_report=state.improvement_report,
                )
                write_doctor_report(replay_dir, generated_doc)
                state.doctor_report = generated_doc
                state.logs.append("generated doctor report for replay session")
            except Exception as exc:
                state.warnings.append(f"doctor_report_build_failed: {exc!r}")
        if state.incident_report is None:
            try:
                generated_incident = build_incident_report(
                    replay_dir,
                    quality_report=state.quality_report,
                    doctor_report=state.doctor_report,
                    improvement_report=state.improvement_report,
                )
                write_incident_report(replay_dir, generated_incident)
                write_incident_markdown(replay_dir, generated_incident)
                state.incident_report = generated_incident
                state.logs.append("generated incident report for replay session")
            except Exception as exc:
                state.warnings.append(f"incident_report_build_failed: {exc!r}")
        _apply_quality_gate()
        _apply_doctor_gate()
        if not _can_preload_trace_index(field_filters=field_filters):
            state.logs.append("trace index preload disabled because field filters are active")
        else:
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
                state.query_rows = _build_in_memory_query_rows(state, limit=int(args.query_limit), now_wall_s=time.time())
            else:
                try:
                    query_out = query_session_db(
                        session_db_path if os.path.exists(session_db_path) else replay_dir,
                        query=state.query_text,
                        limit=int(args.query_limit),
                    )
                    state.query_rows = _rows_from_query_out(query_out, limit=int(args.query_limit))
                    state.query_note = _query_note_from_query_out(query_out)
                except Exception as exc:
                    state.query_note = f"sqlite query failed: {exc!r}"
                    state.query_rows = _build_in_memory_query_rows(state, limit=int(args.query_limit), now_wall_s=time.time())
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
        for key in key_reader.poll():
            if key == "?":
                state.reasoning_show_help = not state.reasoning_show_help
                continue
            if key in PANEL_PRESETS:
                _apply_panel_preset(state, args, key)
                continue
            if key == "h":
                state.cycle_focus_panel(delta=-1)
                continue
            if key == "l":
                state.cycle_focus_panel(delta=1)
                continue
            if key == "j":
                state.move_reasoning_selection(delta=-1, slow_ms=float(args.reasoning_slow_ms))
                continue
            if key == "k":
                state.move_reasoning_selection(delta=1, slow_ms=float(args.reasoning_slow_ms))
                continue
            if key == "u":
                state.move_trace_selection(-1)
                continue
            if key == "i":
                state.move_trace_selection(1)
                continue
            if key == "o":
                if "trace_detail" in state.active_panels:
                    state.focus_panel = "trace_detail"
                continue
            if key == "p":
                state.toggle_trace_pin()
                continue
            if key == "f":
                _cycle_reasoning_filter(state)
                continue
            if key == "r":
                args.reasoning_redact = "off" if args.reasoning_redact == "on" else "on"
                state.logs.append(f"reasoning redaction={args.reasoning_redact}")
                continue
            if key == "x":
                if not state.query_text:
                    state.warnings.append("query slice export skipped: --query is empty")
                    continue
                target = str(args.query_slice_export or "").strip() or "logs/telemetry/query_slice.jsonl"
                try:
                    written = _write_query_slice_export(target, query=state.query_text, rows=list(state.query_rows))
                    state.query_slice_exports += 1
                    state.logs.append(f"query slice exported: rows={written} path={target}")
                except Exception as exc:
                    state.warnings.append(f"query_slice_export_failed: {exc!r}")
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
                        state.doctor_report = load_doctor_report(save_session_dir)
                        state.improvement_report = load_improvement_report(save_session_dir)
                        state.incident_report = load_incident_report(save_session_dir)
                        _apply_quality_gate()
                        _apply_doctor_gate()

        if state.query_text and (now - last_query_refresh_s) >= 1.0:
            last_query_refresh_s = now
            if args.index_mode == "off":
                state.query_rows = _build_in_memory_query_rows(state, limit=int(args.query_limit), now_wall_s=now)
                state.query_note = "memory query (index disabled)"
            else:
                lookup_root = ""
                if session_db_path and os.path.exists(session_db_path):
                    lookup_root = session_db_path
                elif replay_mode and os.path.isdir(str(args.replay).strip()):
                    lookup_root = str(args.replay).strip()
                elif save_session_dir and os.path.isdir(save_session_dir):
                    lookup_root = save_session_dir

                if lookup_root:
                    try:
                        query_out = query_session_db(
                            lookup_root,
                            query=state.query_text,
                            limit=int(args.query_limit),
                        )
                        state.query_rows = _rows_from_query_out(query_out, limit=int(args.query_limit))
                        state.query_note = _query_note_from_query_out(query_out)
                    except Exception as exc:
                        state.query_rows = _build_in_memory_query_rows(state, limit=int(args.query_limit), now_wall_s=now)
                        state.query_note = f"sqlite query fallback: {exc!r}"
                else:
                    state.query_rows = _build_in_memory_query_rows(state, limit=int(args.query_limit), now_wall_s=now)
                    state.query_note = "memory query (session db unavailable)"

        if (
            save_session_dir
            and (args.index_mode in {"auto", "sqlite"})
            and (now - last_index_refresh_s) >= max(2.0, float(args.index_refresh_s))
            and os.path.exists(os.path.join(save_session_dir, "events.jsonl"))
        ):
            last_index_refresh_s = now
            try:
                index_summary = build_session_db(save_session_dir, replace=True)
                session_db_path = resolve_session_db_path(save_session_dir)
                if (now - last_index_refresh_note_s) >= 30.0:
                    state.logs.append(
                        f"session index refreshed: events={index_summary.get('event_count')} traces={index_summary.get('trace_count')}"
                    )
                    last_index_refresh_note_s = now
            except Exception as exc:
                if (now - last_index_error_s) >= 10.0:
                    state.warnings.append(f"session_index_refresh_failed: {exc!r}")
                    last_index_error_s = now

        if save_session_dir and (now - last_insight_reload_s) >= 5.0:
            loaded = load_improvement_report(save_session_dir)
            if loaded is not None:
                state.improvement_report = loaded
            loaded_doctor = load_doctor_report(save_session_dir)
            if loaded_doctor is not None:
                state.doctor_report = loaded_doctor
                _apply_doctor_gate()
            loaded_incident = load_incident_report(save_session_dir)
            if loaded_incident is not None:
                state.incident_report = loaded_incident
            last_insight_reload_s = now

        if str(args.scoreboard_path or "").strip() and (now - last_scoreboard_reload_s) >= max(2.0, float(args.scoreboard_refresh_s)):
            try:
                state.scoreboard = load_scoreboard(str(args.scoreboard_path))
            except Exception as exc:
                if (now - last_scoreboard_error_s) >= 10.0:
                    state.warnings.append(f"scoreboard_reload_failed: {exc!r}")
                    last_scoreboard_error_s = now
            last_scoreboard_reload_s = now

        state.active_alerts = _collect_alerts(state, args, now)
        active_set = set(state.active_alerts)
        for item in sorted(active_set - last_alert_set):
            text = f"alert_on: {item}"
            state.logs.append(text)
            state.alert_history.append(text)
        for item in sorted(last_alert_set - active_set):
            text = f"alert_off: {item}"
            state.logs.append(text)
            state.alert_history.append(text)
        last_alert_set = active_set

        if (
            state.query_text
            and str(args.query_export or "").strip()
            and (now - last_query_export_s) >= max(0.5, float(args.query_export_interval_s))
        ):
            try:
                _write_query_export(
                    str(args.query_export),
                    query=state.query_text,
                    note=state.query_note,
                    rows=list(state.query_rows),
                )
                last_query_export_s = now
            except Exception as exc:
                if (now - last_query_export_error_s) >= 10.0:
                    state.warnings.append(f"query_export_failed: {exc!r}")
                    last_query_export_error_s = now

        if (
            state.query_text
            and str(args.query_slice_export or "").strip()
            and (now - last_query_slice_export_s) >= max(0.5, float(args.query_export_interval_s))
        ):
            try:
                written = _write_query_slice_export(
                    str(args.query_slice_export),
                    query=state.query_text,
                    rows=list(state.query_rows),
                )
                last_query_slice_export_s = now
                state.query_slice_exports += 1
                if (now - last_query_slice_note_s) >= 30.0:
                    state.logs.append(f"query slice exported: rows={written} path={args.query_slice_export}")
                    last_query_slice_note_s = now
            except Exception as exc:
                if (now - last_query_export_error_s) >= 10.0:
                    state.warnings.append(f"query_slice_export_failed: {exc!r}")
                    last_query_export_error_s = now

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
        loaded_doctor = load_doctor_report(save_session_dir)
        if loaded_doctor is not None:
            state.doctor_report = loaded_doctor
            _apply_doctor_gate()
        loaded_improvement = load_improvement_report(save_session_dir)
        if loaded_improvement is not None:
            state.improvement_report = loaded_improvement
        loaded_incident = load_incident_report(save_session_dir)
        if loaded_incident is not None:
            state.incident_report = loaded_incident
    key_reader.stop()

    if video_window is not None:
        try:
            video_window.root.destroy()
        except Exception:
            pass

    print("\n")
    if str(args.quality_gate) == "strict" and state.quality_gate_passed is not True:
        return 2
    if str(args.doctor_gate) == "strict" and state.doctor_gate_passed is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
