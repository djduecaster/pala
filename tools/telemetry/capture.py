from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import os
import socket
import time
from typing import Any, Dict, List, Optional, Sequence

from .labels import derive_weak_labels, write_labels_jsonl
from .doctor import build_doctor_report, write_doctor_report
from .incident import build_incident_report, write_incident_markdown, write_incident_report
from .insights import build_improvement_report, write_improvement_report
from .quality import build_quality_report, write_quality_report
from .reasoning import format_reasoning_snippet, normalize_reasoning_message
from .scoreboard import DEFAULT_SCOREBOARD_PATH, add_scoreboard_session
from .schema_v3 import (
    DOCTOR_REPORT_PATH,
    IMPROVEMENT_REPORT_PATH,
    INCIDENT_REPORT_PATH,
    QUALITY_REPORT_PATH,
    TELEMETRY_SCHEMA_VERSION_V3,
    WEAK_LABELS_PATH,
    upgrade_manifest_v3,
)
from .storage_sqlite import build_session_db
from .trace_graph import TraceGraphBuilder


MANIFEST_SCHEMA_VERSION = TELEMETRY_SCHEMA_VERSION_V3


@dataclass
class CaptureConfig:
    directory: str
    frames_mode: str = "off"  # off | keyframes | all
    max_seconds: float = 0.0
    manifest_version: int = MANIFEST_SCHEMA_VERSION
    trace_match_window_s: float = 2.0
    trace_max_events: int = 20_000
    scenario_tags: Sequence[str] = field(default_factory=list)
    goal_tags: Sequence[str] = field(default_factory=list)
    runbook: str = ""
    golden_sessions: Sequence[str] = field(default_factory=list)
    scoreboard_path: str = DEFAULT_SCOREBOARD_PATH
    scoreboard_update: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionCaptureWriter:
    def __init__(self, cfg: CaptureConfig) -> None:
        directory = str(cfg.directory).strip()
        if not directory:
            raise ValueError("capture directory is required")
        self._cfg = cfg
        self._dir = directory
        self._events_path = os.path.join(self._dir, "events.jsonl")
        self._manifest_path = os.path.join(self._dir, "manifest.json")
        self._index_path = os.path.join(self._dir, "index.json")
        self._reasoning_index_path = os.path.join(self._dir, "reasoning_index.json")
        self._trace_index_path = os.path.join(self._dir, "trace_index.json")
        self._frames_dir = os.path.join(self._dir, "frames")
        self._events_fh = None
        self._start_wall_s = time.time()
        self._event_count = 0
        self._frame_count = 0
        self._bytes_written = 0
        self._index: List[Dict[str, Any]] = []
        self._reasoning_index: List[Dict[str, Any]] = []
        self._source_counts: Dict[str, int] = {}
        self._trace_builder = TraceGraphBuilder(
            match_window_s=float(cfg.trace_match_window_s),
            max_events=int(cfg.trace_max_events),
        )
        self._closed = False

        os.makedirs(self._dir, exist_ok=True)
        os.makedirs(self._frames_dir, exist_ok=True)
        self._events_fh = open(self._events_path, "w", encoding="utf-8")

    def _should_store_frame(self, frame_id: Optional[int]) -> bool:
        mode = str(self._cfg.frames_mode or "off").strip().lower()
        if mode == "off":
            return False
        if mode == "all":
            return True
        if mode == "keyframes":
            if frame_id is None:
                return self._frame_count == 0
            return frame_id <= 0 or (frame_id % 10) == 0
        return False

    def _store_frame(self, event_idx: int, payload: Dict[str, Any]) -> None:
        frame_id_obj = payload.get("frame_id")
        frame_id = frame_id_obj if isinstance(frame_id_obj, int) else None
        if not self._should_store_frame(frame_id):
            payload.pop("bytes_b64", None)
            return

        b64 = payload.get("bytes_b64")
        if not isinstance(b64, str):
            return
        try:
            data = base64.b64decode(b64, validate=True)
        except Exception:
            return
        file_name = f"frame_{event_idx:08d}_{(frame_id or 0):08d}.jpg"
        abs_path = os.path.join(self._frames_dir, file_name)
        with open(abs_path, "wb") as fh:
            fh.write(data)
        self._bytes_written += len(data)
        payload.pop("bytes_b64", None)
        payload["frame_ref"] = f"frames/{file_name}"
        payload["frame_bytes"] = len(data)
        self._index.append(
            {
                "event_index": event_idx,
                "frame_id": frame_id,
                "frame_ref": payload["frame_ref"],
                "pts_ns": payload.get("pts_ns"),
            }
        )
        self._frame_count += 1

    def write(self, msg: Dict[str, Any]) -> bool:
        if self._closed:
            return False

        max_s = max(0.0, float(self._cfg.max_seconds))
        if max_s > 0.0 and (time.time() - self._start_wall_s) > max_s:
            return False

        line_obj = json.loads(json.dumps(msg, ensure_ascii=True))
        source = str(line_obj.get("source", "unknown"))
        self._source_counts[source] = self._source_counts.get(source, 0) + 1
        payload = line_obj.get("payload")
        if isinstance(payload, dict) and source == "video_frame":
            self._store_frame(self._event_count, payload)
        reasoning_event = normalize_reasoning_message(line_obj)
        if reasoning_event is not None:
            self._reasoning_index.append(
                {
                    "event_index": self._event_count,
                    "source": reasoning_event.source,
                    "ts_wall_s": reasoning_event.ts_wall_s,
                    "req_id": reasoning_event.req_id,
                    "phase": reasoning_event.phase,
                    "status": reasoning_event.status,
                    "latency_ms": reasoning_event.latency_ms,
                    "confidence": reasoning_event.confidence,
                    "severity": reasoning_event.severity,
                    "snippet": format_reasoning_snippet(
                        reasoning_event.snippet,
                        max_chars=220,
                        redact=False,
                    ),
                }
            )
        self._trace_builder.ingest(line_obj)

        assert self._events_fh is not None
        encoded = json.dumps(line_obj, separators=(",", ":"), ensure_ascii=True)
        self._events_fh.write(encoded)
        self._events_fh.write("\n")
        self._event_count += 1
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._events_fh is not None:
            self._events_fh.flush()
            self._events_fh.close()
            self._events_fh = None

        with open(self._index_path, "w", encoding="utf-8") as fh:
            json.dump({"frames": self._index}, fh, separators=(",", ":"), ensure_ascii=True)
        with open(self._reasoning_index_path, "w", encoding="utf-8") as fh:
            json.dump({"events": self._reasoning_index}, fh, separators=(",", ":"), ensure_ascii=True)
        trace_records = self._trace_builder.traces()
        trace_index = self._trace_builder.build_trace_index()
        with open(self._trace_index_path, "w", encoding="utf-8") as fh:
            json.dump(trace_index, fh, separators=(",", ":"), ensure_ascii=True)

        manifest = {
            "schema_version": int(self._cfg.manifest_version),
            "created_at_wall_s": self._start_wall_s,
            "closed_at_wall_s": time.time(),
            "host": socket.gethostname(),
            "events_path": "events.jsonl",
            "index_path": "index.json",
            "reasoning_index_path": "reasoning_index.json",
            "trace_index_path": "trace_index.json",
            "frames_dir": "frames",
            "event_count": self._event_count,
            "frame_count": self._frame_count,
            "reasoning_event_count": len(self._reasoning_index),
            "trace_count": len(trace_index.get("traces", [])),
            "stored_frame_bytes": self._bytes_written,
            "frames_mode": str(self._cfg.frames_mode),
            "trace_match_window_s": float(self._cfg.trace_match_window_s),
            "source_counts": dict(self._source_counts),
            "metadata": self._cfg.metadata,
        }

        index_summary: Optional[Dict[str, Any]] = None
        try:
            index_summary = build_session_db(self._dir, replace=True)
        except Exception as exc:
            manifest.setdefault("v3_artifact_errors", []).append(f"session_db: {exc!r}")

        weak_label_count: Optional[int] = None
        try:
            labels = derive_weak_labels(reasoning_index=self._reasoning_index, traces=trace_records)
            labels_path = os.path.join(self._dir, WEAK_LABELS_PATH)
            weak_label_count = write_labels_jsonl(labels_path, labels)
        except Exception as exc:
            manifest.setdefault("v3_artifact_errors", []).append(f"weak_labels: {exc!r}")

        quality_report = None
        try:
            quality_report = build_quality_report(
                event_count=self._event_count,
                source_counts=self._source_counts,
                reasoning_index=self._reasoning_index,
                traces=trace_records,
            )
            write_quality_report(self._dir, quality_report, filename=QUALITY_REPORT_PATH)
        except Exception as exc:
            manifest.setdefault("v3_artifact_errors", []).append(f"quality_report: {exc!r}")

        improvement_report = None
        try:
            improvement_report = build_improvement_report(
                self._dir,
                golden_sessions=self._cfg.golden_sessions,
                scenario_tags=self._cfg.scenario_tags,
                goal_tags=self._cfg.goal_tags,
                runbook=self._cfg.runbook,
            )
            write_improvement_report(self._dir, improvement_report, filename=IMPROVEMENT_REPORT_PATH)
        except Exception as exc:
            manifest.setdefault("v3_artifact_errors", []).append(f"improvement_report: {exc!r}")

        if improvement_report is not None and bool(self._cfg.scoreboard_update):
            try:
                add_scoreboard_session(
                    path=str(self._cfg.scoreboard_path or DEFAULT_SCOREBOARD_PATH),
                    session_dir=self._dir,
                    manifest=manifest,
                    quality_report=quality_report,
                    improvement_report=improvement_report,
                    scenario_tags=self._cfg.scenario_tags,
                    goal_tags=self._cfg.goal_tags,
                    runbook=self._cfg.runbook,
                )
            except Exception as exc:
                manifest.setdefault("v3_artifact_errors", []).append(f"scoreboard: {exc!r}")

        doctor_report = None
        try:
            doctor_report = build_doctor_report(
                self._dir,
                manifest=manifest,
                quality_report=quality_report,
                improvement_report=improvement_report,
                index_summary=index_summary,
            )
            write_doctor_report(self._dir, doctor_report, filename=DOCTOR_REPORT_PATH)
        except Exception as exc:
            manifest.setdefault("v3_artifact_errors", []).append(f"doctor_report: {exc!r}")

        incident_report = None
        try:
            incident_report = build_incident_report(
                self._dir,
                quality_report=quality_report,
                doctor_report=doctor_report,
                improvement_report=improvement_report,
            )
            write_incident_report(self._dir, incident_report, filename=INCIDENT_REPORT_PATH)
            write_incident_markdown(self._dir, incident_report)
        except Exception as exc:
            manifest.setdefault("v3_artifact_errors", []).append(f"incident_report: {exc!r}")

        manifest = upgrade_manifest_v3(
            manifest,
            index_summary=index_summary,
            quality_report=quality_report,
            improvement_report=improvement_report,
            doctor_report=doctor_report,
            incident_report=incident_report,
            weak_label_count=weak_label_count,
        )
        with open(self._manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, separators=(",", ":"), ensure_ascii=True)
