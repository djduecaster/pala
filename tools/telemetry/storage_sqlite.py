from __future__ import annotations

import json
import os
import sqlite3
import shlex
import time
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .reasoning import format_reasoning_snippet, normalize_reasoning_message
from .schema_v3 import SESSION_DB_PATH
from .trace_join import build_reasoning_trace_rows
from .trace_graph import TraceGraphBuilder, TraceRecord, load_trace_index, resolve_trace_index_path


_SESSION_CONTRACT_REQUIRED_TABLES = {
    "events",
    "reasoning",
    "traces",
    "trace_events",
    "reasoning_traces",
    "cases",
    "case_events",
    "case_labels",
    "case_reviews",
    "meta",
}


def resolve_session_db_path(path: str) -> str:
    raw = str(path)
    if raw.endswith(".db"):
        return raw
    return os.path.join(raw, SESSION_DB_PATH)


def _iter_events(events_path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(events_path, "r", encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh):
            text = line.strip()
            if not text:
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            msg.setdefault("seq", idx)
            yield idx, msg


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _trace_ref_lookup(traces: Sequence[TraceRecord]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for trace in traces:
        for ref in trace.event_refs:
            out[int(ref.event_index)] = {
                "req_id": ref.req_id,
                "phase": ref.phase or "",
                "status": ref.status or "",
                "severity": ref.severity or "info",
                "latency_ms": ref.latency_ms,
                "snippet": ref.summary or "",
                "trace_id": trace.trace_id,
            }
    return out


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS events (
            seq INTEGER PRIMARY KEY,
            ts_wall_s REAL,
            source TEXT NOT NULL,
            level TEXT,
            req_id INTEGER,
            phase TEXT,
            status TEXT,
            severity TEXT,
            latency_ms REAL,
            snippet TEXT,
            payload_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
        CREATE INDEX IF NOT EXISTS idx_events_req_id ON events(req_id);
        CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_wall_s);

        CREATE TABLE IF NOT EXISTS reasoning (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_index INTEGER NOT NULL,
            source TEXT NOT NULL,
            ts_wall_s REAL,
            req_id INTEGER,
            phase TEXT,
            status TEXT,
            severity TEXT,
            latency_ms REAL,
            snippet TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_reasoning_req_id ON reasoning(req_id);
        CREATE INDEX IF NOT EXISTS idx_reasoning_status ON reasoning(status);

        CREATE TABLE IF NOT EXISTS traces (
            trace_id TEXT PRIMARY KEY,
            req_id INTEGER,
            start_ts_wall_s REAL,
            end_ts_wall_s REAL,
            duration_ms REAL,
            status TEXT,
            severity TEXT,
            summary TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_traces_req_id ON traces(req_id);
        CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);

        CREATE TABLE IF NOT EXISTS trace_events (
            trace_id TEXT NOT NULL,
            event_index INTEGER NOT NULL,
            source TEXT NOT NULL,
            phase TEXT,
            status TEXT,
            latency_ms REAL,
            severity TEXT,
            summary TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_trace_events_trace ON trace_events(trace_id);
        CREATE INDEX IF NOT EXISTS idx_trace_events_event ON trace_events(event_index);

        CREATE TABLE IF NOT EXISTS reasoning_traces (
            row_id INTEGER PRIMARY KEY,
            event_index INTEGER NOT NULL,
            trace_id TEXT NOT NULL,
            req_id INTEGER,
            component TEXT,
            source TEXT NOT NULL,
            ts_wall_s REAL,
            phase TEXT,
            status TEXT,
            severity TEXT,
            latency_ms REAL,
            primitive TEXT,
            confidence REAL,
            target_zone TEXT,
            model TEXT,
            provider TEXT,
            delta_score REAL,
            snippet TEXT,
            input_preview TEXT,
            output_preview TEXT,
            perception_ts_wall_s REAL,
            perception_person_conf REAL,
            perception_zone_hint TEXT,
            perception_frame_id INTEGER,
            video_frame_event_index INTEGER,
            video_frame_id INTEGER,
            video_frame_ref TEXT,
            video_frame_width INTEGER,
            video_frame_height INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_reasoning_traces_trace ON reasoning_traces(trace_id);
        CREATE INDEX IF NOT EXISTS idx_reasoning_traces_req ON reasoning_traces(req_id);
        CREATE INDEX IF NOT EXISTS idx_reasoning_traces_comp ON reasoning_traces(component);
        CREATE INDEX IF NOT EXISTS idx_reasoning_traces_ts ON reasoning_traces(ts_wall_s);

        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            req_id INTEGER,
            first_event_index INTEGER,
            start_ts_wall_s REAL,
            end_ts_wall_s REAL,
            duration_ms REAL,
            status TEXT,
            severity TEXT,
            component TEXT,
            event_count INTEGER,
            error_count INTEGER,
            warning_count INTEGER,
            max_latency_ms REAL,
            hardness REAL,
            summary TEXT,
            snippet TEXT,
            source TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cases_trace ON cases(trace_id);
        CREATE INDEX IF NOT EXISTS idx_cases_req ON cases(req_id);
        CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
        CREATE INDEX IF NOT EXISTS idx_cases_severity ON cases(severity);
        CREATE INDEX IF NOT EXISTS idx_cases_hardness ON cases(hardness);
        CREATE INDEX IF NOT EXISTS idx_cases_start_ts ON cases(start_ts_wall_s);

        CREATE TABLE IF NOT EXISTS case_events (
            case_id TEXT NOT NULL,
            row_id INTEGER NOT NULL,
            event_index INTEGER NOT NULL,
            source TEXT NOT NULL,
            component TEXT,
            phase TEXT,
            status TEXT,
            severity TEXT,
            ts_wall_s REAL,
            latency_ms REAL,
            snippet TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_case_events_case ON case_events(case_id);
        CREATE INDEX IF NOT EXISTS idx_case_events_event ON case_events(event_index);

        CREATE TABLE IF NOT EXISTS case_labels (
            case_id TEXT NOT NULL,
            label TEXT NOT NULL,
            score REAL,
            reason TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_case_labels_case ON case_labels(case_id);
        CREATE INDEX IF NOT EXISTS idx_case_labels_label ON case_labels(label);

        CREATE TABLE IF NOT EXISTS case_reviews (
            case_id TEXT PRIMARY KEY,
            decision TEXT NOT NULL,
            note TEXT,
            reviewer TEXT,
            reviewed_at_wall_s REAL
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )


def _read_manifest(session_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(session_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            decoded = json.load(fh)
    except Exception:
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded


def _read_trace_records(
    session_dir: str,
    events: Sequence[Tuple[int, Dict[str, Any]]],
    *,
    manifest: Optional[Dict[str, Any]],
) -> List[TraceRecord]:
    trace_index_path = resolve_trace_index_path(session_dir, manifest)
    if os.path.exists(trace_index_path):
        try:
            traces = load_trace_index(trace_index_path)
            if traces:
                return traces
        except Exception:
            pass
    builder = TraceGraphBuilder(match_window_s=2.0, max_events=max(128, len(events) + 8))
    for _, msg in events:
        builder.ingest(msg)
    return builder.traces()


def _normalize_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


def _severity_rank(value: str) -> int:
    sev = str(value or "").strip().lower()
    if sev == "error":
        return 3
    if sev == "warning":
        return 2
    if sev == "info":
        return 1
    return 0


def _status_rank(value: str) -> int:
    status = _normalize_status(value)
    if "parse_fail" in status:
        return 6
    if "timeout" in status:
        return 5
    if "error" in status or "fail" in status:
        return 4
    if "warning" in status:
        return 3
    if status in {"ok", "success"}:
        return 2
    return 1


def _select_status(values: Sequence[str]) -> str:
    if not values:
        return "unknown"
    return sorted((_normalize_status(v) for v in values), key=_status_rank, reverse=True)[0]


def _select_severity(values: Sequence[str]) -> str:
    if not values:
        return "info"
    return sorted((str(v or "info").strip().lower() for v in values), key=_severity_rank, reverse=True)[0]


def _compute_hardness(
    *,
    status: str,
    severity: str,
    duration_ms: Optional[float],
    error_count: int,
    warning_count: int,
    max_latency_ms: Optional[float],
) -> float:
    score = 0.0
    status_norm = _normalize_status(status)
    sev_norm = str(severity or "info").strip().lower()
    if sev_norm == "error":
        score += 0.40
    elif sev_norm == "warning":
        score += 0.25
    if "parse_fail" in status_norm:
        score += 0.30
    elif "timeout" in status_norm:
        score += 0.24
    elif "error" in status_norm or "fail" in status_norm:
        score += 0.18
    if duration_ms is not None:
        if duration_ms >= 4000.0:
            score += 0.16
        elif duration_ms >= 2000.0:
            score += 0.10
    if max_latency_ms is not None:
        if max_latency_ms >= 2500.0:
            score += 0.15
        elif max_latency_ms >= 1200.0:
            score += 0.08
    score += min(0.12, max(0, int(error_count)) * 0.03)
    score += min(0.06, max(0, int(warning_count)) * 0.015)
    return round(max(0.0, min(1.0, score)), 3)


def _case_labels(
    *,
    status: str,
    severity: str,
    hardness: float,
) -> List[Tuple[str, float, str]]:
    labels: List[Tuple[str, float, str]] = []
    norm_status = _normalize_status(status)
    norm_severity = str(severity or "").strip().lower()
    if "parse_fail" in norm_status:
        labels.append(("planner_parse_fail", 0.99, f"status={norm_status}"))
    if "timeout" in norm_status:
        labels.append(("planner_timeout", 0.96, f"status={norm_status}"))
    if norm_severity == "error":
        labels.append(("reasoning_error", 0.84, f"severity={norm_severity}"))
    if hardness >= 0.70:
        labels.append(("hard_case", max(0.7, min(1.0, hardness)), f"hardness={hardness:.3f}"))
    return labels


def _build_cases(
    *,
    reasoning_trace_rows: Sequence[Mapping[str, Any]],
    traces: Sequence[TraceRecord],
) -> Tuple[List[Tuple[Any, ...]], List[Tuple[Any, ...]], List[Tuple[Any, ...]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in reasoning_trace_rows:
        trace_id = _as_text(row.get("trace_id")) or f"event:{_as_int(row.get('event_index')) or 0}"
        grouped.setdefault(trace_id, []).append(row)

    case_rows: List[Tuple[Any, ...]] = []
    case_event_rows: List[Tuple[Any, ...]] = []
    case_label_rows: List[Tuple[Any, ...]] = []
    trace_lookup: Dict[str, TraceRecord] = {trace.trace_id: trace for trace in traces}

    for trace_id in sorted(grouped):
        rows = sorted(grouped[trace_id], key=lambda row: _as_int(row.get("row_id")) or 0)
        if not rows:
            continue
        case_id = f"case:{trace_id}"
        req_id = _as_int(next((row.get("req_id") for row in rows if _as_int(row.get("req_id")) is not None), None))
        first_event_index = min((_as_int(row.get("event_index")) or 0) for row in rows)
        ts_values = [_as_float(row.get("ts_wall_s")) for row in rows if _as_float(row.get("ts_wall_s")) is not None]
        start_ts = min(ts_values) if ts_values else None
        end_ts = max(ts_values) if ts_values else None
        trace = trace_lookup.get(trace_id)
        if trace is not None:
            if start_ts is None:
                start_ts = _as_float(trace.start_ts_wall_s)
            if end_ts is None:
                end_ts = _as_float(trace.end_ts_wall_s)
        duration_ms = None
        if trace is not None and _as_float(trace.duration_ms) is not None:
            duration_ms = _as_float(trace.duration_ms)
        elif start_ts is not None and end_ts is not None:
            duration_ms = max(0.0, (end_ts - start_ts) * 1000.0)

        statuses = [_as_text(row.get("status")) for row in rows]
        severities = [_as_text(row.get("severity")) for row in rows]
        if trace is not None:
            statuses.append(_as_text(trace.status))
            severities.append(_as_text(trace.severity))
        status = _select_status(statuses)
        severity = _select_severity(severities)
        components = [_as_text(row.get("component")) for row in rows if _as_text(row.get("component"))]
        component = components[0] if components else "unknown"
        latencies = [_as_float(row.get("latency_ms")) for row in rows if _as_float(row.get("latency_ms")) is not None]
        max_latency_ms = max(latencies) if latencies else None
        error_count = sum(1 for row in rows if _as_text(row.get("severity")).lower() == "error")
        warning_count = sum(1 for row in rows if _as_text(row.get("severity")).lower() == "warning")
        summary = _as_text(trace.summary) if trace is not None else ""
        if not summary:
            summary = _as_text(rows[-1].get("snippet"))
        snippet = _as_text(rows[-1].get("snippet")) or summary
        hardness = _compute_hardness(
            status=status,
            severity=severity,
            duration_ms=duration_ms,
            error_count=error_count,
            warning_count=warning_count,
            max_latency_ms=max_latency_ms,
        )

        case_rows.append(
            (
                case_id,
                trace_id,
                req_id,
                first_event_index,
                start_ts,
                end_ts,
                duration_ms,
                status,
                severity,
                component,
                len(rows),
                error_count,
                warning_count,
                max_latency_ms,
                hardness,
                summary,
                snippet,
                "sqlite.cases.v4",
            )
        )

        for row in rows:
            case_event_rows.append(
                (
                    case_id,
                    _as_int(row.get("row_id")) or 0,
                    _as_int(row.get("event_index")) or 0,
                    _as_text(row.get("source")),
                    _as_text(row.get("component")),
                    _as_text(row.get("phase")),
                    _as_text(row.get("status")),
                    _as_text(row.get("severity")),
                    _as_float(row.get("ts_wall_s")),
                    _as_float(row.get("latency_ms")),
                    _as_text(row.get("snippet")),
                )
            )

        for label, score, reason in _case_labels(status=status, severity=severity, hardness=hardness):
            case_label_rows.append((case_id, label, score, reason))

    return case_rows, case_event_rows, case_label_rows


def build_session_db(
    session_dir: str,
    *,
    db_path: str = "",
    replace: bool = True,
) -> Dict[str, Any]:
    root = str(session_dir)
    events_path = os.path.join(root, "events.jsonl")
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"events file missing: {events_path}")

    events = list(_iter_events(events_path))
    manifest = _read_manifest(root)
    traces = _read_trace_records(root, events, manifest=manifest)
    trace_lookup = _trace_ref_lookup(traces)

    target_db = resolve_session_db_path(db_path or root)
    os.makedirs(os.path.dirname(target_db) or ".", exist_ok=True)
    if replace and os.path.exists(target_db):
        os.remove(target_db)

    source_counts: Dict[str, int] = {}
    reasoning_rows: List[Tuple[Any, ...]] = []
    event_rows: List[Tuple[Any, ...]] = []

    for event_index, msg in events:
        source = _as_text(msg.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        level = _as_text(msg.get("level") or "info")
        ts_wall_s = _as_float(msg.get("ts_wall_s"))
        payload = msg.get("payload")
        payload_json = json.dumps(payload if isinstance(payload, dict) else payload, separators=(",", ":"), ensure_ascii=True)

        req_id: Optional[int] = None
        phase = ""
        status = ""
        severity = "info"
        latency_ms: Optional[float] = None
        snippet = ""
        trace_id: Optional[str] = None

        reasoning = normalize_reasoning_message(msg)
        if reasoning is not None:
            req_id = reasoning.req_id
            phase = reasoning.phase
            status = reasoning.status
            severity = reasoning.severity
            latency_ms = reasoning.latency_ms
            snippet = format_reasoning_snippet(reasoning.snippet, max_chars=220, redact=False)
            reasoning_rows.append(
                (
                    int(event_index),
                    source,
                    ts_wall_s,
                    reasoning.req_id,
                    reasoning.phase,
                    reasoning.status,
                    reasoning.severity,
                    reasoning.latency_ms,
                    snippet,
                )
            )
        elif event_index in trace_lookup:
            ref = trace_lookup[event_index]
            req_id = _as_int(ref.get("req_id"))
            phase = _as_text(ref.get("phase"))
            status = _as_text(ref.get("status"))
            severity = _as_text(ref.get("severity") or "info")
            latency_ms = _as_float(ref.get("latency_ms"))
            snippet = _as_text(ref.get("snippet"))
            trace_id = _as_text(ref.get("trace_id")) or None
        elif source == "journal" and isinstance(payload, dict):
            snippet = _as_text(payload.get("line"))
            if "error" in snippet.lower() or "fail" in snippet.lower():
                severity = "error"
            elif "warn" in snippet.lower():
                severity = "warning"

        event_rows.append(
            (
                int(event_index),
                ts_wall_s,
                source,
                level,
                req_id,
                phase,
                status,
                severity,
                latency_ms,
                snippet,
                payload_json,
            )
        )

    reasoning_trace_rows = build_reasoning_trace_rows(events, traces=traces)
    case_rows, case_event_rows, case_label_rows = _build_cases(reasoning_trace_rows=reasoning_trace_rows, traces=traces)

    with sqlite3.connect(target_db) as conn:
        _ensure_schema(conn)
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM reasoning")
        conn.execute("DELETE FROM traces")
        conn.execute("DELETE FROM trace_events")
        conn.execute("DELETE FROM reasoning_traces")
        conn.execute("DELETE FROM cases")
        conn.execute("DELETE FROM case_events")
        conn.execute("DELETE FROM case_labels")
        conn.execute("DELETE FROM meta")

        conn.executemany(
            """
            INSERT INTO events (
                seq, ts_wall_s, source, level, req_id, phase, status, severity, latency_ms, snippet, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            event_rows,
        )
        if reasoning_rows:
            conn.executemany(
                """
                INSERT INTO reasoning (
                    event_index, source, ts_wall_s, req_id, phase, status, severity, latency_ms, snippet
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                reasoning_rows,
            )
        if traces:
            conn.executemany(
                """
                INSERT INTO traces (
                    trace_id, req_id, start_ts_wall_s, end_ts_wall_s, duration_ms, status, severity, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        trace.trace_id,
                        trace.req_id,
                        trace.start_ts_wall_s,
                        trace.end_ts_wall_s,
                        trace.duration_ms,
                        trace.status,
                        trace.severity,
                        trace.summary,
                    )
                    for trace in traces
                ],
            )

        if reasoning_trace_rows:
            conn.executemany(
                """
                INSERT INTO reasoning_traces (
                    row_id, event_index, trace_id, req_id, component, source, ts_wall_s, phase, status, severity,
                    latency_ms, primitive, confidence, target_zone, model, provider, delta_score, snippet,
                    input_preview, output_preview, perception_ts_wall_s, perception_person_conf, perception_zone_hint,
                    perception_frame_id, video_frame_event_index, video_frame_id, video_frame_ref, video_frame_width,
                    video_frame_height
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(row.get("row_id")),
                        int(row.get("event_index")),
                        _as_text(row.get("trace_id")),
                        _as_int(row.get("req_id")),
                        _as_text(row.get("component")),
                        _as_text(row.get("source")),
                        _as_float(row.get("ts_wall_s")),
                        _as_text(row.get("phase")),
                        _as_text(row.get("status")),
                        _as_text(row.get("severity")),
                        _as_float(row.get("latency_ms")),
                        _as_text(row.get("primitive")),
                        _as_float(row.get("confidence")),
                        _as_text(row.get("target_zone")),
                        _as_text(row.get("model")),
                        _as_text(row.get("provider")),
                        _as_float(row.get("delta_score")),
                        _as_text(row.get("snippet")),
                        _as_text(row.get("input_preview")),
                        _as_text(row.get("output_preview")),
                        _as_float(row.get("perception_ts_wall_s")),
                        _as_float(row.get("perception_person_conf")),
                        _as_text(row.get("perception_zone_hint")),
                        _as_int(row.get("perception_frame_id")),
                        _as_int(row.get("video_frame_event_index")),
                        _as_int(row.get("video_frame_id")),
                        _as_text(row.get("video_frame_ref")),
                        _as_int(row.get("video_frame_width")),
                        _as_int(row.get("video_frame_height")),
                    )
                    for row in reasoning_trace_rows
                ],
            )
        if traces:
            conn.executemany(
                """
                INSERT INTO trace_events (
                    trace_id, event_index, source, phase, status, latency_ms, severity, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        trace.trace_id,
                        int(ref.event_index),
                        ref.source,
                        ref.phase,
                        ref.status,
                        ref.latency_ms,
                        ref.severity,
                        ref.summary,
                    )
                    for trace in traces
                    for ref in trace.event_refs
                ],
            )
        if case_rows:
            conn.executemany(
                """
                INSERT INTO cases (
                    case_id, trace_id, req_id, first_event_index, start_ts_wall_s, end_ts_wall_s, duration_ms,
                    status, severity, component, event_count, error_count, warning_count, max_latency_ms, hardness,
                    summary, snippet, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                case_rows,
            )
        if case_event_rows:
            conn.executemany(
                """
                INSERT INTO case_events (
                    case_id, row_id, event_index, source, component, phase, status, severity, ts_wall_s, latency_ms, snippet
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                case_event_rows,
            )
        if case_label_rows:
            conn.executemany(
                """
                INSERT INTO case_labels (
                    case_id, label, score, reason
                ) VALUES (?, ?, ?, ?)
                """,
                case_label_rows,
            )

        now = time.time()
        meta_rows = [
            ("generated_at_wall_s", f"{now:.6f}"),
            ("event_count", str(len(event_rows))),
            ("reasoning_count", str(len(reasoning_rows))),
            ("trace_count", str(len(traces))),
            ("reasoning_trace_count", str(len(reasoning_trace_rows))),
            ("case_count", str(len(case_rows))),
            ("case_label_count", str(len(case_label_rows))),
            ("source_counts_json", json.dumps(source_counts, separators=(",", ":"), ensure_ascii=True)),
        ]
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_rows)
        conn.commit()

    contract = ensure_session_db_contract(target_db)

    return {
        "db_path": target_db,
        "event_count": len(event_rows),
        "reasoning_count": len(reasoning_rows),
        "trace_count": len(traces),
        "reasoning_trace_count": len(reasoning_trace_rows),
        "case_count": len(case_rows),
        "case_label_count": len(case_label_rows),
        "source_counts": source_counts,
        "contract_ok": bool(contract.get("ok", False)),
    }


def _parse_query_tokens(query: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {
        "source": [],
        "severity": [],
        "status": [],
        "phase": [],
        "req": [],
        "trace": [],
        "component": [],
        "kind": [],
        "latency_ms": [],
        "duration_ms": [],
        "ts": [],
        "sort": [],
        "order": [],
        "text": [],
    }
    latency_re = re.compile(r"^latency_ms(<=|>=|<|>)(.+)$", re.IGNORECASE)
    duration_re = re.compile(r"^duration_ms(<=|>=|<|>)(.+)$", re.IGNORECASE)
    try:
        tokens = shlex.split(str(query or ""))
    except ValueError:
        tokens = str(query or "").split()
    for raw in tokens:
        token = raw.strip()
        if not token:
            continue
        if ":" in token:
            key, value = token.split(":", 1)
            key_l = key.strip().lower()
            value = value.strip().strip("'").strip('"')
            if value and key_l in groups:
                values = [v.strip() for v in value.split("|") if v.strip()]
                if not values:
                    values = [value]
                groups[key_l].extend(values)
                continue
        lm = latency_re.match(token)
        if lm:
            groups["latency_ms"].append(f"{lm.group(1)}{lm.group(2).strip()}")
            continue
        dm = duration_re.match(token)
        if dm:
            groups["duration_ms"].append(f"{dm.group(1)}{dm.group(2).strip()}")
            continue
        groups["text"].append(token)
    return groups


def _append_numeric_filters(
    clauses: List[str],
    params: List[Any],
    *,
    column: str,
    filters: Sequence[str],
) -> None:
    for expr in filters:
        text = str(expr).strip()
        if not text:
            continue
        op = None
        for candidate in ("<=", ">=", "<", ">"):
            if text.startswith(candidate):
                op = candidate
                text = text[len(candidate) :].strip()
                break
        if op is None:
            continue
        value = _as_float(text)
        if value is None:
            continue
        clauses.append(f"{column} {op} ?")
        params.append(float(value))


def _append_ts_range(clauses: List[str], params: List[Any], *, column: str, filters: Sequence[str]) -> None:
    for expr in filters:
        text = str(expr).strip()
        if not text:
            continue
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        left, sep, right = text.partition(",")
        if not sep:
            continue
        lo = _as_float(left.strip()) if left.strip() else None
        hi = _as_float(right.strip()) if right.strip() else None
        if lo is not None:
            clauses.append(f"{column} >= ?")
            params.append(float(lo))
        if hi is not None:
            clauses.append(f"{column} <= ?")
            params.append(float(hi))


def _severity_order_expr(column: str) -> str:
    return (
        f"CASE LOWER(COALESCE({column}, 'info')) "
        f"WHEN 'error' THEN 3 "
        f"WHEN 'warning' THEN 2 "
        f"WHEN 'info' THEN 1 "
        f"ELSE 0 END"
    )


def _sort_parts(groups: Mapping[str, List[str]]) -> Tuple[str, str]:
    sort_key = str((groups.get("sort") or [""])[0]).strip().lower()
    order = str((groups.get("order") or ["desc"])[0]).strip().lower()
    order_sql = "ASC" if order == "asc" else "DESC"
    return sort_key, order_sql


def _append_or_equals(clauses: List[str], params: List[Any], column: str, values: Sequence[str]) -> None:
    clean = [str(v) for v in values if str(v)]
    if not clean:
        return
    clauses.append("(" + " OR ".join([f"{column} = ?" for _ in clean]) + ")")
    params.extend(clean)


def _append_or_like(clauses: List[str], params: List[Any], column: str, values: Sequence[str]) -> None:
    clean = [str(v) for v in values if str(v)]
    if not clean:
        return
    clauses.append("(" + " OR ".join([f"{column} LIKE ?" for _ in clean]) + ")")
    params.extend([f"%{v}%" for v in clean])


def query_session_db(
    path: str,
    *,
    query: str,
    limit: int = 20,
) -> Dict[str, Any]:
    db_path = resolve_session_db_path(path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"session db missing: {db_path}")
    groups = _parse_query_tokens(query)
    lim = max(1, int(limit))
    kind_values = {str(v).strip().lower() for v in groups.get("kind", []) if str(v).strip()}
    want_events = (not kind_values) or bool(kind_values & {"event", "events"})
    want_reasoning = (not kind_values) or bool(kind_values & {"reasoning", "reason"})
    want_traces = (not kind_values) or bool(kind_values & {"trace", "traces"})
    want_joined = (not kind_values) or bool(kind_values & {"joined", "reasoning_trace", "reasoning_traces"})

    event_clauses: List[str] = []
    event_params: List[Any] = []
    _append_or_equals(event_clauses, event_params, "source", groups["source"])
    _append_or_equals(event_clauses, event_params, "severity", groups["severity"])
    _append_or_equals(event_clauses, event_params, "status", groups["status"])
    _append_or_equals(event_clauses, event_params, "phase", groups["phase"])
    _append_numeric_filters(event_clauses, event_params, column="latency_ms", filters=groups["latency_ms"])
    _append_ts_range(event_clauses, event_params, column="ts_wall_s", filters=groups["ts"])
    if groups["trace"]:
        sub_clauses: List[str] = []
        sub_params: List[Any] = []
        _append_or_like(sub_clauses, sub_params, "trace_id", groups["trace"])
        if sub_clauses:
            event_clauses.append(
                "seq IN (SELECT event_index FROM trace_events WHERE " + " AND ".join(sub_clauses) + ")"
            )
            event_params.extend(sub_params)
    for value in groups["req"]:
        req_value = _as_int(value)
        if req_value is not None:
            event_clauses.append("req_id = ?")
            event_params.append(req_value)
    for value in groups["text"]:
        event_clauses.append("(snippet LIKE ? OR payload_json LIKE ?)")
        like = f"%{value}%"
        event_params.extend([like, like])

    trace_clauses: List[str] = []
    trace_params: List[Any] = []
    for value in groups["trace"]:
        trace_clauses.append("trace_id LIKE ?")
        trace_params.append(f"%{value}%")
    for value in groups["req"]:
        req_value = _as_int(value)
        if req_value is not None:
            trace_clauses.append("req_id = ?")
            trace_params.append(req_value)
    _append_or_equals(trace_clauses, trace_params, "status", groups["status"])
    _append_or_equals(trace_clauses, trace_params, "severity", groups["severity"])
    _append_numeric_filters(trace_clauses, trace_params, column="duration_ms", filters=groups["duration_ms"])
    _append_ts_range(trace_clauses, trace_params, column="start_ts_wall_s", filters=groups["ts"])
    for value in groups["text"]:
        trace_clauses.append("summary LIKE ?")
        trace_params.append(f"%{value}%")

    reasoning_clauses: List[str] = []
    reasoning_params: List[Any] = []
    for value in groups["req"]:
        req_value = _as_int(value)
        if req_value is not None:
            reasoning_clauses.append("req_id = ?")
            reasoning_params.append(req_value)
    _append_or_equals(reasoning_clauses, reasoning_params, "status", groups["status"])
    _append_or_equals(reasoning_clauses, reasoning_params, "severity", groups["severity"])
    _append_or_equals(reasoning_clauses, reasoning_params, "phase", groups["phase"])
    _append_numeric_filters(reasoning_clauses, reasoning_params, column="latency_ms", filters=groups["latency_ms"])
    _append_ts_range(reasoning_clauses, reasoning_params, column="ts_wall_s", filters=groups["ts"])
    for value in groups["text"]:
        reasoning_clauses.append("snippet LIKE ?")
        reasoning_params.append(f"%{value}%")

    joined_clauses: List[str] = []
    joined_params: List[Any] = []
    _append_or_equals(joined_clauses, joined_params, "source", groups["source"])
    _append_or_equals(joined_clauses, joined_params, "severity", groups["severity"])
    _append_or_equals(joined_clauses, joined_params, "status", groups["status"])
    _append_or_equals(joined_clauses, joined_params, "phase", groups["phase"])
    _append_or_equals(joined_clauses, joined_params, "component", groups["component"])
    _append_numeric_filters(joined_clauses, joined_params, column="latency_ms", filters=groups["latency_ms"])
    _append_ts_range(joined_clauses, joined_params, column="ts_wall_s", filters=groups["ts"])
    for value in groups["trace"]:
        joined_clauses.append("trace_id LIKE ?")
        joined_params.append(f"%{value}%")
    for value in groups["req"]:
        req_value = _as_int(value)
        if req_value is not None:
            joined_clauses.append("req_id = ?")
            joined_params.append(req_value)
    for value in groups["text"]:
        joined_clauses.append("(snippet LIKE ? OR input_preview LIKE ? OR output_preview LIKE ?)")
        like = f"%{value}%"
        joined_params.extend([like, like, like])

    where_events = f"WHERE {' AND '.join(event_clauses)}" if event_clauses else ""
    where_traces = f"WHERE {' AND '.join(trace_clauses)}" if trace_clauses else ""
    where_reasoning = f"WHERE {' AND '.join(reasoning_clauses)}" if reasoning_clauses else ""
    where_joined = f"WHERE {' AND '.join(joined_clauses)}" if joined_clauses else ""
    sort_key, order_sql = _sort_parts(groups)
    if sort_key in {"latency", "latency_ms"}:
        order_events = f"COALESCE(e.latency_ms,-1) {order_sql}, e.seq DESC"
        order_reasoning = f"COALESCE(latency_ms,-1) {order_sql}, event_index DESC"
        order_joined = f"COALESCE(latency_ms,-1) {order_sql}, row_id {order_sql}"
        order_traces = f"COALESCE(duration_ms,-1) {order_sql}, start_ts_wall_s DESC"
    elif sort_key == "severity":
        order_events = f"{_severity_order_expr('e.severity')} {order_sql}, e.seq DESC"
        order_reasoning = f"{_severity_order_expr('severity')} {order_sql}, event_index DESC"
        order_joined = f"{_severity_order_expr('severity')} {order_sql}, row_id {order_sql}"
        order_traces = f"{_severity_order_expr('severity')} {order_sql}, start_ts_wall_s DESC"
    elif sort_key in {"ts", "time", "timestamp"}:
        order_events = f"COALESCE(e.ts_wall_s,0) {order_sql}, e.seq DESC"
        order_reasoning = f"COALESCE(ts_wall_s,0) {order_sql}, event_index DESC"
        order_joined = f"COALESCE(ts_wall_s,0) {order_sql}, row_id {order_sql}"
        order_traces = f"COALESCE(start_ts_wall_s,0) {order_sql}, trace_id DESC"
    else:
        order_events = "e.seq DESC"
        order_reasoning = "event_index DESC"
        order_joined = "row_id DESC"
        order_traces = "start_ts_wall_s DESC"

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if want_events:
            event_count = int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM events e {where_events}",
                    tuple(event_params),
                ).fetchone()["n"]
            )
        else:
            event_count = 0
        if want_events:
            events = [
                dict(row)
                for row in conn.execute(
                    "SELECT e.seq, e.ts_wall_s, e.source, e.req_id, e.phase, e.status, e.severity, e.latency_ms, "
                    "e.snippet, (SELECT te.trace_id FROM trace_events te WHERE te.event_index = e.seq LIMIT 1) AS trace_id "
                    f"FROM events e {where_events} ORDER BY {order_events} LIMIT ?",
                    (*event_params, lim),
                ).fetchall()
            ]
        else:
            events = []
        if want_traces:
            trace_count = int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM traces {where_traces}",
                    tuple(trace_params),
                ).fetchone()["n"]
            )
        else:
            trace_count = 0
        if want_traces:
            traces = [
                dict(row)
                for row in conn.execute(
                    f"SELECT trace_id, req_id, status, severity, duration_ms, summary "
                    f"FROM traces {where_traces} ORDER BY {order_traces} LIMIT ?",
                    (*trace_params, lim),
                ).fetchall()
            ]
        else:
            traces = []
        if want_reasoning:
            reasoning_count = int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM reasoning {where_reasoning}",
                    tuple(reasoning_params),
                ).fetchone()["n"]
            )
        else:
            reasoning_count = 0
        if want_reasoning:
            reasoning = [
                dict(row)
                for row in conn.execute(
                    f"SELECT event_index, ts_wall_s, req_id, phase, status, severity, latency_ms, snippet "
                    f"FROM reasoning {where_reasoning} ORDER BY {order_reasoning} LIMIT ?",
                    (*reasoning_params, lim),
                ).fetchall()
            ]
        else:
            reasoning = []
        if want_joined:
            joined_count = int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM reasoning_traces {where_joined}",
                    tuple(joined_params),
                ).fetchone()["n"]
            )
        else:
            joined_count = 0
        if want_joined:
            joined = [
                dict(row)
                for row in conn.execute(
                    f"SELECT row_id, event_index, trace_id, req_id, component, source, ts_wall_s, phase, status, "
                    f"severity, latency_ms, primitive, confidence, target_zone, model, provider, delta_score, snippet, "
                    f"input_preview, output_preview, perception_person_conf, perception_zone_hint, perception_frame_id, "
                    f"video_frame_id, video_frame_ref "
                    f"FROM reasoning_traces {where_joined} ORDER BY {order_joined} LIMIT ?",
                    (*joined_params, lim),
                ).fetchall()
            ]
        else:
            joined = []
        meta_rows = conn.execute("SELECT key, value FROM meta").fetchall()
    meta = {str(row["key"]): row["value"] for row in meta_rows}
    return {
        "db_path": db_path,
        "query": query,
        "events": events,
        "traces": traces,
        "reasoning": reasoning,
        "joined": joined,
        "counts": {
            "events": event_count,
            "traces": trace_count,
            "reasoning": reasoning_count,
            "joined": joined_count,
        },
        "meta": meta,
    }


def query_cases_db(
    path: str,
    *,
    query: str,
    limit: int = 25,
) -> Dict[str, Any]:
    db_path = resolve_session_db_path(path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"session db missing: {db_path}")
    groups = _parse_query_tokens(query)
    lim = max(1, int(limit))

    clauses: List[str] = []
    params: List[Any] = []
    _append_or_like(clauses, params, "c.trace_id", groups["trace"])
    _append_or_equals(clauses, params, "c.severity", groups["severity"])
    _append_or_equals(clauses, params, "c.status", groups["status"])
    _append_or_equals(clauses, params, "c.component", groups["component"])
    _append_numeric_filters(clauses, params, column="c.max_latency_ms", filters=groups["latency_ms"])
    _append_numeric_filters(clauses, params, column="c.duration_ms", filters=groups["duration_ms"])
    _append_ts_range(clauses, params, column="c.start_ts_wall_s", filters=groups["ts"])
    for value in groups["req"]:
        req_value = _as_int(value)
        if req_value is not None:
            clauses.append("c.req_id = ?")
            params.append(req_value)
    for value in groups["text"]:
        clauses.append("(c.case_id LIKE ? OR c.summary LIKE ? OR c.snippet LIKE ?)")
        like = f"%{value}%"
        params.extend([like, like, like])

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sort_key, order_sql = _sort_parts(groups)
    if sort_key in {"severity"}:
        order_sql_expr = f"{_severity_order_expr('c.severity')} {order_sql}, c.hardness DESC, c.start_ts_wall_s DESC"
    elif sort_key in {"latency", "latency_ms"}:
        order_sql_expr = f"COALESCE(c.max_latency_ms, -1) {order_sql}, c.hardness DESC, c.start_ts_wall_s DESC"
    elif sort_key in {"duration", "duration_ms"}:
        order_sql_expr = f"COALESCE(c.duration_ms, -1) {order_sql}, c.hardness DESC, c.start_ts_wall_s DESC"
    elif sort_key in {"hardness", "score"}:
        order_sql_expr = f"COALESCE(c.hardness, -1) {order_sql}, c.start_ts_wall_s DESC"
    elif sort_key in {"ts", "time", "timestamp"}:
        order_sql_expr = f"COALESCE(c.start_ts_wall_s, 0) {order_sql}, c.hardness DESC"
    else:
        order_sql_expr = "c.hardness DESC, c.start_ts_wall_s DESC"

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        total_count = int(
            conn.execute(
                f"SELECT COUNT(*) AS n FROM cases c {where_sql}",
                tuple(params),
            ).fetchone()["n"]
        )
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT
                    c.case_id,
                    c.trace_id,
                    c.req_id,
                    c.first_event_index,
                    c.start_ts_wall_s,
                    c.end_ts_wall_s,
                    c.duration_ms,
                    c.status,
                    c.severity,
                    c.component,
                    c.event_count,
                    c.error_count,
                    c.warning_count,
                    c.max_latency_ms,
                    c.hardness,
                    c.summary,
                    c.snippet,
                    c.source,
                    r.decision,
                    r.reviewed_at_wall_s,
                    (
                        SELECT GROUP_CONCAT(label, ',')
                        FROM case_labels cl
                        WHERE cl.case_id = c.case_id
                    ) AS labels_csv
                FROM cases c
                LEFT JOIN case_reviews r ON r.case_id = c.case_id
                {where_sql}
                ORDER BY {order_sql_expr}
                LIMIT ?
                """,
                (*params, lim),
            ).fetchall()
        ]
        meta_rows = conn.execute("SELECT key, value FROM meta").fetchall()

    meta = {str(row["key"]): row["value"] for row in meta_rows}
    return {
        "db_path": db_path,
        "query": query,
        "cases": rows,
        "total_count": total_count,
        "meta": meta,
    }


def query_case_detail_db(
    path: str,
    *,
    case_id: str,
    event_limit: int = 200,
) -> Dict[str, Any]:
    db_path = resolve_session_db_path(path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"session db missing: {db_path}")
    case_key = str(case_id or "").strip()
    if not case_key:
        return {"case": None, "events": [], "labels": [], "review": None}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        case_row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_key,)).fetchone()
        labels = [
            dict(row)
            for row in conn.execute(
                "SELECT label, score, reason FROM case_labels WHERE case_id = ? ORDER BY score DESC, label ASC",
                (case_key,),
            ).fetchall()
        ]
        events = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    ce.row_id,
                    ce.event_index,
                    ce.source,
                    ce.component,
                    ce.phase,
                    ce.status,
                    ce.severity,
                    ce.ts_wall_s,
                    ce.latency_ms,
                    ce.snippet,
                    rt.input_preview,
                    rt.output_preview,
                    rt.primitive,
                    rt.confidence,
                    rt.target_zone,
                    rt.model,
                    rt.provider,
                    rt.delta_score,
                    rt.perception_person_conf,
                    rt.perception_zone_hint,
                    rt.perception_frame_id,
                    rt.video_frame_id,
                    rt.video_frame_ref,
                    rt.video_frame_width,
                    rt.video_frame_height
                FROM case_events ce
                LEFT JOIN reasoning_traces rt ON rt.row_id = ce.row_id
                WHERE ce.case_id = ?
                ORDER BY ce.row_id DESC
                LIMIT ?
                """,
                (case_key, max(1, int(event_limit))),
            ).fetchall()
        ]
        review_row = conn.execute(
            "SELECT decision, note, reviewer, reviewed_at_wall_s FROM case_reviews WHERE case_id = ?",
            (case_key,),
        ).fetchone()
    contexts: Dict[str, Any] = {
        "env_context": {},
        "planner_context": {},
        "arbiter_context": {},
        "perception_context": {},
        "video_context": {"frames": []},
    }
    latest_by_component: Dict[str, Dict[str, Any]] = {}
    seen_video: set[tuple[Any, Any, Any, Any]] = set()
    for item in reversed(events):
        if not isinstance(item, dict):
            continue
        component = str(item.get("component") or "").strip().lower()
        if component and component not in latest_by_component:
            latest_by_component[component] = {
                "row_id": item.get("row_id"),
                "event_index": item.get("event_index"),
                "phase": item.get("phase"),
                "status": item.get("status"),
                "severity": item.get("severity"),
                "latency_ms": item.get("latency_ms"),
                "snippet": item.get("snippet"),
                "input_preview": item.get("input_preview"),
                "output_preview": item.get("output_preview"),
                "primitive": item.get("primitive"),
                "confidence": item.get("confidence"),
                "target_zone": item.get("target_zone"),
                "delta_score": item.get("delta_score"),
                "model": item.get("model"),
                "provider": item.get("provider"),
            }

        if not contexts["perception_context"]:
            person_conf = item.get("perception_person_conf")
            zone_hint = item.get("perception_zone_hint")
            frame_id = item.get("perception_frame_id")
            if person_conf is not None or zone_hint or frame_id is not None:
                contexts["perception_context"] = {
                    "person_conf": person_conf,
                    "zone_hint": zone_hint,
                    "frame_id": frame_id,
                }

        video_ref = str(item.get("video_frame_ref") or "")
        video_id = item.get("video_frame_id")
        if video_ref or video_id is not None:
            key = (video_ref, video_id, item.get("video_frame_width"), item.get("video_frame_height"))
            if key not in seen_video:
                seen_video.add(key)
                contexts["video_context"]["frames"].append(
                    {
                        "frame_ref": video_ref,
                        "frame_id": video_id,
                        "width": item.get("video_frame_width"),
                        "height": item.get("video_frame_height"),
                    }
                )

    contexts["env_context"] = latest_by_component.get("env_processor", {})
    contexts["planner_context"] = latest_by_component.get("planner", {})
    contexts["arbiter_context"] = latest_by_component.get("arbiter", {})

    return {
        "case": dict(case_row) if case_row is not None else None,
        "events": events,
        "labels": labels,
        "review": dict(review_row) if review_row is not None else None,
        "contexts": contexts,
    }


def review_case(
    path: str,
    *,
    case_id: str,
    decision: str,
    note: str = "",
    reviewer: str = "viewer",
) -> Dict[str, Any]:
    db_path = resolve_session_db_path(path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"session db missing: {db_path}")
    case_key = str(case_id or "").strip()
    if not case_key:
        raise ValueError("case_id is required")
    decision_key = str(decision or "").strip().lower()
    if decision_key not in {"accept", "reject", "needs_context", "label"}:
        raise ValueError(f"unsupported decision: {decision}")
    reviewed_at = time.time()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO case_reviews (case_id, decision, note, reviewer, reviewed_at_wall_s)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
                decision=excluded.decision,
                note=excluded.note,
                reviewer=excluded.reviewer,
                reviewed_at_wall_s=excluded.reviewed_at_wall_s
            """,
            (case_key, decision_key, str(note or ""), str(reviewer or "viewer"), float(reviewed_at)),
        )
        conn.commit()
    return {
        "db_path": db_path,
        "case_id": case_key,
        "decision": decision_key,
        "note": str(note or ""),
        "reviewer": str(reviewer or "viewer"),
        "reviewed_at_wall_s": float(reviewed_at),
    }



def validate_session_db_contract(path: str) -> Dict[str, Any]:
    db_path = resolve_session_db_path(path)
    errors: List[str] = []
    warnings: List[str] = []
    if not os.path.exists(db_path):
        return {
            "ok": False,
            "db_path": db_path,
            "errors": ["session_db_missing"],
            "warnings": [],
        }

    table_names: set[str] = set()
    case_count = 0
    bad_case_sources: List[str] = []
    meta_keys: set[str] = set()
    try:
        with sqlite3.connect(db_path) as conn:
            table_names = {
                str(row[0])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if row and isinstance(row[0], str)
            }
            missing_tables = sorted(_SESSION_CONTRACT_REQUIRED_TABLES - table_names)
            if missing_tables:
                errors.append("missing_tables=" + ",".join(missing_tables))

            if "cases" in table_names:
                case_count = int(conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0])
                bad_case_sources = [
                    str(row[0])
                    for row in conn.execute(
                        "SELECT DISTINCT source FROM cases WHERE COALESCE(source, '') != 'sqlite.cases.v4'"
                    ).fetchall()
                    if row and row[0] is not None
                ]
                if bad_case_sources:
                    errors.append("case_source_mismatch=" + ",".join(sorted(set(bad_case_sources))))

            if "meta" in table_names:
                meta_keys = {
                    str(row[0])
                    for row in conn.execute("SELECT key FROM meta").fetchall()
                    if row and isinstance(row[0], str)
                }
    except Exception as exc:
        errors.append(f"db_open_failed={exc!r}")

    required_meta = {"event_count", "reasoning_count", "trace_count", "reasoning_trace_count", "case_count"}
    missing_meta = sorted(required_meta - meta_keys)
    if missing_meta:
        warnings.append("missing_meta=" + ",".join(missing_meta))

    return {
        "ok": len(errors) == 0,
        "db_path": db_path,
        "errors": errors,
        "warnings": warnings,
        "case_count": int(case_count),
    }


def ensure_session_db_contract(path: str) -> Dict[str, Any]:
    report = validate_session_db_contract(path)
    if not bool(report.get("ok")):
        details = "; ".join(str(item) for item in report.get("errors", []))
        raise ValueError(f"session db contract failed: {details}")
    return report
