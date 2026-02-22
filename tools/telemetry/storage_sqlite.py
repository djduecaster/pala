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

    with sqlite3.connect(target_db) as conn:
        _ensure_schema(conn)
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM reasoning")
        conn.execute("DELETE FROM traces")
        conn.execute("DELETE FROM trace_events")
        conn.execute("DELETE FROM reasoning_traces")
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

        now = time.time()
        meta_rows = [
            ("generated_at_wall_s", f"{now:.6f}"),
            ("event_count", str(len(event_rows))),
            ("reasoning_count", str(len(reasoning_rows))),
            ("trace_count", str(len(traces))),
            ("reasoning_trace_count", str(len(reasoning_trace_rows))),
            ("source_counts_json", json.dumps(source_counts, separators=(",", ":"), ensure_ascii=True)),
        ]
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_rows)
        conn.commit()

    return {
        "db_path": target_db,
        "event_count": len(event_rows),
        "reasoning_count": len(reasoning_rows),
        "trace_count": len(traces),
        "reasoning_trace_count": len(reasoning_trace_rows),
        "source_counts": source_counts,
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
        order_joined = f"COALESCE(latency_ms,-1) {order_sql}, row_id DESC"
        order_traces = f"COALESCE(duration_ms,-1) {order_sql}, start_ts_wall_s DESC"
    elif sort_key == "severity":
        order_events = f"{_severity_order_expr('e.severity')} {order_sql}, e.seq DESC"
        order_reasoning = f"{_severity_order_expr('severity')} {order_sql}, event_index DESC"
        order_joined = f"{_severity_order_expr('severity')} {order_sql}, row_id DESC"
        order_traces = f"{_severity_order_expr('severity')} {order_sql}, start_ts_wall_s DESC"
    elif sort_key in {"ts", "time", "timestamp"}:
        order_events = f"COALESCE(e.ts_wall_s,0) {order_sql}, e.seq DESC"
        order_reasoning = f"COALESCE(ts_wall_s,0) {order_sql}, event_index DESC"
        order_joined = f"COALESCE(ts_wall_s,0) {order_sql}, row_id DESC"
        order_traces = f"COALESCE(start_ts_wall_s,0) {order_sql}, trace_id DESC"
    else:
        order_events = "e.seq DESC"
        order_reasoning = "event_index DESC"
        order_joined = "row_id DESC"
        order_traces = "start_ts_wall_s DESC"

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
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
        "meta": meta,
    }
