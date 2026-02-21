from __future__ import annotations

import json
import os
import shlex
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .reasoning import format_reasoning_snippet, normalize_reasoning_message
from .schema_v3 import SESSION_DB_PATH
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

    with sqlite3.connect(target_db) as conn:
        _ensure_schema(conn)
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM reasoning")
        conn.execute("DELETE FROM traces")
        conn.execute("DELETE FROM trace_events")
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
            ("source_counts_json", json.dumps(source_counts, separators=(",", ":"), ensure_ascii=True)),
        ]
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_rows)
        conn.commit()

    return {
        "db_path": target_db,
        "event_count": len(event_rows),
        "reasoning_count": len(reasoning_rows),
        "trace_count": len(traces),
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
        "since": [],
        "kind": [],
        "text": [],
    }
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
            value = value.strip()
            if value and key_l in groups:
                groups[key_l].append(value)
                continue
        groups["text"].append(token)
    return groups


def _parse_since_seconds(raw_values: Sequence[str]) -> Optional[float]:
    seconds: List[float] = []
    for raw in raw_values:
        text = str(raw or "").strip().lower()
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
            value = float(text)
        except ValueError:
            continue
        if value <= 0.0:
            continue
        seconds.append(value * scale)
    if not seconds:
        return None
    return min(seconds)


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
    since_s = _parse_since_seconds(groups.get("since", []))
    since_cutoff = (time.time() - since_s) if since_s is not None else None
    kind_values = {str(v).strip().lower() for v in groups.get("kind", []) if str(v).strip()}
    want_events = (not kind_values) or bool(kind_values & {"event", "events"})
    want_reasoning = (not kind_values) or bool(kind_values & {"reasoning", "reason"})
    want_traces = (not kind_values) or bool(kind_values & {"trace", "traces"})

    event_clauses: List[str] = []
    event_params: List[Any] = []
    _append_or_equals(event_clauses, event_params, "source", groups["source"])
    _append_or_equals(event_clauses, event_params, "severity", groups["severity"])
    _append_or_equals(event_clauses, event_params, "status", groups["status"])
    _append_or_equals(event_clauses, event_params, "phase", groups["phase"])
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
    if since_cutoff is not None:
        event_clauses.append("ts_wall_s >= ?")
        event_params.append(float(since_cutoff))

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
    for value in groups["text"]:
        trace_clauses.append("summary LIKE ?")
        trace_params.append(f"%{value}%")
    if since_cutoff is not None:
        trace_clauses.append("COALESCE(end_ts_wall_s, start_ts_wall_s, 0) >= ?")
        trace_params.append(float(since_cutoff))

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
    for value in groups["text"]:
        reasoning_clauses.append("snippet LIKE ?")
        reasoning_params.append(f"%{value}%")
    if since_cutoff is not None:
        reasoning_clauses.append("ts_wall_s >= ?")
        reasoning_params.append(float(since_cutoff))

    where_events = f"WHERE {' AND '.join(event_clauses)}" if event_clauses else ""
    where_traces = f"WHERE {' AND '.join(trace_clauses)}" if trace_clauses else ""
    where_reasoning = f"WHERE {' AND '.join(reasoning_clauses)}" if reasoning_clauses else ""

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if want_events:
            events = [
                dict(row)
                for row in conn.execute(
                    "SELECT e.seq, e.ts_wall_s, e.source, e.req_id, e.phase, e.status, e.severity, e.latency_ms, "
                    "e.snippet, (SELECT te.trace_id FROM trace_events te WHERE te.event_index = e.seq LIMIT 1) AS trace_id "
                    f"FROM events e {where_events} ORDER BY e.seq DESC LIMIT ?",
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
                    f"FROM traces {where_traces} ORDER BY start_ts_wall_s DESC LIMIT ?",
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
                    f"FROM reasoning {where_reasoning} ORDER BY event_index DESC LIMIT ?",
                    (*reasoning_params, lim),
                ).fetchall()
            ]
        else:
            reasoning = []
        meta_rows = conn.execute("SELECT key, value FROM meta").fetchall()
        event_count = (
            int(conn.execute(f"SELECT COUNT(*) FROM events e {where_events}", tuple(event_params)).fetchone()[0])
            if want_events
            else 0
        )
        trace_count = (
            int(conn.execute(f"SELECT COUNT(*) FROM traces {where_traces}", tuple(trace_params)).fetchone()[0])
            if want_traces
            else 0
        )
        reasoning_count = (
            int(conn.execute(f"SELECT COUNT(*) FROM reasoning {where_reasoning}", tuple(reasoning_params)).fetchone()[0])
            if want_reasoning
            else 0
        )
    meta = {str(row["key"]): row["value"] for row in meta_rows}
    return {
        "db_path": db_path,
        "query": query,
        "events": events,
        "traces": traces,
        "reasoning": reasoning,
        "counts": {
            "events": event_count,
            "traces": trace_count,
            "reasoning": reasoning_count,
        },
        "meta": meta,
    }
