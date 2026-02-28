from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import os
import re
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TraceEventRef:
    event_index: int
    source: str
    ts_wall_s: Optional[float]
    req_id: Optional[int]
    phase: Optional[str]
    status: Optional[str]
    latency_ms: Optional[float]
    severity: str
    summary: str


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    req_id: Optional[int]
    start_ts_wall_s: Optional[float]
    end_ts_wall_s: Optional[float]
    duration_ms: Optional[float]
    status: str
    severity: str
    summary: str
    event_refs: Tuple[TraceEventRef, ...]


_REQ_RE = re.compile(r"(?i)\b(?:req(?:uest)?[_\s-]?id)\s*[:=]\s*(\d+)\b")
_REQ_WORD_RE = re.compile(r"(?i)\breq(?:uest)?\s*#?\s*(\d+)\b")
_GENERIC_ID_RE = re.compile(r"(?i)\bid\s*[:=]\s*(\d+)\b")

_ERROR_HINTS = ("parse_fail", "invalid", "error", "timeout", "stale", "no_content", "crash", "fail")
_WARN_HINTS = ("warn", "retry", "drop")

_STATUS_RANK = {
    "unknown": 0,
    "ok": 1,
    "warning": 2,
    "stale": 3,
    "timeout": 4,
    "no_content": 5,
    "invalid": 6,
    "parse_fail": 7,
    "error": 8,
}
_SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truncate(text: str, max_len: int = 180) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _extract_req_id_from_text(text: str) -> Optional[int]:
    match = _REQ_RE.search(text)
    if match:
        return _as_optional_int(match.group(1))
    match = _REQ_WORD_RE.search(text)
    if match:
        return _as_optional_int(match.group(1))
    lowered = text.lower()
    if "request" in lowered or "orchestrator" in lowered or " req " in f" {lowered} ":
        match = _GENERIC_ID_RE.search(text)
        if match:
            return _as_optional_int(match.group(1))
    return None


def _normalize_status(raw: Optional[str]) -> str:
    text = (raw or "").strip().lower()
    if not text:
        return "unknown"
    for key in ("parse_fail", "no_content", "timeout", "stale", "invalid", "error"):
        if key in text:
            return key
    if text in {"ok", "success"}:
        return "ok"
    if "warn" in text:
        return "warning"
    return text


def _status_from_text(text: str) -> str:
    lowered = text.lower()
    for key in ("parse_fail", "no_content", "timeout", "stale", "invalid"):
        if key in lowered:
            return key
    if "error" in lowered or "fail" in lowered or "traceback" in lowered:
        return "error"
    if "warn" in lowered or "retry" in lowered:
        return "warning"
    return "unknown"


def _severity_from_fields(*parts: Optional[str]) -> str:
    text = " ".join(p for p in parts if p).lower()
    if any(hint in text for hint in _ERROR_HINTS):
        return "error"
    if any(hint in text for hint in _WARN_HINTS):
        return "warning"
    return "info"


def _make_event_ref(msg: Dict[str, Any], event_index: int) -> Optional[TraceEventRef]:
    source = _as_optional_str(msg.get("source"))
    if not source:
        return None
    ts_wall_s = _as_optional_float(msg.get("ts_wall_s"))
    level = _as_optional_str(msg.get("level")) or ""
    payload = _as_dict(msg.get("payload"))

    req_id: Optional[int] = None
    phase: Optional[str] = None
    status: Optional[str] = None
    latency_ms: Optional[float] = None
    summary = ""

    if source == "timeline_log":
        data = _as_dict(payload.get("data"))
        if not data:
            return None
        dp = _as_dict(data.get("payload"))
        req_id = _as_optional_int(
            dp.get("request_id") if dp.get("request_id") is not None else dp.get("req_id")
        )
        if req_id is None:
            req_id = _as_optional_int(dp.get("id"))
        if req_id is None:
            req_id = _as_optional_int(data.get("request_id"))
        if req_id is None:
            req_id = _as_optional_int(data.get("req_id"))
        if req_id is None:
            req_id = _as_optional_int(data.get("id"))
        phase = _as_optional_str(data.get("type")) or "timeline"
        status = _as_optional_str(dp.get("status")) or _as_optional_str(data.get("status"))
        latency_ms = _as_optional_float(dp.get("latency_ms")) or _as_optional_float(data.get("latency_ms"))
        summary = (
            _as_optional_str(dp.get("reasoning"))
            or _as_optional_str(dp.get("rationale"))
            or _as_optional_str(dp.get("preview"))
            or _as_optional_str(dp.get("detail"))
            or _as_optional_str(dp.get("message"))
            or ""
        )
    elif source == "actions_log":
        data = _as_dict(payload.get("data"))
        if not data:
            return None
        req_id = _as_optional_int(data.get("request_id"))
        if req_id is None:
            req_id = _as_optional_int(data.get("req_id"))
        phase = "action_plan"
        status = _as_optional_str(data.get("status")) or "ok"
        primitive = _as_optional_str(data.get("primitive"))
        confidence = data.get("confidence")
        if primitive and confidence is not None:
            summary = f"primitive={primitive} confidence={confidence}"
        elif primitive:
            summary = f"primitive={primitive}"
        else:
            summary = _as_optional_str(data.get("explanation")) or ""
    elif source == "agent":
        detail = _as_optional_str(payload.get("error"))
        if not detail:
            return None
        req_id = _extract_req_id_from_text(detail)
        phase = "agent_error"
        status = "error"
        summary = detail
    elif source == "journal":
        line = _as_optional_str(payload.get("line"))
        if not line:
            return None
        req_id = _extract_req_id_from_text(line)
        phase = "journal"
        status = _status_from_text(line)
        summary = line
    elif source in {"behavior_env_log", "behavior_env"}:
        data = _as_dict(payload.get("data"))
        if not data:
            data = payload
        if not data:
            return None
        req_id = _as_optional_int(data.get("request_id"))
        if req_id is None:
            req_id = _as_optional_int(data.get("req_id"))
        if req_id is None:
            req_id = _as_optional_int(data.get("id"))
        phase = _as_optional_str(data.get("phase")) or _as_optional_str(data.get("stage")) or "env_processor"
        status = _as_optional_str(data.get("status")) or _as_optional_str(data.get("parse_status")) or "ok"
        latency_ms = _as_optional_float(data.get("latency_ms")) or _as_optional_float(data.get("duration_ms"))
        summary = (
            _as_optional_str(data.get("summary"))
            or _as_optional_str(data.get("events"))
            or _as_optional_str(data.get("message"))
            or ""
        )
    elif source in {"behavior_planner_log", "behavior_planner"}:
        data = _as_dict(payload.get("data"))
        if not data:
            data = payload
        if not data:
            return None
        req_id = _as_optional_int(data.get("request_id"))
        if req_id is None:
            req_id = _as_optional_int(data.get("req_id"))
        if req_id is None:
            req_id = _as_optional_int(data.get("id"))
        phase = _as_optional_str(data.get("phase")) or _as_optional_str(data.get("stage")) or "planner"
        status = _as_optional_str(data.get("status")) or _as_optional_str(data.get("parse_status")) or "ok"
        latency_ms = _as_optional_float(data.get("latency_ms")) or _as_optional_float(data.get("duration_ms"))
        summary = (
            _as_optional_str(data.get("rationale_short"))
            or _as_optional_str(data.get("explanation"))
            or _as_optional_str(data.get("message"))
            or ""
        )
    elif source in {"behavior_reasoning_log", "behavior_reasoning"}:
        data = _as_dict(payload.get("data"))
        if not data:
            data = payload
        if not data:
            return None
        req_id = _as_optional_int(data.get("request_id"))
        if req_id is None:
            req_id = _as_optional_int(data.get("req_id"))
        if req_id is None:
            req_id = _as_optional_int(data.get("id"))
        phase = _as_optional_str(data.get("phase")) or _as_optional_str(data.get("module")) or "behavior_reasoning"
        status = _as_optional_str(data.get("status")) or _as_optional_str(data.get("result")) or "ok"
        latency_ms = _as_optional_float(data.get("latency_ms")) or _as_optional_float(data.get("duration_ms"))
        summary = (
            _as_optional_str(data.get("summary"))
            or _as_optional_str(data.get("reasoning"))
            or _as_optional_str(data.get("message"))
            or ""
        )
    elif source in {"behavior_trace_log", "behavior_trace"}:
        data = _as_dict(payload.get("data"))
        if not data:
            data = payload
        if not data:
            return None
        decision = _as_dict(data.get("decision"))
        mode_transition = _as_dict(data.get("mode_transition"))
        req_id = _as_optional_int(data.get("request_id"))
        if req_id is None:
            req_id = _as_optional_int(data.get("req_id"))
        if req_id is None:
            req_id = _as_optional_int(data.get("id"))
        phase = _as_optional_str(data.get("mode")) or _as_optional_str(mode_transition.get("to")) or "behavior_trace"
        committed = decision.get("committed")
        if isinstance(committed, bool):
            status = "committed" if committed else "no_commit"
        else:
            status = _as_optional_str(decision.get("status")) or _as_optional_str(data.get("status")) or "trace"
        latency_ms = _as_optional_float(data.get("latency_ms")) or _as_optional_float(data.get("duration_ms"))
        summary = _as_optional_str(decision.get("reason")) or _as_optional_str(mode_transition.get("reason")) or ""
        if not summary:
            top = data.get("top_candidates")
            if isinstance(top, list) and top and isinstance(top[0], dict):
                top0 = top[0]
                primitive = _as_optional_str(top0.get("primitive"))
                intent = _as_optional_str(top0.get("intent"))
                utility = _as_optional_float(top0.get("utility"))
                parts = [part for part in [primitive, intent] if part]
                if utility is not None:
                    parts.append(f"utility={utility:.3f}")
                summary = " ".join(parts)
        if not summary:
            summary = _as_optional_str(data.get("mode")) or "behavior_trace"
    else:
        return None

    norm_status = _normalize_status(status)
    severity = _severity_from_fields(level, phase, norm_status, summary)
    return TraceEventRef(
        event_index=int(event_index),
        source=source,
        ts_wall_s=ts_wall_s,
        req_id=req_id,
        phase=phase,
        status=norm_status,
        latency_ms=latency_ms,
        severity=severity,
        summary=_truncate(summary),
    )


@dataclass
class _TraceBucket:
    trace_id: str
    req_id: Optional[int]
    events: List[TraceEventRef] = field(default_factory=list)
    last_ts_wall_s: Optional[float] = None

    def add(self, event: TraceEventRef) -> None:
        self.events.append(event)
        if event.ts_wall_s is not None:
            self.last_ts_wall_s = float(event.ts_wall_s)


class TraceGraphBuilder:
    """Best-effort request correlation for timeline/actions/agent/journal events."""

    def __init__(self, *, match_window_s: float = 2.0, max_events: int = 1000) -> None:
        self._match_window_s = max(0.1, float(match_window_s))
        self._events: Deque[TraceEventRef] = deque(maxlen=max(64, int(max_events)))
        self._next_event_index = 0
        self._cached_traces: List[TraceRecord] = []
        self._dirty = True

    @property
    def match_window_s(self) -> float:
        return self._match_window_s

    def ingest(self, msg: Dict[str, Any]) -> bool:
        event_index = _as_optional_int(msg.get("seq"))
        if event_index is None:
            event_index = self._next_event_index
            self._next_event_index += 1
        else:
            self._next_event_index = max(self._next_event_index, event_index + 1)
        ref = _make_event_ref(msg, event_index=event_index)
        if ref is None:
            return False
        self._events.append(ref)
        self._dirty = True
        return True

    def traces(self) -> List[TraceRecord]:
        if self._dirty:
            self._cached_traces = self._rebuild_traces(list(self._events))
            self._dirty = False
        return list(self._cached_traces)

    def build_trace_index(self) -> Dict[str, Any]:
        traces = self.traces()
        return {
            "version": 1,
            "generated_at_wall_s": time.time(),
            "match_window_s": self._match_window_s,
            "traces": [_trace_to_dict(trace) for trace in traces],
        }

    def _rebuild_traces(self, events: Sequence[TraceEventRef]) -> List[TraceRecord]:
        if not events:
            return []
        ordered = sorted(
            events,
            key=lambda ev: (
                float(ev.ts_wall_s) if ev.ts_wall_s is not None else float("inf"),
                ev.event_index,
            ),
        )
        req_buckets: Dict[int, List[_TraceBucket]] = defaultdict(list)
        inferred: List[_TraceBucket] = []

        def _can_attach(bucket: _TraceBucket, ev: TraceEventRef) -> bool:
            if ev.ts_wall_s is None or bucket.last_ts_wall_s is None:
                return True
            return abs(float(ev.ts_wall_s) - float(bucket.last_ts_wall_s)) <= self._match_window_s

        def _new_req_bucket(req_id: int) -> _TraceBucket:
            bucket_idx = len(req_buckets[req_id]) + 1
            trace_id = f"req:{req_id}" if bucket_idx == 1 else f"req:{req_id}:{bucket_idx}"
            bucket = _TraceBucket(trace_id=trace_id, req_id=req_id)
            return bucket

        def _new_inferred_bucket(ev: TraceEventRef) -> _TraceBucket:
            ts_ms = int((ev.ts_wall_s or 0.0) * 1000.0)
            trace_id = f"time:{ts_ms}-{len(inferred) + 1}"
            bucket = _TraceBucket(trace_id=trace_id, req_id=None)
            inferred.append(bucket)
            return bucket

        for ev in ordered:
            if ev.req_id is not None:
                buckets = req_buckets[ev.req_id]
                if not buckets or not _can_attach(buckets[-1], ev):
                    buckets.append(_new_req_bucket(ev.req_id))
                buckets[-1].add(ev)
                continue

            # Attach missing req_id events to the nearest req bucket first.
            best_bucket = None
            best_delta = None
            if ev.ts_wall_s is not None:
                for buckets in req_buckets.values():
                    if not buckets:
                        continue
                    candidate = buckets[-1]
                    if candidate.last_ts_wall_s is None:
                        continue
                    delta = abs(float(ev.ts_wall_s) - float(candidate.last_ts_wall_s))
                    if delta <= self._match_window_s and (best_delta is None or delta < best_delta):
                        best_delta = delta
                        best_bucket = candidate

            if best_bucket is None:
                if inferred and _can_attach(inferred[-1], ev):
                    best_bucket = inferred[-1]
                else:
                    best_bucket = _new_inferred_bucket(ev)
            best_bucket.add(ev)

        all_buckets: List[_TraceBucket] = []
        for req_id in sorted(req_buckets):
            all_buckets.extend(req_buckets[req_id])
        all_buckets.extend(inferred)

        traces = [_bucket_to_trace(bucket) for bucket in all_buckets if bucket.events]
        traces.sort(
            key=lambda trace: (
                -_SEVERITY_RANK.get(trace.severity, 0),
                -(trace.end_ts_wall_s or 0.0),
                trace.trace_id,
            )
        )
        return traces


def _bucket_to_trace(bucket: _TraceBucket) -> TraceRecord:
    ordered = sorted(bucket.events, key=lambda ev: ev.event_index)
    ts_values = [float(ev.ts_wall_s) for ev in ordered if ev.ts_wall_s is not None]
    start_ts = min(ts_values) if ts_values else None
    end_ts = max(ts_values) if ts_values else None
    duration_ms = ((end_ts - start_ts) * 1000.0) if (start_ts is not None and end_ts is not None) else None

    status = "unknown"
    severity = "info"
    summary = ""
    for ev in ordered:
        candidate_status = _normalize_status(ev.status)
        if _STATUS_RANK.get(candidate_status, 0) >= _STATUS_RANK.get(status, 0):
            status = candidate_status
        if _SEVERITY_RANK.get(ev.severity, 0) >= _SEVERITY_RANK.get(severity, 0):
            severity = ev.severity
        if ev.summary:
            summary = ev.summary
    if not summary:
        last = ordered[-1]
        summary = f"{last.source}:{last.phase or '-'} status={last.status or 'unknown'}"

    return TraceRecord(
        trace_id=bucket.trace_id,
        req_id=bucket.req_id,
        start_ts_wall_s=start_ts,
        end_ts_wall_s=end_ts,
        duration_ms=duration_ms,
        status=status,
        severity=severity,
        summary=_truncate(summary),
        event_refs=tuple(ordered),
    )


def _trace_to_dict(trace: TraceRecord) -> Dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "req_id": trace.req_id,
        "start_ts_wall_s": trace.start_ts_wall_s,
        "end_ts_wall_s": trace.end_ts_wall_s,
        "duration_ms": trace.duration_ms,
        "status": trace.status,
        "severity": trace.severity,
        "summary": trace.summary,
        "event_refs": [
            {
                "event_index": ref.event_index,
                "source": ref.source,
                "phase": ref.phase,
                "status": ref.status,
                "latency_ms": ref.latency_ms,
                "ts_wall_s": ref.ts_wall_s,
                "req_id": ref.req_id,
                "severity": ref.severity,
                "summary": ref.summary,
            }
            for ref in trace.event_refs
        ],
    }


def _trace_from_dict(data: Dict[str, Any]) -> Optional[TraceRecord]:
    trace_id = _as_optional_str(data.get("trace_id"))
    if not trace_id:
        return None
    refs_raw = data.get("event_refs")
    if not isinstance(refs_raw, list):
        refs_raw = []
    refs: List[TraceEventRef] = []
    for raw in refs_raw:
        if not isinstance(raw, dict):
            continue
        event_index = _as_optional_int(raw.get("event_index"))
        source = _as_optional_str(raw.get("source"))
        if event_index is None or not source:
            continue
        refs.append(
            TraceEventRef(
                event_index=event_index,
                source=source,
                ts_wall_s=_as_optional_float(raw.get("ts_wall_s")),
                req_id=_as_optional_int(raw.get("req_id")),
                phase=_as_optional_str(raw.get("phase")),
                status=_as_optional_str(raw.get("status")),
                latency_ms=_as_optional_float(raw.get("latency_ms")),
                severity=_as_optional_str(raw.get("severity")) or "info",
                summary=_as_optional_str(raw.get("summary")) or "",
            )
        )
    return TraceRecord(
        trace_id=trace_id,
        req_id=_as_optional_int(data.get("req_id")),
        start_ts_wall_s=_as_optional_float(data.get("start_ts_wall_s")),
        end_ts_wall_s=_as_optional_float(data.get("end_ts_wall_s")),
        duration_ms=_as_optional_float(data.get("duration_ms")),
        status=_normalize_status(_as_optional_str(data.get("status"))),
        severity=_as_optional_str(data.get("severity")) or "info",
        summary=_as_optional_str(data.get("summary")) or "",
        event_refs=tuple(sorted(refs, key=lambda ref: ref.event_index)),
    )


def load_trace_index(path: str) -> List[TraceRecord]:
    with open(path, "r", encoding="utf-8") as fh:
        decoded = json.load(fh)
    if not isinstance(decoded, dict):
        return []
    items = decoded.get("traces")
    if not isinstance(items, list):
        return []
    traces: List[TraceRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        trace = _trace_from_dict(item)
        if trace is not None:
            traces.append(trace)
    traces.sort(
        key=lambda trace: (
            -_SEVERITY_RANK.get(trace.severity, 0),
            -(trace.end_ts_wall_s or 0.0),
            trace.trace_id,
        )
    )
    return traces


def resolve_trace_index_path(session_dir: str, manifest: Optional[Dict[str, Any]]) -> str:
    rel = "trace_index.json"
    if isinstance(manifest, dict):
        override = _as_optional_str(manifest.get("trace_index_path"))
        if override:
            rel = override
    if os.path.isabs(rel):
        return rel
    return os.path.join(session_dir, rel)
