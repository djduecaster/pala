from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .reasoning import ReasoningEvent, format_reasoning_snippet, normalize_reasoning_message
from .schema_v3 import REASONING_TRACE_INDEX_PATH
from .trace_graph import TraceRecord, load_trace_index, resolve_trace_index_path


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _trace_lookup(traces: Sequence[TraceRecord]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for trace in traces:
        for ref in trace.event_refs:
            out[int(ref.event_index)] = {
                "trace_id": trace.trace_id,
                "req_id": ref.req_id if ref.req_id is not None else trace.req_id,
                "phase": ref.phase,
                "status": ref.status,
                "severity": ref.severity,
            }
    return out


def _extract_perception_snapshot(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = _as_dict(msg.get("payload"))
    data = _as_dict(payload.get("data"))
    if not data:
        data = payload
    if not isinstance(data, dict) or not data:
        return None
    debug = _as_dict(data.get("debug"))
    return {
        "ts_wall_s": _as_optional_float(msg.get("ts_wall_s")),
        "person_conf": _as_optional_float(data.get("primary_person_conf")),
        "zone_hint": _as_optional_str(debug.get("zone_hint")),
        "frame_id": _as_optional_int(data.get("frame_id")),
        "fps": _as_optional_float(data.get("fps")),
        "latency_ms": _as_optional_float(data.get("latency_ms")),
    }


def _extract_video_snapshot(msg: Dict[str, Any], event_index: int) -> Optional[Dict[str, Any]]:
    payload = _as_dict(msg.get("payload"))
    if not payload:
        return None
    return {
        "event_index": int(event_index),
        "ts_wall_s": _as_optional_float(msg.get("ts_wall_s")),
        "frame_id": _as_optional_int(payload.get("frame_id")),
        "frame_ref": _as_optional_str(payload.get("frame_ref")),
        "width": _as_optional_int(payload.get("width")),
        "height": _as_optional_int(payload.get("height")),
    }


def _payload_input_preview(msg: Dict[str, Any]) -> str:
    payload = _as_dict(msg.get("payload"))
    data = _as_dict(payload.get("data"))
    if not data:
        data = payload
    for key in ("request_text", "prompt", "context", "input_text", "scene", "events"):
        text = _as_optional_str(data.get(key))
        if text:
            return format_reasoning_snippet(text, max_chars=180, redact=True)
    return ""


def _payload_output_preview(msg: Dict[str, Any]) -> str:
    payload = _as_dict(msg.get("payload"))
    data = _as_dict(payload.get("data"))
    if not data:
        data = payload
    for key in ("response_text", "raw_text", "output_text", "rationale_short", "summary", "reasoning"):
        text = _as_optional_str(data.get(key))
        if text:
            return format_reasoning_snippet(text, max_chars=180, redact=True)
    return ""


def _reasoning_row(
    *,
    row_id: int,
    event_index: int,
    msg: Dict[str, Any],
    reasoning: ReasoningEvent,
    trace_meta: Dict[str, Any],
    latest_perception: Optional[Dict[str, Any]],
    latest_video: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    trace_id = _as_optional_str(trace_meta.get("trace_id"))
    if not trace_id:
        if reasoning.req_id is not None:
            trace_id = f"req:{reasoning.req_id}"
        else:
            trace_id = f"event:{event_index}"
    return {
        "row_id": int(row_id),
        "event_index": int(event_index),
        "trace_id": trace_id,
        "req_id": reasoning.req_id if reasoning.req_id is not None else _as_optional_int(trace_meta.get("req_id")),
        "component": _as_optional_str(reasoning.component) or "unknown",
        "source": reasoning.source,
        "ts_wall_s": reasoning.ts_wall_s,
        "phase": reasoning.phase,
        "status": reasoning.status,
        "severity": reasoning.severity,
        "latency_ms": reasoning.latency_ms,
        "primitive": reasoning.primitive,
        "confidence": reasoning.confidence,
        "target_zone": reasoning.target_zone,
        "model": reasoning.model,
        "provider": reasoning.provider,
        "delta_score": reasoning.delta_score,
        "mode": reasoning.mode,
        "active_skill": reasoning.active_skill,
        "guard_accepted": reasoning.guard_accepted,
        "guard_fallback": reasoning.guard_fallback,
        "guard_reason": reasoning.guard_reason,
        "guard_skill": reasoning.guard_skill,
        "guard_primitive": reasoning.guard_primitive,
        "planner_enabled": reasoning.planner_enabled,
        "planner_inflight": reasoning.planner_inflight,
        "planner_pending": reasoning.planner_pending,
        "planner_last_parse_stage": reasoning.planner_last_parse_stage,
        "planner_last_error": reasoning.planner_error,
        "planner_next_allowed_in_s": reasoning.planner_next_allowed_in_s,
        "mode_transition_from": reasoning.mode_transition_from,
        "mode_transition_to": reasoning.mode_transition_to,
        "mode_transition_reason": reasoning.mode_transition_reason,
        "mode_transitioned": reasoning.mode_transitioned,
        "snippet": format_reasoning_snippet(reasoning.snippet, max_chars=220, redact=True),
        "input_preview": _payload_input_preview(msg),
        "output_preview": _payload_output_preview(msg),
        "perception_ts_wall_s": (latest_perception or {}).get("ts_wall_s"),
        "perception_person_conf": (latest_perception or {}).get("person_conf"),
        "perception_zone_hint": (latest_perception or {}).get("zone_hint"),
        "perception_frame_id": (latest_perception or {}).get("frame_id"),
        "perception_fps": (latest_perception or {}).get("fps"),
        "perception_latency_ms": (latest_perception or {}).get("latency_ms"),
        "video_frame_event_index": (latest_video or {}).get("event_index"),
        "video_frame_id": (latest_video or {}).get("frame_id"),
        "video_frame_ref": (latest_video or {}).get("frame_ref"),
        "video_frame_width": (latest_video or {}).get("width"),
        "video_frame_height": (latest_video or {}).get("height"),
    }


def build_reasoning_trace_rows(
    events: Sequence[Tuple[int, Dict[str, Any]]],
    *,
    traces: Sequence[TraceRecord],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    lookup = _trace_lookup(traces)
    latest_perception: Optional[Dict[str, Any]] = None
    latest_video: Optional[Dict[str, Any]] = None
    row_id = 0

    for event_index, msg in events:
        source = _as_optional_str(msg.get("source")) or "unknown"
        if source == "perception_log":
            snap = _extract_perception_snapshot(msg)
            if snap is not None:
                latest_perception = snap
        elif source == "video_frame":
            snap = _extract_video_snapshot(msg, int(event_index))
            if snap is not None:
                latest_video = snap

        reasoning = normalize_reasoning_message(msg)
        if reasoning is None:
            continue

        row_id += 1
        rows.append(
            _reasoning_row(
                row_id=row_id,
                event_index=int(event_index),
                msg=msg,
                reasoning=reasoning,
                trace_meta=lookup.get(int(event_index), {}),
                latest_perception=latest_perception,
                latest_video=latest_video,
            )
        )
    return rows


def build_reasoning_trace_index(
    session_dir: str,
    *,
    traces: Optional[Sequence[TraceRecord]] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    root = str(session_dir)
    events_path = os.path.join(root, "events.jsonl")
    if not os.path.exists(events_path):
        return {"version": 1, "generated_at_wall_s": time.time(), "rows": [], "row_count": 0}
    events = list(_iter_events(events_path))
    if traces is None:
        trace_path = resolve_trace_index_path(root, manifest if isinstance(manifest, dict) else None)
        if os.path.exists(trace_path):
            try:
                traces = load_trace_index(trace_path)
            except Exception:
                traces = []
        else:
            traces = []
    rows = build_reasoning_trace_rows(events, traces=traces or [])
    return {
        "version": 1,
        "generated_at_wall_s": time.time(),
        "row_count": len(rows),
        "rows": rows,
    }


def write_reasoning_trace_index(
    session_dir: str,
    index_obj: Mapping[str, Any],
    *,
    filename: str = REASONING_TRACE_INDEX_PATH,
) -> str:
    path = os.path.join(str(session_dir), str(filename))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(index_obj), fh, separators=(",", ":"), ensure_ascii=True)
    return path


def load_reasoning_trace_index(path_or_dir: str) -> List[Dict[str, Any]]:
    path = str(path_or_dir)
    if os.path.isdir(path):
        path = os.path.join(path, REASONING_TRACE_INDEX_PATH)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception:
        return []
    rows = obj.get("rows") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]
