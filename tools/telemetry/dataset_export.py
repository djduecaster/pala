from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .labels import load_labels_jsonl
from .schema_v3 import DATASET_ROWS_PATH, WEAK_LABELS_PATH
from .trace_graph import TraceRecord, load_trace_index, resolve_trace_index_path


def _load_reasoning_index(session_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(session_dir, "reasoning_index.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    events = obj.get("events")
    if not isinstance(events, list):
        return []
    return [item for item in events if isinstance(item, dict)]


def _load_manifest(session_dir: str) -> Optional[Dict[str, Any]]:
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


def _label_map(labels: Sequence[Mapping[str, Any]], *, min_confidence: float) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    threshold = max(0.0, min(1.0, float(min_confidence)))
    for row in labels:
        try:
            idx = int(row.get("event_index"))
            conf = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if conf < threshold:
            continue
        out.setdefault(idx, []).append(dict(row))
    return out


def _trace_lookup(traces: Sequence[TraceRecord]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for trace in traces:
        for ref in trace.event_refs:
            out[int(ref.event_index)] = {
                "trace_id": trace.trace_id,
                "trace_status": trace.status,
                "trace_severity": trace.severity,
                "trace_summary": trace.summary,
            }
    return out


def export_dataset_rows(
    session_dir: str,
    *,
    output_path: str = "",
    include_unlabeled: bool = False,
    min_label_confidence: float = 0.6,
) -> Dict[str, Any]:
    root = str(session_dir)
    reasoning_rows = _load_reasoning_index(root)
    labels_path = os.path.join(root, WEAK_LABELS_PATH)
    labels = load_labels_jsonl(labels_path)
    labels_by_event = _label_map(labels, min_confidence=min_label_confidence)

    manifest = _load_manifest(root)
    trace_index_path = resolve_trace_index_path(root, manifest)
    traces = load_trace_index(trace_index_path) if os.path.exists(trace_index_path) else []
    trace_by_event = _trace_lookup(traces)

    rows: List[Dict[str, Any]] = []
    for item in reasoning_rows:
        try:
            event_index = int(item.get("event_index"))
        except (TypeError, ValueError):
            continue
        row_labels = labels_by_event.get(event_index, [])
        if (not include_unlabeled) and (not row_labels):
            continue
        out: Dict[str, Any] = {
            "event_index": event_index,
            "req_id": item.get("req_id"),
            "source": item.get("source"),
            "phase": item.get("phase"),
            "status": item.get("status"),
            "severity": item.get("severity"),
            "latency_ms": item.get("latency_ms"),
            "snippet": item.get("snippet"),
            "labels": row_labels,
        }
        trace_data = trace_by_event.get(event_index)
        if trace_data is not None:
            out.update(trace_data)
        rows.append(out)

    target = output_path.strip() or os.path.join(root, DATASET_ROWS_PATH)
    with open(target, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True))
            fh.write("\n")

    return {"output_path": target, "row_count": len(rows)}
