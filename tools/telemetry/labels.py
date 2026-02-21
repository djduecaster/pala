from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .trace_graph import TraceRecord


def _coerce_event_index(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _push_label(
    labels: Dict[Tuple[int, str], Dict[str, Any]],
    *,
    event_index: Optional[int],
    label: str,
    confidence: float,
    reason: str,
    req_id: Optional[int] = None,
    source: Optional[str] = None,
) -> None:
    idx = _coerce_event_index(event_index)
    if idx is None:
        return
    key = (idx, label)
    candidate = {
        "event_index": idx,
        "label": str(label),
        "confidence": round(max(0.0, min(1.0, float(confidence))), 3),
        "reason": str(reason),
    }
    if req_id is not None:
        candidate["req_id"] = int(req_id)
    if source:
        candidate["source"] = str(source)
    current = labels.get(key)
    if current is None or float(candidate["confidence"]) > float(current.get("confidence", 0.0)):
        labels[key] = candidate


def derive_weak_labels(
    *,
    reasoning_index: Sequence[Mapping[str, Any]],
    traces: Sequence[TraceRecord],
    low_confidence_threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    low_conf = float(low_confidence_threshold)

    for row in reasoning_index:
        event_index = _coerce_event_index(row.get("event_index"))
        status = str(row.get("status") or "").lower()
        phase = str(row.get("phase") or "").lower()
        severity = str(row.get("severity") or "").lower()
        req_id = _coerce_event_index(row.get("req_id"))
        source = str(row.get("source") or "")

        if "parse_fail" in status or "parse_fail" in phase:
            _push_label(
                out,
                event_index=event_index,
                label="planner_parse_fail",
                confidence=0.99,
                reason=f"status={status or phase}",
                req_id=req_id,
                source=source,
            )
        if "timeout" in status or "timeout" in phase:
            _push_label(
                out,
                event_index=event_index,
                label="planner_timeout",
                confidence=0.96,
                reason=f"status={status or phase}",
                req_id=req_id,
                source=source,
            )
        if "no_content" in status:
            _push_label(
                out,
                event_index=event_index,
                label="planner_no_content",
                confidence=0.93,
                reason=f"status={status}",
                req_id=req_id,
                source=source,
            )
        if severity == "error":
            _push_label(
                out,
                event_index=event_index,
                label="reasoning_error",
                confidence=0.82,
                reason=f"severity={severity}",
                req_id=req_id,
                source=source,
            )

        confidence_value = row.get("confidence")
        if confidence_value is not None:
            try:
                conf = float(confidence_value)
            except (TypeError, ValueError):
                conf = None
            if conf is not None and conf < low_conf:
                _push_label(
                    out,
                    event_index=event_index,
                    label="low_confidence_action",
                    confidence=0.7,
                    reason=f"confidence={conf:.3f}<threshold={low_conf:.3f}",
                    req_id=req_id,
                    source=source,
                )

    for trace in traces:
        if not trace.event_refs:
            continue
        tail = trace.event_refs[-1]
        if trace.severity == "error":
            _push_label(
                out,
                event_index=tail.event_index,
                label="trace_failure",
                confidence=0.84,
                reason=f"trace_status={trace.status}",
                req_id=trace.req_id,
                source=tail.source,
            )
        if trace.status in {"timeout", "parse_fail", "no_content"}:
            _push_label(
                out,
                event_index=tail.event_index,
                label=f"trace_{trace.status}",
                confidence=0.9,
                reason=f"trace_status={trace.status}",
                req_id=trace.req_id,
                source=tail.source,
            )

    rows = list(out.values())
    rows.sort(key=lambda row: (int(row["event_index"]), str(row["label"])))
    return rows


def write_labels_jsonl(path: str, labels: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in labels:
            fh.write(json.dumps(dict(row), separators=(",", ":"), ensure_ascii=True))
            fh.write("\n")
            count += 1
    return count


def load_labels_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path:
        return rows
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return rows
    with fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows
