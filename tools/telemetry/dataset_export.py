from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .annotations import annotation_key, load_annotations
from .labels import load_labels_jsonl
from .schema_v3 import DATASET_MANIFEST_PATH, DATASET_ROWS_PATH, WEAK_LABELS_PATH
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


def _profile_config(profile: str) -> Tuple[float, bool]:
    name = str(profile or "fast").strip().lower()
    if name == "strict":
        return 0.75, False
    if name == "hard_cases":
        return 0.55, False
    return 0.5, False


def _row_is_hard_case(row: Mapping[str, Any]) -> bool:
    severity = str(row.get("severity") or "").lower()
    status = str(row.get("status") or "").lower()
    trace_sev = str(row.get("trace_severity") or "").lower()
    latency = row.get("latency_ms")
    if severity in {"error", "warning"}:
        return True
    if trace_sev == "error":
        return True
    if "parse_fail" in status or "timeout" in status or "stale" in status:
        return True
    try:
        return float(latency) >= 2000.0
    except (TypeError, ValueError):
        return False


def _label_has_failure_signal(labels: Sequence[Mapping[str, Any]]) -> bool:
    for label in labels:
        label_name = str(label.get("label") or "").lower()
        if label_name in {"planner_parse_fail", "planner_timeout", "reasoning_error", "trace_failure"}:
            return True
    return False


def _inclusion_reasons(row: Mapping[str, Any], labels: Sequence[Mapping[str, Any]]) -> List[str]:
    reasons: List[str] = []
    if _row_is_hard_case(row):
        reasons.append("hard_case")
    if row.get("annotations"):
        reasons.append("annotation")
    if _label_has_failure_signal(labels):
        reasons.append("weak_label")
    if not reasons:
        reasons.append("baseline")
    return reasons


def _safe_ratio(numer: int, denom: int) -> float:
    base = max(0, int(denom))
    if base <= 0:
        return 0.0
    return round(max(0.0, float(numer)) / float(base), 4)


def _profile_allows_row(profile: str, row: Mapping[str, Any], labels: Sequence[Mapping[str, Any]]) -> bool:
    name = str(profile or "fast").strip().lower()
    if name == "strict":
        if row.get("annotations"):
            return True
        return _row_is_hard_case(row)
    if name == "hard_cases":
        reasons = _inclusion_reasons(row, labels)
        return any(reason in {"hard_case", "annotation", "weak_label"} for reason in reasons)
    return True


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _annotation_buckets(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]], Dict[str, int]]:
    by_event: Dict[int, List[Dict[str, Any]]] = {}
    by_trace: Dict[str, List[Dict[str, Any]]] = {}
    by_req: Dict[int, List[Dict[str, Any]]] = {}
    tag_counts: Dict[str, int] = {}
    for raw in rows:
        row = dict(raw)
        tag = str(row.get("tag") or "").strip().lower()
        if tag:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        event_index = _as_int(row.get("event_index"))
        if event_index is not None:
            by_event.setdefault(event_index, []).append(row)
        trace_id = str(row.get("trace_id") or "").strip()
        if trace_id:
            by_trace.setdefault(trace_id, []).append(row)
        req_id = _as_int(row.get("req_id"))
        if req_id is not None:
            by_req.setdefault(req_id, []).append(row)
    return by_event, by_trace, by_req, tag_counts


def export_dataset_rows(
    session_dir: str,
    *,
    output_path: str = "",
    include_unlabeled: bool = False,
    min_label_confidence: float = 0.6,
    profile: str = "fast",
    write_manifest: bool = True,
) -> Dict[str, Any]:
    root = str(session_dir)
    reasoning_rows = _load_reasoning_index(root)
    labels_path = os.path.join(root, WEAK_LABELS_PATH)
    labels = load_labels_jsonl(labels_path)
    prof_min_conf, prof_include_unlabeled = _profile_config(profile)
    threshold = max(float(min_label_confidence), float(prof_min_conf))
    labels_by_event = _label_map(labels, min_confidence=threshold)
    annotations = load_annotations(root, limit=0)
    ann_by_event, ann_by_trace, ann_by_req, session_annotation_tag_counts = _annotation_buckets(annotations)

    manifest = _load_manifest(root)
    trace_index_path = resolve_trace_index_path(root, manifest)
    traces = load_trace_index(trace_index_path) if os.path.exists(trace_index_path) else []
    trace_by_event = _trace_lookup(traces)

    rows: List[Dict[str, Any]] = []
    effective_include_unlabeled = bool(include_unlabeled) or bool(prof_include_unlabeled)
    label_counts: Dict[str, int] = {}
    component_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    phase_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {}
    annotation_tag_counts: Dict[str, int] = {}
    inclusion_reason_counts: Dict[str, int] = {}
    annotation_count = 0
    annotated_row_count = 0
    labeled_row_count = 0
    hard_case_row_count = 0
    for item in reasoning_rows:
        try:
            event_index = int(item.get("event_index"))
        except (TypeError, ValueError):
            continue
        row_labels = labels_by_event.get(event_index, [])
        row_annotations: List[Dict[str, Any]] = []
        seen_annotation_keys: set[str] = set()
        for ann in ann_by_event.get(event_index, []):
            key = annotation_key(ann)
            if key in seen_annotation_keys:
                continue
            seen_annotation_keys.add(key)
            row_annotations.append(dict(ann))
        trace_id = str(trace_by_event.get(event_index, {}).get("trace_id") or "")
        if trace_id:
            for ann in ann_by_trace.get(trace_id, []):
                key = annotation_key(ann)
                if key in seen_annotation_keys:
                    continue
                seen_annotation_keys.add(key)
                row_annotations.append(dict(ann))
        req_id = _as_int(item.get("req_id"))
        if req_id is not None:
            for ann in ann_by_req.get(req_id, []):
                key = annotation_key(ann)
                if key in seen_annotation_keys:
                    continue
                seen_annotation_keys.add(key)
                row_annotations.append(dict(ann))
        if (not effective_include_unlabeled) and (not row_labels) and (not row_annotations):
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
            "annotations": row_annotations,
        }
        trace_data = trace_by_event.get(event_index)
        if trace_data is not None:
            out.update(trace_data)
        reasons = _inclusion_reasons(out, row_labels)
        out["inclusion_reasons"] = reasons
        if not _profile_allows_row(profile, out, row_labels):
            continue
        rows.append(out)
        for reason in reasons:
            inclusion_reason_counts[reason] = inclusion_reason_counts.get(reason, 0) + 1
        if _row_is_hard_case(out):
            hard_case_row_count += 1
        if row_labels:
            labeled_row_count += 1
        if row_annotations:
            annotated_row_count += 1
            annotation_count += len(row_annotations)
            for ann in row_annotations:
                tag = str(ann.get("tag") or "").strip().lower()
                if not tag:
                    continue
                annotation_tag_counts[tag] = annotation_tag_counts.get(tag, 0) + 1
        for label in row_labels:
            name = str(label.get("label") or "").strip().lower()
            if not name:
                continue
            label_counts[name] = label_counts.get(name, 0) + 1
        comp = str(item.get("component") or "").strip().lower()
        if comp:
            component_counts[comp] = component_counts.get(comp, 0) + 1
        src = str(item.get("source") or "").strip().lower()
        if src:
            source_counts[src] = source_counts.get(src, 0) + 1
        phase = str(item.get("phase") or "").strip().lower()
        if phase:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        status = str(item.get("status") or "").strip().lower()
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        sev = str(item.get("severity") or "").strip().lower()
        if sev:
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
    rows.sort(
        key=lambda row: (
            int(row.get("event_index", 0) or 0),
            str(row.get("source") or ""),
            str(row.get("phase") or ""),
        )
    )

    target = output_path.strip() or os.path.join(root, DATASET_ROWS_PATH)
    with open(target, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True))
            fh.write("\n")

    manifest_path = os.path.join(root, DATASET_MANIFEST_PATH)
    if bool(write_manifest):
        out_manifest = {
            "version": 1,
            "profile": str(profile or "fast"),
            "row_count": len(rows),
            "label_counts": label_counts,
            "component_counts": component_counts,
            "source_counts": source_counts,
            "phase_counts": phase_counts,
            "status_counts": status_counts,
            "severity_counts": severity_counts,
            "annotation_count": annotation_count,
            "annotated_row_count": annotated_row_count,
            "labeled_row_count": labeled_row_count,
            "hard_case_row_count": hard_case_row_count,
            "annotation_coverage_ratio": _safe_ratio(annotated_row_count, len(rows)),
            "label_coverage_ratio": _safe_ratio(labeled_row_count, len(rows)),
            "hard_case_ratio": _safe_ratio(hard_case_row_count, len(rows)),
            "inclusion_reason_counts": inclusion_reason_counts,
            "annotation_tag_counts": annotation_tag_counts,
            "session_annotation_tag_counts": session_annotation_tag_counts,
            "min_label_confidence": threshold,
            "include_unlabeled": effective_include_unlabeled,
            "output_path": target,
        }
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(out_manifest, fh, separators=(",", ":"), ensure_ascii=True)

    return {
        "output_path": target,
        "row_count": len(rows),
        "profile": str(profile or "fast"),
        "manifest_path": manifest_path if bool(write_manifest) else "",
    }
