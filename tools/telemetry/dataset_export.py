from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .annotations import annotation_key, load_annotations
from .labels import load_labels_jsonl
from .schema_v3 import DATASET_MANIFEST_PATH, DATASET_ROWS_PATH, TELEMETRY_SCHEMA_VERSION_V3, WEAK_LABELS_PATH
from .storage_sqlite import resolve_session_db_path
from .mode_health_fsm import ModeHealthFSM, ingest_mode_event


_REQUIRED_DB_TABLES = {"cases", "case_events", "case_labels", "case_reviews", "reasoning_traces"}
_REQUIRED_ROW_KEYS = (
    "schema_version",
    "row_version",
    "session_dir",
    "case_id",
    "trace_id",
    "source",
    "case_status",
    "case_severity",
    "review_decision",
    "inclusion_reasons",
    "provenance_refs",
    "reasoning_context",
    "env_context",
    "planner_context",
    "arbiter_context",
    "mode_context",
    "mode_fsm_state",
    "perception_context",
    "video_context",
)


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(text)


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


def _profile_config(profile: str) -> Tuple[float, bool]:
    name = str(profile or "fast").strip().lower()
    if name == "strict":
        return 0.75, False
    if name == "hard_cases":
        return 0.55, False
    return 0.50, True


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


def _db_table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if row and isinstance(row[0], str)
    }


def _validate_session_contract_for_export(session_dir: str, db_path: str) -> List[str]:
    errors: List[str] = []
    manifest = _load_manifest(session_dir)
    if isinstance(manifest, dict):
        try:
            schema_version = int(manifest.get("schema_version", 0) or 0)
        except (TypeError, ValueError):
            schema_version = 0
        if schema_version > 0 and schema_version < int(TELEMETRY_SCHEMA_VERSION_V3):
            errors.append(
                f"schema_version={schema_version} (<{TELEMETRY_SCHEMA_VERSION_V3}); "
                "run: uv run python -m tools.telemetry.migrate_session <session_dir>"
            )
    try:
        with sqlite3.connect(db_path) as conn:
            tables = _db_table_names(conn)
            missing = sorted(_REQUIRED_DB_TABLES - tables)
            if missing:
                errors.append(f"missing_required_tables={','.join(missing)}")
            if "cases" in tables:
                bad_sources = [
                    str(row[0])
                    for row in conn.execute("SELECT DISTINCT source FROM cases WHERE COALESCE(source, '') != 'sqlite.cases.v4'")
                    if row and row[0] is not None
                ]
                if bad_sources:
                    errors.append(f"case_source_mismatch={','.join(sorted(set(bad_sources)))}")
    except Exception as exc:
        errors.append(f"session_db_open_failed={exc!r}")
    return errors


def _load_case_rows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
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
            r.note,
            r.reviewer,
            r.reviewed_at_wall_s
        FROM cases c
        LEFT JOIN case_reviews r ON r.case_id = c.case_id
        ORDER BY c.hardness DESC, c.start_ts_wall_s DESC, c.case_id ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _load_case_labels_map(conn: sqlite3.Connection) -> Dict[str, List[Dict[str, Any]]]:
    conn.row_factory = sqlite3.Row
    out: Dict[str, List[Dict[str, Any]]] = {}
    rows = conn.execute(
        """
        SELECT case_id, label, score, reason
        FROM case_labels
        ORDER BY case_id ASC, score DESC, label ASC
        """
    ).fetchall()
    for row in rows:
        item = dict(row)
        case_id = str(item.get("case_id") or "")
        if not case_id:
            continue
        out.setdefault(case_id, []).append(item)
    return out


def _load_case_event_context_map(conn: sqlite3.Connection) -> Dict[str, List[Dict[str, Any]]]:
    conn.row_factory = sqlite3.Row
    out: Dict[str, List[Dict[str, Any]]] = {}
    rows = conn.execute(
        """
        SELECT
            ce.case_id AS case_id,
            ce.row_id AS row_id,
            ce.event_index AS event_index,
            ce.source AS event_source,
            ce.component AS event_component,
            ce.phase AS event_phase,
            ce.status AS event_status,
            ce.severity AS event_severity,
            ce.ts_wall_s AS event_ts_wall_s,
            ce.latency_ms AS event_latency_ms,
            ce.snippet AS event_snippet,
            rt.component AS rt_component,
            rt.source AS rt_source,
            rt.phase AS rt_phase,
            rt.status AS rt_status,
            rt.severity AS rt_severity,
            rt.ts_wall_s AS rt_ts_wall_s,
            rt.latency_ms AS rt_latency_ms,
            rt.snippet AS rt_snippet,
            rt.input_preview AS rt_input_preview,
            rt.output_preview AS rt_output_preview,
            rt.primitive AS rt_primitive,
            rt.confidence AS rt_confidence,
            rt.target_zone AS rt_target_zone,
            rt.model AS rt_model,
            rt.provider AS rt_provider,
            rt.delta_score AS rt_delta_score,
            rt.mode AS rt_mode,
            rt.active_skill AS rt_active_skill,
            rt.guard_accepted AS rt_guard_accepted,
            rt.guard_fallback AS rt_guard_fallback,
            rt.guard_reason AS rt_guard_reason,
            rt.guard_skill AS rt_guard_skill,
            rt.guard_primitive AS rt_guard_primitive,
            rt.planner_enabled AS rt_planner_enabled,
            rt.planner_inflight AS rt_planner_inflight,
            rt.planner_pending AS rt_planner_pending,
            rt.planner_last_parse_stage AS rt_planner_last_parse_stage,
            rt.planner_last_error AS rt_planner_last_error,
            rt.planner_next_allowed_in_s AS rt_planner_next_allowed_in_s,
            rt.mode_transition_from AS rt_mode_transition_from,
            rt.mode_transition_to AS rt_mode_transition_to,
            rt.mode_transition_reason AS rt_mode_transition_reason,
            rt.mode_transitioned AS rt_mode_transitioned,
            rt.perception_person_conf AS rt_perception_person_conf,
            rt.perception_zone_hint AS rt_perception_zone_hint,
            rt.perception_frame_id AS rt_perception_frame_id,
            rt.video_frame_id AS rt_video_frame_id,
            rt.video_frame_ref AS rt_video_frame_ref,
            rt.video_frame_width AS rt_video_frame_width,
            rt.video_frame_height AS rt_video_frame_height
        FROM case_events ce
        LEFT JOIN reasoning_traces rt ON rt.row_id = ce.row_id
        ORDER BY ce.case_id ASC, ce.row_id ASC
        """
    ).fetchall()
    for row in rows:
        item = dict(row)
        case_id = str(item.get("case_id") or "")
        if not case_id:
            continue
        out.setdefault(case_id, []).append(item)
    return out


def _is_failure_label(name: str) -> bool:
    label = str(name or "").strip().lower()
    return label in {
        "planner_parse_fail",
        "planner_timeout",
        "reasoning_error",
        "trace_failure",
        "hard_case",
        "planner_transport_error",
        "guard_fallback",
        "mode_transition_churn",
    }


def _case_is_hard_case(case_row: Mapping[str, Any], labels: Sequence[Mapping[str, Any]], weak_labels: Sequence[Mapping[str, Any]]) -> bool:
    status = str(case_row.get("status") or "").lower()
    severity = str(case_row.get("severity") or "").lower()
    hardness = _as_float(case_row.get("hardness")) or 0.0
    max_latency = _as_float(case_row.get("max_latency_ms")) or 0.0
    if severity in {"error", "warning"}:
        return True
    if any(token in status for token in ("parse_fail", "timeout", "stale", "fail", "error")):
        return True
    if hardness >= 0.70:
        return True
    if max_latency >= 2000.0:
        return True
    if any(_is_failure_label(str(item.get("label") or "")) for item in labels):
        return True
    if any(_is_failure_label(str(item.get("label") or "")) for item in weak_labels):
        return True
    return False


def _has_case_label(rows: Sequence[Mapping[str, Any]], name: str) -> bool:
    target = str(name or "").strip().lower()
    if not target:
        return False
    for item in rows:
        if str(item.get("label") or "").strip().lower() == target:
            return True
    return False


def _inclusion_reasons(
    case_row: Mapping[str, Any],
    *,
    case_labels: Sequence[Mapping[str, Any]],
    weak_labels: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
) -> List[str]:
    reasons: List[str] = []
    if _case_is_hard_case(case_row, case_labels, weak_labels):
        reasons.append("hard_case")
    if annotations:
        reasons.append("annotation")
    if any(_is_failure_label(str(item.get("label") or "")) for item in weak_labels):
        reasons.append("weak_label")
    if _has_case_label(case_labels, "planner_transport_error"):
        reasons.append("planner_failure")
    if _has_case_label(case_labels, "guard_fallback"):
        reasons.append("guard_fallback")
    if _has_case_label(case_labels, "mode_transition_churn"):
        reasons.append("transition_instability")
    if not reasons:
        reasons.append("baseline")
    return reasons


def _profile_allows_case(*, profile: str, reviewed: bool, reasons: Sequence[str], include_unlabeled: bool) -> bool:
    name = str(profile or "fast").strip().lower()
    reason_set = {str(item) for item in reasons}
    if name == "strict":
        return bool(reviewed)
    if name == "hard_cases":
        if include_unlabeled:
            return True
        return bool(reason_set & {"hard_case", "annotation", "weak_label", "planner_failure", "guard_fallback", "transition_instability"})
    return True


def _dedupe_annotations(
    *,
    event_indices: Sequence[int],
    trace_id: str,
    req_id: Optional[int],
    ann_by_event: Mapping[int, Sequence[Mapping[str, Any]]],
    ann_by_trace: Mapping[str, Sequence[Mapping[str, Any]]],
    ann_by_req: Mapping[int, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for event_index in event_indices:
        for ann in ann_by_event.get(int(event_index), []):
            key = annotation_key(ann)
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(ann))
    if trace_id:
        for ann in ann_by_trace.get(str(trace_id), []):
            key = annotation_key(ann)
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(ann))
    if req_id is not None:
        for ann in ann_by_req.get(int(req_id), []):
            key = annotation_key(ann)
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(ann))
    return out


def _event_context_item(raw: Mapping[str, Any]) -> Dict[str, Any]:
    row_id = _as_int(raw.get("row_id")) or 0
    event_index = _as_int(raw.get("event_index")) or 0
    component = _as_text(raw.get("rt_component") or raw.get("event_component") or "unknown")
    source = _as_text(raw.get("rt_source") or raw.get("event_source"))
    phase = _as_text(raw.get("rt_phase") or raw.get("event_phase"))
    status = _as_text(raw.get("rt_status") or raw.get("event_status"))
    severity = _as_text(raw.get("rt_severity") or raw.get("event_severity") or "info")
    ts_wall_s = _as_float(raw.get("rt_ts_wall_s"))
    if ts_wall_s is None:
        ts_wall_s = _as_float(raw.get("event_ts_wall_s"))
    latency_ms = _as_float(raw.get("rt_latency_ms"))
    if latency_ms is None:
        latency_ms = _as_float(raw.get("event_latency_ms"))
    snippet = _as_text(raw.get("rt_snippet") or raw.get("event_snippet"))
    return {
        "row_id": int(row_id),
        "event_index": int(event_index),
        "component": component,
        "source": source,
        "phase": phase,
        "status": status,
        "severity": severity,
        "ts_wall_s": ts_wall_s,
        "latency_ms": latency_ms,
        "snippet": snippet,
        "input_preview": _as_text(raw.get("rt_input_preview")),
        "output_preview": _as_text(raw.get("rt_output_preview")),
        "primitive": _as_text(raw.get("rt_primitive")),
        "confidence": _as_float(raw.get("rt_confidence")),
        "target_zone": _as_text(raw.get("rt_target_zone")),
        "model": _as_text(raw.get("rt_model")),
        "provider": _as_text(raw.get("rt_provider")),
        "delta_score": _as_float(raw.get("rt_delta_score")),
        "mode": _as_text(raw.get("rt_mode")),
        "active_skill": _as_text(raw.get("rt_active_skill")),
        "guard_accepted": _as_int(raw.get("rt_guard_accepted")),
        "guard_fallback": _as_int(raw.get("rt_guard_fallback")),
        "guard_reason": _as_text(raw.get("rt_guard_reason")),
        "guard_skill": _as_text(raw.get("rt_guard_skill")),
        "guard_primitive": _as_text(raw.get("rt_guard_primitive")),
        "planner_enabled": _as_int(raw.get("rt_planner_enabled")),
        "planner_inflight": _as_int(raw.get("rt_planner_inflight")),
        "planner_pending": _as_int(raw.get("rt_planner_pending")),
        "planner_last_parse_stage": _as_text(raw.get("rt_planner_last_parse_stage")),
        "planner_last_error": _as_text(raw.get("rt_planner_last_error")),
        "planner_next_allowed_in_s": _as_float(raw.get("rt_planner_next_allowed_in_s")),
        "mode_transition_from": _as_text(raw.get("rt_mode_transition_from")),
        "mode_transition_to": _as_text(raw.get("rt_mode_transition_to")),
        "mode_transition_reason": _as_text(raw.get("rt_mode_transition_reason")),
        "mode_transitioned": _as_int(raw.get("rt_mode_transitioned")),
        "perception_person_conf": _as_float(raw.get("rt_perception_person_conf")),
        "perception_zone_hint": _as_text(raw.get("rt_perception_zone_hint")),
        "perception_frame_id": _as_int(raw.get("rt_perception_frame_id")),
        "video_frame_id": _as_int(raw.get("rt_video_frame_id")),
        "video_frame_ref": _as_text(raw.get("rt_video_frame_ref")),
        "video_frame_width": _as_int(raw.get("rt_video_frame_width")),
        "video_frame_height": _as_int(raw.get("rt_video_frame_height")),
    }


def _compress_context_event(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "row_id": item.get("row_id"),
        "event_index": item.get("event_index"),
        "component": item.get("component"),
        "source": item.get("source"),
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
        "mode": item.get("mode"),
        "active_skill": item.get("active_skill"),
        "guard_accepted": item.get("guard_accepted"),
        "guard_fallback": item.get("guard_fallback"),
        "guard_reason": item.get("guard_reason"),
        "guard_skill": item.get("guard_skill"),
        "guard_primitive": item.get("guard_primitive"),
        "planner_enabled": item.get("planner_enabled"),
        "planner_inflight": item.get("planner_inflight"),
        "planner_pending": item.get("planner_pending"),
        "planner_last_parse_stage": item.get("planner_last_parse_stage"),
        "planner_last_error": item.get("planner_last_error"),
        "planner_next_allowed_in_s": item.get("planner_next_allowed_in_s"),
        "mode_transition_from": item.get("mode_transition_from"),
        "mode_transition_to": item.get("mode_transition_to"),
        "mode_transition_reason": item.get("mode_transition_reason"),
        "mode_transitioned": item.get("mode_transitioned"),
        "model": item.get("model"),
        "provider": item.get("provider"),
    }


def _build_contexts(case_events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    reasoning_context = [_compress_context_event(item) for item in case_events]
    latest_by_component: Dict[str, Dict[str, Any]] = {}
    perception_context: Dict[str, Any] = {}
    video_frames: List[Dict[str, Any]] = []
    seen_video: set[Tuple[Any, ...]] = set()
    for item in case_events:
        component = str(item.get("component") or "").strip().lower()
        if component:
            latest_by_component[component] = _compress_context_event(item)
        person_conf = item.get("perception_person_conf")
        zone_hint = item.get("perception_zone_hint")
        frame_id = item.get("perception_frame_id")
        if person_conf is not None or zone_hint or frame_id is not None:
            perception_context = {
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
                video_frames.append(
                    {
                        "frame_ref": video_ref,
                        "frame_id": video_id,
                        "width": item.get("video_frame_width"),
                        "height": item.get("video_frame_height"),
                    }
                )

    mode_context: Dict[str, Any] = {}
    for item in reversed(case_events):
        mode = str(item.get("mode") or "").strip()
        skill = str(item.get("active_skill") or "").strip()
        trans_from = str(item.get("mode_transition_from") or "").strip()
        trans_to = str(item.get("mode_transition_to") or "").strip()
        if mode or skill or trans_from or trans_to:
            mode_context = {
                "mode": mode,
                "active_skill": skill,
                "transition_from": trans_from,
                "transition_to": trans_to,
                "transition_reason": item.get("mode_transition_reason"),
                "transitioned": item.get("mode_transitioned"),
            }
            break

    return {
        "reasoning_context": reasoning_context,
        "env_context": latest_by_component.get("env_processor", {}),
        "planner_context": latest_by_component.get("planner_v4", latest_by_component.get("planner", {})),
        "arbiter_context": latest_by_component.get("arbiter", {}),
        "mode_context": mode_context,
        "perception_context": perception_context,
        "video_context": {"frames": video_frames},
    }



def _mode_health_for_case(case_events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    fsm = ModeHealthFSM()
    saw = False
    for item in case_events:
        if not isinstance(item, Mapping):
            continue
        mode = str(item.get("mode") or "").strip()
        transitioned = _as_boolish(item.get("mode_transitioned"))
        reason = str(item.get("mode_transition_reason") or "").strip()
        planner_error = str(item.get("planner_last_error") or "").strip()
        guard_fallback = _as_boolish(item.get("guard_fallback"))
        if (not mode) and (not transitioned) and (not reason) and (not planner_error) and (not guard_fallback):
            continue
        ingest_mode_event(fsm, item)
        saw = True
    snapshot = fsm.last_snapshot
    state = snapshot.state if saw else "unknown"
    return {
        "state": state,
        "transition_reason_counts": fsm.transition_reason_counts(),
    }


def _collect_case_weak_labels(
    *,
    event_indices: Sequence[int],
    labels_by_event: Mapping[int, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()
    for event_index in event_indices:
        for item in labels_by_event.get(int(event_index), []):
            label = str(item.get("label") or "")
            confidence = _as_float(item.get("confidence"))
            reason = str(item.get("reason") or "")
            key = (int(event_index), label, confidence, reason)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "event_index": int(event_index),
                    "label": label,
                    "confidence": confidence,
                    "reason": reason,
                }
            )
    return out


def _validate_case_export_row_contract(row: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in _REQUIRED_ROW_KEYS:
        if key not in row:
            errors.append(f"missing_key:{key}")
    case_id = str(row.get("case_id") or "").strip()
    trace_id = str(row.get("trace_id") or "").strip()
    source = str(row.get("source") or "").strip()
    mode_fsm_state = str(row.get("mode_fsm_state") or "").strip()
    if not case_id:
        errors.append("empty_case_id")
    if not trace_id:
        errors.append("empty_trace_id")
    if source != "sqlite.cases.v4":
        errors.append(f"bad_source:{source}")
    if not mode_fsm_state:
        errors.append("empty_mode_fsm_state")
    reasons = row.get("inclusion_reasons")
    if not isinstance(reasons, list) or (not reasons):
        errors.append("bad_inclusion_reasons")
    refs = row.get("provenance_refs")
    if not isinstance(refs, dict):
        errors.append("bad_provenance_refs")
    else:
        if not isinstance(refs.get("event_indices"), list):
            errors.append("bad_provenance_event_indices")
        if not isinstance(refs.get("row_ids"), list):
            errors.append("bad_provenance_row_ids")
    reasoning_context = row.get("reasoning_context")
    if not isinstance(reasoning_context, list):
        errors.append("bad_reasoning_context")
    return errors


def _safe_ratio(numer: int, denom: int) -> float:
    base = max(0, int(denom))
    if base <= 0:
        return 0.0
    return round(max(0.0, float(numer)) / float(base), 4)


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
    db_path = resolve_session_db_path(root)
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"session db missing: {db_path}; run: uv run python -m tools.telemetry.pipeline compile {root}"
        )

    contract_errors = _validate_session_contract_for_export(root, db_path)
    if contract_errors:
        raise ValueError("session contract validation failed: " + "; ".join(contract_errors))

    labels = load_labels_jsonl(os.path.join(root, WEAK_LABELS_PATH))
    prof_min_conf, prof_include_unlabeled = _profile_config(profile)
    threshold = max(float(min_label_confidence), float(prof_min_conf))
    labels_by_event = _label_map(labels, min_confidence=threshold)

    annotations = load_annotations(root, limit=0)
    ann_by_event, ann_by_trace, ann_by_req, session_annotation_tag_counts = _annotation_buckets(annotations)

    with sqlite3.connect(db_path) as conn:
        case_rows = _load_case_rows(conn)
        case_labels_map = _load_case_labels_map(conn)
        case_event_map = _load_case_event_context_map(conn)

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
    planner_parse_stage_counts: Dict[str, int] = {}
    guard_reason_counts: Dict[str, int] = {}
    mode_transition_to_counts: Dict[str, int] = {}
    mode_fsm_state_counts: Dict[str, int] = {}
    mode_transition_reason_counts: Dict[str, int] = {}

    annotation_count = 0
    annotated_row_count = 0
    labeled_row_count = 0
    hard_case_row_count = 0
    planner_failure_row_count = 0
    guard_fallback_row_count = 0
    transition_instability_row_count = 0
    reviewed_row_count = 0

    for case_row in case_rows:
        case_id = str(case_row.get("case_id") or "").strip()
        if not case_id:
            continue
        trace_id = str(case_row.get("trace_id") or "").strip()
        req_id = _as_int(case_row.get("req_id"))

        raw_events = case_event_map.get(case_id, [])
        case_events = [_event_context_item(item) for item in raw_events]
        event_indices = [int(item.get("event_index") or 0) for item in case_events]
        row_ids = [int(item.get("row_id") or 0) for item in case_events]

        weak_labels = _collect_case_weak_labels(event_indices=event_indices, labels_by_event=labels_by_event)
        case_labels = [dict(item) for item in case_labels_map.get(case_id, [])]
        case_annotations = _dedupe_annotations(
            event_indices=event_indices,
            trace_id=trace_id,
            req_id=req_id,
            ann_by_event=ann_by_event,
            ann_by_trace=ann_by_trace,
            ann_by_req=ann_by_req,
        )

        decision = str(case_row.get("decision") or "").strip().lower()
        reviewed = bool(decision)
        reasons = _inclusion_reasons(
            case_row,
            case_labels=case_labels,
            weak_labels=weak_labels,
            annotations=case_annotations,
        )

        if not _profile_allows_case(
            profile=profile,
            reviewed=reviewed,
            reasons=reasons,
            include_unlabeled=effective_include_unlabeled,
        ):
            continue

        contexts = _build_contexts(case_events)
        mode_health = _mode_health_for_case(case_events)
        mode_fsm_state = str(mode_health.get("state") or "unknown").strip().lower() or "unknown"
        transition_reason_counts = mode_health.get("transition_reason_counts") if isinstance(mode_health.get("transition_reason_counts"), dict) else {}
        provenance = {
            "event_indices": sorted(set(event_indices)),
            "row_ids": sorted(set(row_ids)),
            "video_frames": list(contexts["video_context"].get("frames", [])),
            "db_path": db_path,
        }

        row: Dict[str, Any] = {
            "schema_version": int(TELEMETRY_SCHEMA_VERSION_V3),
            "row_version": 1,
            "session_dir": root,
            "case_id": case_id,
            "trace_id": trace_id,
            "req_id": req_id,
            "source": str(case_row.get("source") or "sqlite.cases.v4"),
            "case_status": str(case_row.get("status") or ""),
            "case_severity": str(case_row.get("severity") or ""),
            "case_hardness": _as_float(case_row.get("hardness")),
            "case_summary": str(case_row.get("summary") or ""),
            "case_snippet": str(case_row.get("snippet") or ""),
            "case_start_ts_wall_s": _as_float(case_row.get("start_ts_wall_s")),
            "case_end_ts_wall_s": _as_float(case_row.get("end_ts_wall_s")),
            "case_duration_ms": _as_float(case_row.get("duration_ms")),
            "case_event_count": _as_int(case_row.get("event_count")),
            "case_error_count": _as_int(case_row.get("error_count")),
            "case_warning_count": _as_int(case_row.get("warning_count")),
            "case_max_latency_ms": _as_float(case_row.get("max_latency_ms")),
            "review_decision": decision,
            "review_note": str(case_row.get("note") or ""),
            "reviewer": str(case_row.get("reviewer") or ""),
            "reviewed_at_wall_s": _as_float(case_row.get("reviewed_at_wall_s")),
            "reviewed": reviewed,
            "case_labels": case_labels,
            "weak_labels": weak_labels,
            "labels": weak_labels,
            "annotations": case_annotations,
            "inclusion_reasons": reasons,
            "provenance_refs": provenance,
            "reasoning_context": contexts["reasoning_context"],
            "env_context": contexts["env_context"],
            "planner_context": contexts["planner_context"],
            "arbiter_context": contexts["arbiter_context"],
            "mode_context": contexts["mode_context"],
            "mode_fsm_state": mode_fsm_state,
            "perception_context": contexts["perception_context"],
            "video_context": contexts["video_context"],
        }

        row_errors = _validate_case_export_row_contract(row)
        if row_errors:
            raise ValueError(f"dataset row contract failed for {case_id}: {', '.join(row_errors)}")

        rows.append(row)

        mode_fsm_state_counts[mode_fsm_state] = mode_fsm_state_counts.get(mode_fsm_state, 0) + 1
        for reason_name, reason_count in transition_reason_counts.items():
            name = str(reason_name or "").strip().lower()
            if not name:
                continue
            try:
                count_value = int(reason_count)
            except (TypeError, ValueError):
                continue
            if count_value <= 0:
                continue
            mode_transition_reason_counts[name] = mode_transition_reason_counts.get(name, 0) + count_value

        if reviewed:
            reviewed_row_count += 1
        if case_annotations:
            annotated_row_count += 1
            annotation_count += len(case_annotations)
            for ann in case_annotations:
                tag = str(ann.get("tag") or "").strip().lower()
                if not tag:
                    continue
                annotation_tag_counts[tag] = annotation_tag_counts.get(tag, 0) + 1
        if case_labels or weak_labels:
            labeled_row_count += 1
        if "hard_case" in reasons:
            hard_case_row_count += 1
        if "planner_failure" in reasons:
            planner_failure_row_count += 1
        if "guard_fallback" in reasons:
            guard_fallback_row_count += 1
        if "transition_instability" in reasons:
            transition_instability_row_count += 1
        for reason in reasons:
            inclusion_reason_counts[reason] = inclusion_reason_counts.get(reason, 0) + 1

        status = str(row.get("case_status") or "").strip().lower()
        severity = str(row.get("case_severity") or "").strip().lower()
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        if severity:
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        for item in row.get("reasoning_context", []):
            if not isinstance(item, dict):
                continue
            comp = str(item.get("component") or "").strip().lower()
            src = str(item.get("source") or "").strip().lower()
            phs = str(item.get("phase") or "").strip().lower()
            if comp:
                component_counts[comp] = component_counts.get(comp, 0) + 1
            if src:
                source_counts[src] = source_counts.get(src, 0) + 1
            if phs:
                phase_counts[phs] = phase_counts.get(phs, 0) + 1
            parse_stage = str(item.get("planner_last_parse_stage") or "").strip().lower()
            guard_reason = str(item.get("guard_reason") or "").strip().lower()
            transition_to = str(item.get("mode_transition_to") or "").strip().lower()
            if parse_stage:
                planner_parse_stage_counts[parse_stage] = planner_parse_stage_counts.get(parse_stage, 0) + 1
            if guard_reason:
                guard_reason_counts[guard_reason] = guard_reason_counts.get(guard_reason, 0) + 1
            if transition_to:
                mode_transition_to_counts[transition_to] = mode_transition_to_counts.get(transition_to, 0) + 1

        for item in case_labels:
            name = str(item.get("label") or "").strip().lower()
            if name:
                label_counts[name] = label_counts.get(name, 0) + 1
        for item in weak_labels:
            name = str(item.get("label") or "").strip().lower()
            if name:
                label_counts[name] = label_counts.get(name, 0) + 1

    rows.sort(
        key=lambda row: (
            -float(row.get("case_hardness") or -1.0),
            float(row.get("case_start_ts_wall_s") or 0.0),
            str(row.get("case_id") or ""),
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
            "version": 2,
            "schema_version": int(TELEMETRY_SCHEMA_VERSION_V3),
            "row_granularity": "case",
            "profile": str(profile or "fast"),
            "row_count": len(rows),
            "reviewed_row_count": reviewed_row_count,
            "review_coverage_ratio": _safe_ratio(reviewed_row_count, len(rows)),
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
            "planner_failure_row_count": planner_failure_row_count,
            "guard_fallback_row_count": guard_fallback_row_count,
            "transition_instability_row_count": transition_instability_row_count,
            "annotation_coverage_ratio": _safe_ratio(annotated_row_count, len(rows)),
            "label_coverage_ratio": _safe_ratio(labeled_row_count, len(rows)),
            "hard_case_ratio": _safe_ratio(hard_case_row_count, len(rows)),
            "planner_failure_ratio": _safe_ratio(planner_failure_row_count, len(rows)),
            "guard_fallback_ratio": _safe_ratio(guard_fallback_row_count, len(rows)),
            "transition_instability_ratio": _safe_ratio(transition_instability_row_count, len(rows)),
            "inclusion_reason_counts": inclusion_reason_counts,
            "planner_parse_stage_counts": planner_parse_stage_counts,
            "guard_reason_counts": guard_reason_counts,
            "mode_transition_to_counts": mode_transition_to_counts,
            "mode_fsm_state_counts": mode_fsm_state_counts,
            "mode_transition_reason_counts": mode_transition_reason_counts,
            "annotation_tag_counts": annotation_tag_counts,
            "session_annotation_tag_counts": session_annotation_tag_counts,
            "min_label_confidence": threshold,
            "include_unlabeled": effective_include_unlabeled,
            "output_path": target,
            "db_path": db_path,
        }
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(out_manifest, fh, separators=(",", ":"), ensure_ascii=True)

    return {
        "output_path": target,
        "row_count": len(rows),
        "profile": str(profile or "fast"),
        "manifest_path": manifest_path if bool(write_manifest) else "",
    }
