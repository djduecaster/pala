from __future__ import annotations

import argparse
import json
import os
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .storage_sqlite import resolve_session_db_path


VIEWER_SUMMARY_PATH = "viewer_summary.json"
VIEWER_RUNS_PATH = "viewer_runs.jsonl"
DATASET_MANIFEST_PATH = "dataset_manifest.json"
SESSION_MARKER_FILES = (VIEWER_RUNS_PATH, VIEWER_SUMMARY_PATH, "manifest.json", "events.jsonl")


def _expand_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    return os.path.normpath(os.path.expanduser(raw))


def _iter_session_dirs(*, paths: Sequence[str], root: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in paths:
        expanded = _expand_path(value)
        if not expanded:
            continue
        if os.path.isdir(expanded) and expanded not in seen:
            out.append(expanded)
            seen.add(expanded)
    if out:
        return out
    root_path = _expand_path(root)
    if not root_path or (not os.path.isdir(root_path)):
        return out
    for name in sorted(os.listdir(root_path)):
        path = os.path.join(root_path, name)
        if os.path.isdir(path) and _looks_like_session_dir(path) and path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _looks_like_session_dir(path: str) -> bool:
    root = _expand_path(path)
    if not root or (not os.path.isdir(root)):
        return False
    for rel in SESSION_MARKER_FILES:
        if os.path.exists(os.path.join(root, rel)):
            return True
    return False


def _parse_json_obj(raw: str) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _run_ts(run: Mapping[str, Any]) -> float:
    for key in ("ended_at_wall_s", "created_at_wall_s", "started_at_wall_s"):
        value = run.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _load_session_runs(session_dir: str) -> List[Dict[str, Any]]:
    root = _expand_path(session_dir)
    if not root:
        return []
    runs_path = os.path.join(root, VIEWER_RUNS_PATH)
    summary_path = os.path.join(root, VIEWER_SUMMARY_PATH)
    rows: List[Dict[str, Any]] = []
    if os.path.exists(runs_path):
        try:
            with open(runs_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    obj = _parse_json_obj(line)
                    if obj is None:
                        continue
                    obj.setdefault("_artifact", VIEWER_RUNS_PATH)
                    rows.append(obj)
        except Exception:
            rows = []
    if rows:
        rows.sort(key=_run_ts)
        return rows
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            if isinstance(obj, dict):
                obj.setdefault("_artifact", VIEWER_SUMMARY_PATH)
                return [obj]
        except Exception:
            return []
    return []


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    pos = (max(0.0, min(100.0, float(pct))) / 100.0) * (len(ordered) - 1)
    idx = int(round(pos))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None



def _as_counter_map(value: Any) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, int] = {}
    for key, raw in value.items():
        name = str(key or "").strip()
        if not name:
            continue
        if isinstance(raw, (int, float)):
            count = int(raw)
        else:
            continue
        if count <= 0:
            continue
        out[name] = count
    return out


def _merge_counter(dst: Dict[str, int], src: Mapping[str, int]) -> None:
    for key, value in src.items():
        name = str(key or "").strip()
        if not name:
            continue
        count = int(value)
        if count <= 0:
            continue
        dst[name] = dst.get(name, 0) + count


def _load_dataset_manifest_stats(session_dir: str) -> Optional[Dict[str, Any]]:
    root = _expand_path(session_dir)
    if not root:
        return None
    manifest_path = os.path.join(root, DATASET_MANIFEST_PATH)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    row_count = obj.get("row_count")
    row_count_val = int(row_count) if isinstance(row_count, (int, float)) else 0
    return {
        "path": manifest_path,
        "row_count": max(0, row_count_val),
        "planner_failure_ratio": _as_float(obj.get("planner_failure_ratio")),
        "guard_fallback_ratio": _as_float(obj.get("guard_fallback_ratio")),
        "transition_instability_ratio": _as_float(obj.get("transition_instability_ratio")),
        "planner_parse_stage_counts": _as_counter_map(obj.get("planner_parse_stage_counts")),
        "guard_reason_counts": _as_counter_map(obj.get("guard_reason_counts")),
        "mode_transition_to_counts": _as_counter_map(obj.get("mode_transition_to_counts")),
        "mode_fsm_state_counts": _as_counter_map(obj.get("mode_fsm_state_counts")),
        "mode_transition_reason_counts": _as_counter_map(obj.get("mode_transition_reason_counts")),
    }


def _safe_ratio(numer: int, denom: int) -> Optional[float]:
    if denom <= 0:
        return None
    return float(numer) / float(denom)


def _event_count_total(run: Mapping[str, Any]) -> Optional[int]:
    direct = run.get("event_count_total")
    if isinstance(direct, (int, float)):
        return max(0, int(direct))
    counts = run.get("event_counts")
    if isinstance(counts, Mapping):
        total = 0
        seen = False
        for value in counts.values():
            if isinstance(value, (int, float)):
                total += int(value)
                seen = True
        if seen:
            return max(0, int(total))
    return None


def _load_case_stats(session_dir: str) -> Optional[Dict[str, Any]]:
    root = _expand_path(session_dir)
    if not root:
        return None
    db_path = resolve_session_db_path(root)
    if not os.path.exists(db_path):
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            tables = {
                str(row[0])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if row and isinstance(row[0], str)
            }
            if "cases" not in tables:
                return None
            total_cases = int(conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0])
            if "case_reviews" in tables:
                reviewed_cases = int(conn.execute("SELECT COUNT(*) FROM case_reviews").fetchone()[0])
                decision_counts = {
                    str(row[0]): int(row[1])
                    for row in conn.execute(
                        "SELECT decision, COUNT(*) FROM case_reviews GROUP BY decision"
                    ).fetchall()
                    if row and row[0] is not None
                }
            else:
                reviewed_cases = 0
                decision_counts = {}
            if "case_labels" in tables:
                label_counts = {
                    str(row[0]): int(row[1])
                    for row in conn.execute(
                        "SELECT label, COUNT(*) FROM case_labels GROUP BY label"
                    ).fetchall()
                    if row and row[0] is not None
                }
            else:
                label_counts = {}
    except Exception:
        return None
    return {
        "db_path": db_path,
        "total_cases": max(0, int(total_cases)),
        "reviewed_cases": max(0, int(reviewed_cases)),
        "decision_counts": decision_counts,
        "label_counts": label_counts,
    }


def _build_alerts(
    rows: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    latest_case_stats: Optional[Mapping[str, Any]] = None,
    latest_dataset_stats: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    if not rows:
        return []
    latest = rows[-1][1]
    alerts: List[str] = []

    latest_exit = latest.get("exit_code")
    if isinstance(latest_exit, int) and latest_exit != 0:
        alerts.append(f"latest_exit_code_nonzero:{latest_exit}")

    latest_gate = latest.get("quality_gate_passed")
    if latest_gate is False:
        alerts.append("latest_quality_gate_failed")

    latest_curation = latest.get("curation_result")
    if isinstance(latest_curation, dict) and latest_curation.get("ok") is False:
        alerts.append("latest_curation_failed")
    latest_case_source = str(latest.get("case_source") or "").strip()
    latest_case_reason = str(latest.get("case_unavailable_reason") or "").strip()
    if latest_case_source and latest_case_source != "sqlite.cases.v4":
        suffix = f":{latest_case_reason}" if latest_case_reason else ""
        alerts.append(f"latest_case_unavailable{suffix}")

    if isinstance(latest_case_stats, Mapping):
        total_cases = int(latest_case_stats.get("total_cases", 0) or 0)
        reviewed_cases = int(latest_case_stats.get("reviewed_cases", 0) or 0)
        if total_cases >= 10 and reviewed_cases == 0:
            alerts.append("latest_case_reviews_empty")
        elif total_cases >= 20:
            coverage = float(reviewed_cases) / float(total_cases)
            if coverage < 0.15:
                alerts.append(f"latest_case_review_coverage_low:{coverage:.3f}")
        labels = latest_case_stats.get("label_counts") if isinstance(latest_case_stats.get("label_counts"), Mapping) else {}
        planner_fail = int(labels.get("planner_transport_error", 0) or 0)
        guard_fallback = int(labels.get("guard_fallback", 0) or 0)
        transition_churn = int(labels.get("mode_transition_churn", 0) or 0)
        if total_cases > 0 and planner_fail > 0:
            rate = float(planner_fail) / float(total_cases)
            if rate >= 0.25:
                alerts.append(f"latest_planner_failure_high:{rate:.3f}")
        if total_cases > 0 and guard_fallback > 0:
            rate = float(guard_fallback) / float(total_cases)
            if rate >= 0.35:
                alerts.append(f"latest_guard_fallback_high:{rate:.3f}")
        if transition_churn > 0:
            alerts.append(f"latest_transition_churn_cases:{transition_churn}")

    if isinstance(latest_dataset_stats, Mapping):
        planner_fail_ratio = _as_float(latest_dataset_stats.get("planner_failure_ratio"))
        guard_fallback_ratio = _as_float(latest_dataset_stats.get("guard_fallback_ratio"))
        transition_ratio = _as_float(latest_dataset_stats.get("transition_instability_ratio"))
        if planner_fail_ratio is not None and planner_fail_ratio >= 0.35:
            alerts.append(f"latest_dataset_planner_failure_ratio_high:{planner_fail_ratio:.3f}")
        if guard_fallback_ratio is not None and guard_fallback_ratio >= 0.45:
            alerts.append(f"latest_dataset_guard_fallback_ratio_high:{guard_fallback_ratio:.3f}")
        if transition_ratio is not None and transition_ratio >= 0.30:
            alerts.append(f"latest_dataset_transition_instability_ratio_high:{transition_ratio:.3f}")

        row_count = int(latest_dataset_stats.get("row_count", 0) or 0)
        fsm_counts = _as_counter_map(latest_dataset_stats.get("mode_fsm_state_counts"))
        churn_rows = int(fsm_counts.get("churn", 0) or 0)
        if row_count > 0 and churn_rows > 0:
            churn_ratio = float(churn_rows) / float(row_count)
            if churn_ratio >= 0.20:
                alerts.append("latest_dataset_mode_churn_high:{:.3f}".format(churn_ratio))

    latest_transport_peak = _as_float(latest.get("transport_queue_peak_utilization"))
    if latest_transport_peak is not None and latest_transport_peak >= 0.85:
        alerts.append(f"latest_transport_queue_pressure_high:{latest_transport_peak:.3f}")
    latest_local_peak = _as_float(latest.get("local_queue_peak_utilization"))
    if latest_local_peak is not None and latest_local_peak >= 0.85:
        alerts.append(f"latest_local_queue_pressure_high:{latest_local_peak:.3f}")
    latest_reconnect_total = _as_float(latest.get("reconnect_total"))
    if latest_reconnect_total is not None and latest_reconnect_total >= 3.0:
        alerts.append(f"latest_reconnect_churn:{int(round(latest_reconnect_total))}")
    latest_reconnect_stale = _as_float(latest.get("reconnect_stale"))
    if latest_reconnect_stale is not None and latest_reconnect_stale >= 1.0:
        alerts.append(f"latest_stale_reconnects:{int(round(latest_reconnect_stale))}")

    latest_mode = str(latest.get("mode") or "").strip().lower()
    latest_duration_s = _as_float(latest.get("session_duration_s"))
    latest_event_total = _event_count_total(latest)
    latest_rx_peak = _as_float(latest.get("rx_rate_peak_5s"))
    if latest_mode == "live" and latest_duration_s is not None and latest_duration_s >= 30.0:
        if latest_event_total is not None and latest_event_total < 30:
            alerts.append(f"latest_live_low_activity_events:{latest_event_total}/{int(latest_duration_s)}s")
        if latest_rx_peak is not None and latest_rx_peak < 0.5:
            alerts.append(f"latest_live_low_rx_peak:{latest_rx_peak:.3f}")

    if len(rows) < 2:
        return alerts
    previous_runs = [run for _, run in rows[:-1]]

    prev_quality = [_as_float(run.get("quality_score")) for run in previous_runs]
    prev_quality = [v for v in prev_quality if v is not None]
    latest_quality = _as_float(latest.get("quality_score"))
    if prev_quality and latest_quality is not None:
        baseline = _percentile(prev_quality, 50.0)
        if baseline is not None and latest_quality <= (baseline - 10.0):
            alerts.append(f"quality_score_regression:{latest_quality:.2f}<={baseline - 10.0:.2f}")

    prev_agent_drops = [_as_float(run.get("dropped_events_agent")) for run in previous_runs]
    prev_agent_drops = [v for v in prev_agent_drops if v is not None]
    latest_agent_drops = _as_float(latest.get("dropped_events_agent"))
    if prev_agent_drops and latest_agent_drops is not None:
        baseline = _percentile(prev_agent_drops, 95.0)
        threshold = max(20.0, (baseline or 0.0) * 2.0)
        if latest_agent_drops >= threshold:
            alerts.append(f"agent_drop_spike:{latest_agent_drops:.0f}>={threshold:.0f}")

    prev_local_drops = [_as_float(run.get("dropped_events_local")) for run in previous_runs]
    prev_local_drops = [v for v in prev_local_drops if v is not None]
    latest_local_drops = _as_float(latest.get("dropped_events_local"))
    if prev_local_drops and latest_local_drops is not None:
        baseline = _percentile(prev_local_drops, 95.0)
        threshold = max(20.0, (baseline or 0.0) * 2.0)
        if latest_local_drops >= threshold:
            alerts.append(f"local_drop_spike:{latest_local_drops:.0f}>={threshold:.0f}")

    return alerts


def build_run_report(
    *,
    session_dirs: Sequence[str],
    limit: int = 0,
    mode_filter: str = "",
) -> Dict[str, Any]:
    rows: List[Tuple[str, Dict[str, Any]]] = []
    mode_filter_norm = str(mode_filter or "").strip().lower()
    case_stats_by_session: Dict[str, Dict[str, Any]] = {}
    dataset_stats_by_session: Dict[str, Dict[str, Any]] = {}

    for session_dir in session_dirs:
        case_stats = _load_case_stats(session_dir)
        if isinstance(case_stats, dict):
            case_stats_by_session[session_dir] = case_stats
        dataset_stats = _load_dataset_manifest_stats(session_dir)
        if isinstance(dataset_stats, dict):
            dataset_stats_by_session[session_dir] = dataset_stats
        runs = _load_session_runs(session_dir)
        if not runs:
            continue
        for run in runs:
            mode = str(run.get("mode") or "").strip().lower()
            if mode_filter_norm and mode_filter_norm != mode:
                continue
            rows.append((session_dir, run))

    rows.sort(key=lambda item: _run_ts(item[1]))
    if int(limit) > 0 and len(rows) > int(limit):
        rows = rows[-int(limit) :]

    sessions_with_runs: set[str] = set()
    mode_counts: Dict[str, int] = {}
    exit_counts: Dict[str, int] = {}
    duration_values: List[float] = []
    quality_score_values: List[float] = []
    dropped_agent_values: List[float] = []
    dropped_local_values: List[float] = []
    transport_queue_peak_values: List[float] = []
    local_queue_peak_values: List[float] = []
    reconnect_total_values: List[float] = []
    reconnect_stale_values: List[float] = []
    reconnect_disconnect_values: List[float] = []
    reconnect_start_fail_values: List[float] = []
    rx_rate_peak_values: List[float] = []
    event_total_values: List[float] = []
    case_source_counts: Dict[str, int] = {}
    case_unavailable_count = 0
    exit_known = 0
    exit_nonzero = 0
    quality_known = 0
    quality_passed = 0
    curation_known = 0
    curation_ok = 0
    for session_dir, run in rows:
        sessions_with_runs.add(session_dir)
        mode = str(run.get("mode") or "").strip().lower()
        mode_counts[mode or "unknown"] = mode_counts.get(mode or "unknown", 0) + 1
        exit_code = run.get("exit_code")
        if isinstance(exit_code, int):
            exit_known += 1
            if exit_code != 0:
                exit_nonzero += 1
            exit_counts[str(exit_code)] = exit_counts.get(str(exit_code), 0) + 1
        duration_s = run.get("session_duration_s")
        if isinstance(duration_s, (int, float)):
            duration_values.append(max(0.0, float(duration_s)))
        quality_score = run.get("quality_score")
        if isinstance(quality_score, (int, float)):
            quality_score_values.append(float(quality_score))
        dropped_agent = run.get("dropped_events_agent")
        if isinstance(dropped_agent, (int, float)):
            dropped_agent_values.append(max(0.0, float(dropped_agent)))
        dropped_local = run.get("dropped_events_local")
        if isinstance(dropped_local, (int, float)):
            dropped_local_values.append(max(0.0, float(dropped_local)))
        transport_q_peak = _as_float(run.get("transport_queue_peak_utilization"))
        if transport_q_peak is not None:
            transport_queue_peak_values.append(max(0.0, min(1.0, transport_q_peak)))
        local_q_peak = _as_float(run.get("local_queue_peak_utilization"))
        if local_q_peak is not None:
            local_queue_peak_values.append(max(0.0, min(1.0, local_q_peak)))
        reconnect_total = _as_float(run.get("reconnect_total"))
        if reconnect_total is not None:
            reconnect_total_values.append(max(0.0, reconnect_total))
        reconnect_stale = _as_float(run.get("reconnect_stale"))
        if reconnect_stale is not None:
            reconnect_stale_values.append(max(0.0, reconnect_stale))
        reconnect_disconnect = _as_float(run.get("reconnect_disconnect"))
        if reconnect_disconnect is not None:
            reconnect_disconnect_values.append(max(0.0, reconnect_disconnect))
        reconnect_start_fail = _as_float(run.get("reconnect_start_fail"))
        if reconnect_start_fail is not None:
            reconnect_start_fail_values.append(max(0.0, reconnect_start_fail))
        rx_rate_peak = _as_float(run.get("rx_rate_peak_5s"))
        if rx_rate_peak is not None:
            rx_rate_peak_values.append(max(0.0, rx_rate_peak))
        event_total = _event_count_total(run)
        if event_total is not None:
            event_total_values.append(max(0.0, float(event_total)))
        case_source = str(run.get("case_source") or "").strip()
        if case_source:
            case_source_counts[case_source] = case_source_counts.get(case_source, 0) + 1
            if case_source != "sqlite.cases.v4":
                case_unavailable_count += 1
        qgp = run.get("quality_gate_passed")
        if isinstance(qgp, bool):
            quality_known += 1
            if qgp:
                quality_passed += 1
        curation = run.get("curation_result")
        if isinstance(curation, dict):
            ok = curation.get("ok")
            if isinstance(ok, bool):
                curation_known += 1
                if ok:
                    curation_ok += 1

    total_cases = 0
    reviewed_cases = 0
    case_decision_counts: Dict[str, int] = {}
    case_label_counts: Dict[str, int] = {}

    dataset_rows_total = 0
    dataset_planner_failure_ratios: List[float] = []
    dataset_guard_fallback_ratios: List[float] = []
    dataset_transition_instability_ratios: List[float] = []
    dataset_parse_stage_counts: Dict[str, int] = {}
    dataset_guard_reason_counts: Dict[str, int] = {}
    dataset_transition_to_counts: Dict[str, int] = {}
    dataset_mode_fsm_state_counts: Dict[str, int] = {}
    dataset_transition_reason_counts: Dict[str, int] = {}
    for case_stats in case_stats_by_session.values():
        total_cases += int(case_stats.get("total_cases", 0) or 0)
        reviewed_cases += int(case_stats.get("reviewed_cases", 0) or 0)
        decisions = case_stats.get("decision_counts")
        if isinstance(decisions, dict):
            for key, value in decisions.items():
                label = str(key)
                case_decision_counts[label] = case_decision_counts.get(label, 0) + int(value or 0)
        labels = case_stats.get("label_counts")
        if isinstance(labels, dict):
            for key, value in labels.items():
                label = str(key)
                case_label_counts[label] = case_label_counts.get(label, 0) + int(value or 0)

    for dataset_stats in dataset_stats_by_session.values():
        dataset_rows_total += int(dataset_stats.get("row_count", 0) or 0)
        planner_ratio = _as_float(dataset_stats.get("planner_failure_ratio"))
        guard_ratio = _as_float(dataset_stats.get("guard_fallback_ratio"))
        transition_ratio = _as_float(dataset_stats.get("transition_instability_ratio"))
        if planner_ratio is not None:
            dataset_planner_failure_ratios.append(max(0.0, planner_ratio))
        if guard_ratio is not None:
            dataset_guard_fallback_ratios.append(max(0.0, guard_ratio))
        if transition_ratio is not None:
            dataset_transition_instability_ratios.append(max(0.0, transition_ratio))
        _merge_counter(dataset_parse_stage_counts, _as_counter_map(dataset_stats.get("planner_parse_stage_counts")))
        _merge_counter(dataset_guard_reason_counts, _as_counter_map(dataset_stats.get("guard_reason_counts")))
        _merge_counter(dataset_transition_to_counts, _as_counter_map(dataset_stats.get("mode_transition_to_counts")))
        _merge_counter(dataset_mode_fsm_state_counts, _as_counter_map(dataset_stats.get("mode_fsm_state_counts")))
        _merge_counter(dataset_transition_reason_counts, _as_counter_map(dataset_stats.get("mode_transition_reason_counts")))

    latest = None
    latest_case_stats: Optional[Dict[str, Any]] = None
    latest_dataset_stats: Optional[Dict[str, Any]] = None
    if rows:
        latest_session, latest_run = rows[-1]
        latest_case_stats = case_stats_by_session.get(latest_session)
        latest_dataset_stats = dataset_stats_by_session.get(latest_session)
        latest = {
            "session_dir": latest_session,
            "run_id": latest_run.get("run_id"),
            "mode": latest_run.get("mode"),
            "exit_code": latest_run.get("exit_code"),
            "ts_wall_s": _run_ts(latest_run),
            "quality_score": latest_run.get("quality_score"),
            "quality_gate_passed": latest_run.get("quality_gate_passed"),
            "dropped_events_agent": latest_run.get("dropped_events_agent"),
            "dropped_events_local": latest_run.get("dropped_events_local"),
            "transport_queue_peak_utilization": latest_run.get("transport_queue_peak_utilization"),
            "local_queue_peak_utilization": latest_run.get("local_queue_peak_utilization"),
            "rx_rate_peak_5s": latest_run.get("rx_rate_peak_5s"),
            "reconnect_total": latest_run.get("reconnect_total"),
            "reconnect_stale": latest_run.get("reconnect_stale"),
            "reconnect_disconnect": latest_run.get("reconnect_disconnect"),
            "reconnect_start_fail": latest_run.get("reconnect_start_fail"),
            "event_count_total": _event_count_total(latest_run),
            "case_source": latest_run.get("case_source"),
            "case_unavailable_reason": latest_run.get("case_unavailable_reason"),
        }
        if isinstance(latest_case_stats, dict):
            latest_total = int(latest_case_stats.get("total_cases", 0) or 0)
            latest_reviewed = int(latest_case_stats.get("reviewed_cases", 0) or 0)
            latest["case_total"] = latest_total
            latest["case_reviewed"] = latest_reviewed
            latest["case_review_coverage"] = (
                (float(latest_reviewed) / float(latest_total)) if latest_total > 0 else None
            )
        if isinstance(latest_dataset_stats, dict):
            latest["dataset_row_count"] = int(latest_dataset_stats.get("row_count", 0) or 0)
            latest["dataset_planner_failure_ratio"] = _as_float(latest_dataset_stats.get("planner_failure_ratio"))
            latest["dataset_guard_fallback_ratio"] = _as_float(latest_dataset_stats.get("guard_fallback_ratio"))
            latest["dataset_transition_instability_ratio"] = _as_float(latest_dataset_stats.get("transition_instability_ratio"))
            latest_fsm_counts = _as_counter_map(latest_dataset_stats.get("mode_fsm_state_counts"))
            churn_rows = int(latest_fsm_counts.get("churn", 0) or 0)
            latest_rows = int(latest_dataset_stats.get("row_count", 0) or 0)
            latest["dataset_mode_churn_ratio"] = (float(churn_rows) / float(latest_rows)) if latest_rows > 0 else None

    avg_duration = None
    if duration_values:
        avg_duration = sum(duration_values) / len(duration_values)
    avg_quality = None
    if quality_score_values:
        avg_quality = sum(quality_score_values) / len(quality_score_values)
    alerts = _build_alerts(rows, latest_case_stats=latest_case_stats, latest_dataset_stats=latest_dataset_stats)
    quality_failed = max(0, quality_known - quality_passed)
    curation_failed = max(0, curation_known - curation_ok)
    sessions_without_runs = sorted(path for path in session_dirs if path not in sessions_with_runs)
    if session_dirs and len(rows) <= 0:
        alerts.append("no_run_artifacts")
    elif sessions_without_runs:
        alerts.append(f"sessions_missing_runs:{len(sessions_without_runs)}")

    return {
        "sessions_scanned": len(session_dirs),
        "sessions_with_runs": len(sessions_with_runs),
        "sessions_without_runs_count": len(sessions_without_runs),
        "sessions_without_runs": sessions_without_runs[:20],
        "runs_total": len(rows),
        "mode_counts": mode_counts,
        "exit_code_counts": exit_counts,
        "exit_code_sample_count": exit_known,
        "exit_nonzero_rate": _safe_ratio(exit_nonzero, exit_known),
        "quality_gate_pass_rate": (quality_passed / quality_known) if quality_known > 0 else None,
        "quality_gate_sample_count": quality_known,
        "quality_gate_fail_rate": _safe_ratio(quality_failed, quality_known),
        "curation_success_rate": (curation_ok / curation_known) if curation_known > 0 else None,
        "curation_sample_count": curation_known,
        "curation_fail_rate": _safe_ratio(curation_failed, curation_known),
        "duration_s": {
            "avg": avg_duration,
            "p50": _percentile(duration_values, 50.0),
            "p95": _percentile(duration_values, 95.0),
        },
        "quality_score": {
            "avg": avg_quality,
            "p50": _percentile(quality_score_values, 50.0),
            "p95": _percentile(quality_score_values, 95.0),
        },
        "drops": {
            "agent_total": sum(dropped_agent_values) if dropped_agent_values else 0.0,
            "agent_p95": _percentile(dropped_agent_values, 95.0),
            "local_total": sum(dropped_local_values) if dropped_local_values else 0.0,
            "local_p95": _percentile(dropped_local_values, 95.0),
        },
        "stream_health": {
            "transport_queue_peak_sample_count": len(transport_queue_peak_values),
            "transport_queue_peak_p95": _percentile(transport_queue_peak_values, 95.0),
            "transport_queue_peak_max": (max(transport_queue_peak_values) if transport_queue_peak_values else None),
            "local_queue_peak_sample_count": len(local_queue_peak_values),
            "local_queue_peak_p95": _percentile(local_queue_peak_values, 95.0),
            "local_queue_peak_max": (max(local_queue_peak_values) if local_queue_peak_values else None),
            "reconnect_total_sample_count": len(reconnect_total_values),
            "reconnect_total_avg": (
                (sum(reconnect_total_values) / len(reconnect_total_values)) if reconnect_total_values else None
            ),
            "reconnect_total_p95": _percentile(reconnect_total_values, 95.0),
            "reconnect_stale_total": sum(reconnect_stale_values) if reconnect_stale_values else 0.0,
            "reconnect_disconnect_total": (
                sum(reconnect_disconnect_values) if reconnect_disconnect_values else 0.0
            ),
            "reconnect_start_fail_total": sum(reconnect_start_fail_values) if reconnect_start_fail_values else 0.0,
            "rx_rate_peak_sample_count": len(rx_rate_peak_values),
            "rx_rate_peak_p95": _percentile(rx_rate_peak_values, 95.0),
            "rx_rate_peak_max": (max(rx_rate_peak_values) if rx_rate_peak_values else None),
            "event_total_sample_count": len(event_total_values),
            "event_total_avg": ((sum(event_total_values) / len(event_total_values)) if event_total_values else None),
            "event_total_p50": _percentile(event_total_values, 50.0),
            "event_total_p95": _percentile(event_total_values, 95.0),
        },
        "cases": {
            "source_counts": case_source_counts,
            "unavailable_count": case_unavailable_count,
            "sessions_with_case_db": len(case_stats_by_session),
            "total_cases": int(total_cases),
            "reviewed_cases": int(reviewed_cases),
            "review_coverage": (float(reviewed_cases) / float(total_cases)) if total_cases > 0 else None,
            "decision_counts": case_decision_counts,
            "label_counts": case_label_counts,
        },
        "dataset_v4": {
            "sessions_with_manifest": len(dataset_stats_by_session),
            "rows_total": int(dataset_rows_total),
            "planner_failure_ratio_p50": _percentile(dataset_planner_failure_ratios, 50.0),
            "planner_failure_ratio_p95": _percentile(dataset_planner_failure_ratios, 95.0),
            "guard_fallback_ratio_p50": _percentile(dataset_guard_fallback_ratios, 50.0),
            "guard_fallback_ratio_p95": _percentile(dataset_guard_fallback_ratios, 95.0),
            "transition_instability_ratio_p50": _percentile(dataset_transition_instability_ratios, 50.0),
            "transition_instability_ratio_p95": _percentile(dataset_transition_instability_ratios, 95.0),
            "planner_parse_stage_counts": dataset_parse_stage_counts,
            "guard_reason_counts": dataset_guard_reason_counts,
            "mode_transition_to_counts": dataset_transition_to_counts,
            "mode_fsm_state_counts": dataset_mode_fsm_state_counts,
            "mode_transition_reason_counts": dataset_transition_reason_counts,
            "mode_churn_ratio_global": (float(dataset_mode_fsm_state_counts.get("churn", 0)) / float(dataset_rows_total)) if dataset_rows_total > 0 else None,
        },
        "alerts": alerts,
        "alerts_count": len(alerts),
        "health": "warn" if alerts else "ok",
        "latest_run": latest,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize telemetry viewer run artifacts across session directories.")
    parser.add_argument("paths", nargs="*", help="Session directories (defaults to scanning --root).")
    parser.add_argument("--root", default="logs/telemetry", help="Root directory to scan when no paths are provided.")
    parser.add_argument("--limit", type=int, default=0, help="Include only the latest N runs in output stats.")
    parser.add_argument("--mode", default="", choices=["", "live", "replay", "curate"], help="Optional mode filter.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when alerts are detected.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    session_dirs = _iter_session_dirs(paths=[str(p) for p in args.paths], root=str(args.root))
    report = build_run_report(session_dirs=session_dirs, limit=int(args.limit), mode_filter=str(args.mode))
    if args.json:
        print(json.dumps(report, separators=(",", ":"), ensure_ascii=True))
        return 0

    print("Telemetry Run Report")
    print(
        "summary: "
        f"sessions_scanned={report['sessions_scanned']} "
        f"sessions_with_runs={report['sessions_with_runs']} "
        f"sessions_without_runs={report.get('sessions_without_runs_count', 0)} "
        f"runs_total={report['runs_total']}"
    )
    print(f"modes: {report['mode_counts'] or '{}'}")
    print(f"exit_codes: {report['exit_code_counts'] or '{}'}")
    exit_nonzero_rate = report.get("exit_nonzero_rate")
    exit_n = int(report.get("exit_code_sample_count", 0) or 0)
    if isinstance(exit_nonzero_rate, float):
        print(f"exit_nonzero_rate: {exit_nonzero_rate:.3f} (n={exit_n})")
    else:
        print("exit_nonzero_rate: n/a")
    q_rate = report.get("quality_gate_pass_rate")
    q_n = int(report.get("quality_gate_sample_count", 0) or 0)
    if isinstance(q_rate, float):
        print(f"quality_gate_pass_rate: {q_rate:.3f} (n={q_n})")
    else:
        print("quality_gate_pass_rate: n/a")
    q_fail_rate = report.get("quality_gate_fail_rate")
    if isinstance(q_fail_rate, float):
        print(f"quality_gate_fail_rate: {q_fail_rate:.3f} (n={q_n})")
    c_rate = report.get("curation_success_rate")
    c_n = int(report.get("curation_sample_count", 0) or 0)
    if isinstance(c_rate, float):
        print(f"curation_success_rate: {c_rate:.3f} (n={c_n})")
    else:
        print("curation_success_rate: n/a")
    c_fail_rate = report.get("curation_fail_rate")
    if isinstance(c_fail_rate, float):
        print(f"curation_fail_rate: {c_fail_rate:.3f} (n={c_n})")
    duration = report.get("duration_s") if isinstance(report.get("duration_s"), dict) else {}
    print(
        "duration_s: "
        f"avg={duration.get('avg') if duration else None} "
        f"p50={duration.get('p50') if duration else None} "
        f"p95={duration.get('p95') if duration else None}"
    )
    quality_score = report.get("quality_score") if isinstance(report.get("quality_score"), dict) else {}
    print(
        "quality_score: "
        f"avg={quality_score.get('avg') if quality_score else None} "
        f"p50={quality_score.get('p50') if quality_score else None} "
        f"p95={quality_score.get('p95') if quality_score else None}"
    )
    drops = report.get("drops") if isinstance(report.get("drops"), dict) else {}
    print(
        "drops: "
        f"agent_total={drops.get('agent_total') if drops else None} "
        f"agent_p95={drops.get('agent_p95') if drops else None} "
        f"local_total={drops.get('local_total') if drops else None} "
        f"local_p95={drops.get('local_p95') if drops else None}"
    )
    stream_health = report.get("stream_health") if isinstance(report.get("stream_health"), dict) else {}
    print(
        "stream_health: "
        f"agent_q_peak_max={stream_health.get('transport_queue_peak_max') if stream_health else None} "
        f"viewer_q_peak_max={stream_health.get('local_queue_peak_max') if stream_health else None} "
        f"reconnect_p95={stream_health.get('reconnect_total_p95') if stream_health else None} "
        f"rx_rate_peak_max={stream_health.get('rx_rate_peak_max') if stream_health else None} "
        f"event_p50={stream_health.get('event_total_p50') if stream_health else None}"
    )
    cases = report.get("cases") if isinstance(report.get("cases"), dict) else {}
    print(
        "cases: "
        f"sources={cases.get('source_counts') if cases else None} "
        f"unavailable_count={cases.get('unavailable_count') if cases else None} "
        f"total={cases.get('total_cases') if cases else None} "
        f"reviewed={cases.get('reviewed_cases') if cases else None} "
        f"coverage={cases.get('review_coverage') if cases else None}"
    )
    dataset_v4 = report.get("dataset_v4") if isinstance(report.get("dataset_v4"), dict) else {}
    print(
        "dataset_v4: "
        f"sessions={dataset_v4.get('sessions_with_manifest') if dataset_v4 else None} "
        f"rows_total={dataset_v4.get('rows_total') if dataset_v4 else None} "
        f"planner_fail_p50={dataset_v4.get('planner_failure_ratio_p50') if dataset_v4 else None} "
        f"guard_fallback_p50={dataset_v4.get('guard_fallback_ratio_p50') if dataset_v4 else None} "
        f"transition_instability_p50={dataset_v4.get('transition_instability_ratio_p50') if dataset_v4 else None} "
        f"mode_churn_global={dataset_v4.get('mode_churn_ratio_global') if dataset_v4 else None}"
    )
    latest = report.get("latest_run")
    if isinstance(latest, dict):
        print(
            "latest: "
            f"session={latest.get('session_dir')} mode={latest.get('mode')} "
            f"exit={latest.get('exit_code')} run_id={latest.get('run_id')} "
            f"events={latest.get('event_count_total')} rx_peak={latest.get('rx_rate_peak_5s')}"
        )
    else:
        print("latest: n/a")
    alerts = report.get("alerts") if isinstance(report.get("alerts"), list) else []
    if alerts:
        print("alerts:")
        for item in alerts:
            print(f"- {item}")
    else:
        print("alerts: none")
    print(f"health: {report.get('health')}")
    if bool(args.strict) and bool(alerts):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
