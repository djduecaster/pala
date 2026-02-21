from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .trace_graph import TraceRecord


QUALITY_REPORT_VERSION = 1


def _coerce_latency_ms(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def build_quality_report(
    *,
    event_count: int,
    source_counts: Mapping[str, int],
    reasoning_index: Sequence[Mapping[str, Any]],
    traces: Sequence[TraceRecord],
    slow_ms: float = 2000.0,
) -> Dict[str, Any]:
    total_events = max(0, int(event_count))
    src = {str(k): max(0, int(v)) for k, v in source_counts.items()}
    reasoning_total = len(reasoning_index)
    trace_total = len(traces)

    reasoning_errors = 0
    reasoning_slow = 0
    parse_fail = 0
    timeout_count = 0
    latencies = []
    for item in reasoning_index:
        severity = str(item.get("severity") or "")
        status = str(item.get("status") or "")
        phase = str(item.get("phase") or "")
        if severity in {"error", "warning"}:
            reasoning_errors += 1
        if "parse_fail" in status or "parse_fail" in phase:
            parse_fail += 1
        if "timeout" in status or "timeout" in phase:
            timeout_count += 1
        latency = _coerce_latency_ms(item.get("latency_ms"))
        if latency is not None:
            latencies.append(latency)
            if latency >= float(slow_ms):
                reasoning_slow += 1

    trace_errors = sum(1 for trace in traces if trace.severity == "error")
    trace_warnings = sum(1 for trace in traces if trace.severity == "warning")

    score = 100.0
    if total_events == 0:
        score = 0.0
    else:
        score -= 25.0 * _safe_ratio(reasoning_errors, max(1, reasoning_total))
        score -= 20.0 * _safe_ratio(trace_errors, max(1, trace_total))
        score -= 15.0 * _safe_ratio(reasoning_slow, max(1, reasoning_total))
        if src.get("agent", 0) == 0:
            score -= 12.0
        if src.get("video_frame", 0) == 0:
            score -= 8.0
        if reasoning_total == 0:
            score -= 15.0
        if trace_total == 0:
            score -= 10.0
    score = max(0.0, min(100.0, score))

    if score >= 80.0:
        grade = "pass"
    elif score >= 60.0:
        grade = "warn"
    else:
        grade = "fail"

    report = {
        "version": QUALITY_REPORT_VERSION,
        "generated_at_wall_s": time.time(),
        "score": round(score, 2),
        "grade": grade,
        "metrics": {
            "event_count": total_events,
            "source_count": len(src),
            "reasoning_count": reasoning_total,
            "trace_count": trace_total,
            "reasoning_errors": reasoning_errors,
            "trace_errors": trace_errors,
            "trace_warnings": trace_warnings,
            "reasoning_slow": reasoning_slow,
            "parse_fail_count": parse_fail,
            "timeout_count": timeout_count,
            "latency_ms_p50": _percentile(latencies, 50.0),
            "latency_ms_p95": _percentile(latencies, 95.0),
            "source_counts": src,
        },
        "gates": {
            "has_events": total_events > 0,
            "has_reasoning": reasoning_total > 0,
            "has_traces": trace_total > 0,
            "agent_heartbeat_seen": src.get("agent", 0) > 0,
            "video_frames_seen": src.get("video_frame", 0) > 0,
            "reasoning_error_rate_ok": _safe_ratio(reasoning_errors, max(1, reasoning_total)) <= 0.25,
            "trace_error_rate_ok": _safe_ratio(trace_errors, max(1, trace_total)) <= 0.30,
        },
    }
    return report


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    idx = int(round((max(0.0, min(100.0, float(pct))) / 100.0) * (len(ordered) - 1)))
    return round(ordered[idx], 2)


def write_quality_report(session_dir: str, report: Mapping[str, Any], *, filename: str = "quality_report.json") -> str:
    path = os.path.join(str(session_dir), filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(report), fh, separators=(",", ":"), ensure_ascii=True)
    return path


def load_quality_report(path_or_dir: str) -> Optional[Dict[str, Any]]:
    path = str(path_or_dir)
    if os.path.isdir(path):
        path = os.path.join(path, "quality_report.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except Exception:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def evaluate_quality_gate(report: Optional[Mapping[str, Any]], mode: str) -> Tuple[bool, str]:
    gate_mode = str(mode or "off").strip().lower()
    if gate_mode == "off":
        return True, "quality gate disabled"
    if report is None:
        return False, "quality report missing"
    grade = str(report.get("grade") or "").lower()
    score = report.get("score")
    score_text = f"{float(score):.1f}" if isinstance(score, (int, float)) else "n/a"
    if gate_mode == "warn":
        if grade == "fail":
            return False, f"quality gate warn failed (grade={grade}, score={score_text})"
        return True, f"quality gate warn passed (grade={grade}, score={score_text})"
    if gate_mode == "strict":
        if grade != "pass":
            return False, f"quality gate strict failed (grade={grade}, score={score_text})"
        return True, f"quality gate strict passed (grade={grade}, score={score_text})"
    return False, f"unknown quality gate mode: {mode}"
