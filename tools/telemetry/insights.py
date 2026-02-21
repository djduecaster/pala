from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .labels import load_labels_jsonl
from .quality import load_quality_report
from .trace_graph import TraceRecord, load_trace_index, resolve_trace_index_path


IMPROVEMENT_REPORT_VERSION = 1


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


def _load_reasoning_index(session_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(session_dir, "reasoning_index.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception:
        return []
    events = obj.get("events") if isinstance(obj, dict) else None
    if not isinstance(events, list):
        return []
    return [item for item in events if isinstance(item, dict)]


def _load_traces(session_dir: str, manifest: Optional[Dict[str, Any]]) -> List[TraceRecord]:
    trace_path = resolve_trace_index_path(session_dir, manifest)
    if not os.path.exists(trace_path):
        return []
    try:
        return load_trace_index(trace_path)
    except Exception:
        return []


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _latency_percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    idx = int(round((max(0.0, min(100.0, float(pct))) / 100.0) * (len(ordered) - 1)))
    return round(ordered[idx], 2)


def _top_signatures(reasoning: Sequence[Mapping[str, Any]], *, limit: int = 6) -> List[Dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in reasoning:
        phase = str(row.get("phase") or "unknown")
        status = str(row.get("status") or "unknown")
        severity = str(row.get("severity") or "info")
        key = f"{phase}|{status}|{severity}"
        counts[key] += 1
    total = max(1, sum(counts.values()))
    out: List[Dict[str, Any]] = []
    for signature, count in counts.most_common(max(1, int(limit))):
        phase, status, severity = signature.split("|", 2)
        out.append(
            {
                "signature": signature,
                "phase": phase,
                "status": status,
                "severity": severity,
                "count": int(count),
                "rate": round((100.0 * float(count) / float(total)), 2),
            }
        )
    return out


def _top_trace_failures(traces: Sequence[TraceRecord], *, limit: int = 6) -> List[Dict[str, Any]]:
    counts: Counter[str] = Counter()
    for trace in traces:
        status = str(trace.status or "unknown")
        severity = str(trace.severity or "info")
        counts[f"{status}|{severity}"] += 1
    total = max(1, sum(counts.values()))
    out: List[Dict[str, Any]] = []
    for key, count in counts.most_common(max(1, int(limit))):
        status, severity = key.split("|", 1)
        out.append(
            {
                "status": status,
                "severity": severity,
                "count": int(count),
                "rate": round((100.0 * float(count) / float(total)), 2),
            }
        )
    return out


def _failure_fingerprints(
    *,
    reasoning: Sequence[Mapping[str, Any]],
    traces: Sequence[TraceRecord],
    labels: Sequence[Mapping[str, Any]],
    limit: int = 8,
) -> List[Dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in reasoning:
        phase = str(row.get("phase") or "unknown")
        status = str(row.get("status") or "unknown")
        sev = str(row.get("severity") or "info")
        if sev == "error" or "fail" in status or "timeout" in status:
            counts[f"reasoning:{phase}:{status}:{sev}"] += 1
    for trace in traces:
        status = str(trace.status or "unknown")
        sev = str(trace.severity or "info")
        if sev in {"error", "warning"} or status not in {"ok", "success"}:
            counts[f"trace:{status}:{sev}"] += 1
    for row in labels:
        label = str(row.get("label") or "").strip()
        if label:
            counts[f"label:{label}"] += 1
    out: List[Dict[str, Any]] = []
    total = max(1, sum(counts.values()))
    for key, count in counts.most_common(max(1, int(limit))):
        out.append(
            {
                "fingerprint": key,
                "count": int(count),
                "rate": round((100.0 * float(count) / float(total)), 2),
            }
        )
    return out


def _session_metrics_from_dir(session_dir: str) -> Optional[Dict[str, float]]:
    quality = load_quality_report(session_dir)
    improvement = load_improvement_report(session_dir)
    if quality is None and improvement is None:
        return None
    summary = improvement.get("summary") if isinstance(improvement, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    reasoning_count = int(summary.get("reasoning_count", 0) or 0)
    parse_fail = int(summary.get("parse_fail_count", 0) or 0)
    timeout = int(summary.get("timeout_count", 0) or 0)
    slow = int(summary.get("slow_count", 0) or 0)
    return {
        "quality_score": float(quality.get("score", 0.0) if isinstance(quality, dict) else 0.0),
        "parse_fail_rate": _safe_ratio(parse_fail, max(1, reasoning_count)),
        "timeout_rate": _safe_ratio(timeout, max(1, reasoning_count)),
        "slow_rate": _safe_ratio(slow, max(1, reasoning_count)),
    }


def _golden_comparison(
    *,
    current: Dict[str, float],
    golden_sessions: Sequence[str],
) -> Optional[Dict[str, Any]]:
    baselines: List[Dict[str, float]] = []
    for item in golden_sessions:
        metrics = _session_metrics_from_dir(str(item))
        if metrics is not None:
            baselines.append(metrics)
    if not baselines:
        return None
    n = float(len(baselines))
    avg = {
        "quality_score": sum(m["quality_score"] for m in baselines) / n,
        "parse_fail_rate": sum(m["parse_fail_rate"] for m in baselines) / n,
        "timeout_rate": sum(m["timeout_rate"] for m in baselines) / n,
        "slow_rate": sum(m["slow_rate"] for m in baselines) / n,
    }
    return {
        "golden_session_count": int(len(baselines)),
        "baseline": {k: round(v, 5) for k, v in avg.items()},
        "delta": {
            "quality_score": round(current["quality_score"] - avg["quality_score"], 5),
            "parse_fail_rate": round(current["parse_fail_rate"] - avg["parse_fail_rate"], 5),
            "timeout_rate": round(current["timeout_rate"] - avg["timeout_rate"], 5),
            "slow_rate": round(current["slow_rate"] - avg["slow_rate"], 5),
        },
    }


def _label_counts(labels: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in labels:
        name = str(row.get("label") or "").strip()
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def _recommendations(
    *,
    reasoning_total: int,
    parse_fail_count: int,
    timeout_count: int,
    slow_count: int,
    labels_count: int,
    traces_count: int,
    quality_grade: str,
) -> List[Dict[str, str]]:
    recs: List[Dict[str, str]] = []
    parse_fail_rate = _safe_ratio(parse_fail_count, reasoning_total)
    timeout_rate = _safe_ratio(timeout_count, reasoning_total)
    slow_rate = _safe_ratio(slow_count, reasoning_total)

    if quality_grade in {"warn", "fail"}:
        recs.append(
            {
                "priority": "high",
                "title": "Raise session quality gate to pass",
                "why": "Current session grade is below pass.",
                "action": "Stabilize major failure sources before dataset export.",
            }
        )
    if parse_fail_rate >= 0.12:
        recs.append(
            {
                "priority": "high",
                "title": "Reduce planner parse failures",
                "why": f"parse_fail rate is {parse_fail_rate*100:.1f}%",
                "action": "Harden parser schema + add fallback response normalizers.",
            }
        )
    if timeout_rate >= 0.08:
        recs.append(
            {
                "priority": "high",
                "title": "Address planner/model timeouts",
                "why": f"timeout rate is {timeout_rate*100:.1f}%",
                "action": "Lower prompt size, tighten timeouts, or add faster fallback model path.",
            }
        )
    if slow_rate >= 0.3:
        recs.append(
            {
                "priority": "medium",
                "title": "Lower long-tail latency",
                "why": f"slow-event rate is {slow_rate*100:.1f}%",
                "action": "Profile latency contributors and trim high-cost reasoning branches.",
            }
        )
    if traces_count == 0:
        recs.append(
            {
                "priority": "high",
                "title": "Enable request trace visibility",
                "why": "No traces were captured.",
                "action": "Verify timeline source wiring and trace correlation IDs.",
            }
        )
    if labels_count < 10:
        recs.append(
            {
                "priority": "medium",
                "title": "Increase labeled data yield",
                "why": f"Only {labels_count} weak labels produced.",
                "action": "Run longer capture or targeted failure scenarios to raise label density.",
            }
        )
    if reasoning_total < 20:
        recs.append(
            {
                "priority": "low",
                "title": "Increase scenario coverage",
                "why": f"Only {reasoning_total} reasoning rows observed.",
                "action": "Capture additional sessions across varied environments/lighting.",
            }
        )
    if not recs:
        recs.append(
            {
                "priority": "low",
                "title": "Maintain current quality trajectory",
                "why": "No dominant bottleneck crossed action thresholds.",
                "action": "Continue collecting sessions and monitor for regressions.",
            }
        )
    return recs


def build_improvement_report(
    session_dir: str,
    *,
    golden_sessions: Optional[Sequence[str]] = None,
    scenario_tags: Optional[Sequence[str]] = None,
    goal_tags: Optional[Sequence[str]] = None,
    runbook: str = "",
) -> Dict[str, Any]:
    root = str(session_dir)
    manifest = _load_manifest(root)
    reasoning = _load_reasoning_index(root)
    traces = _load_traces(root, manifest)
    labels = load_labels_jsonl(os.path.join(root, "labels.weak.jsonl"))
    quality = load_quality_report(root) or {}

    parse_fail_count = 0
    timeout_count = 0
    slow_count = 0
    latencies: List[float] = []
    for row in reasoning:
        status = str(row.get("status") or "").lower()
        phase = str(row.get("phase") or "").lower()
        if "parse_fail" in status or "parse_fail" in phase:
            parse_fail_count += 1
        if "timeout" in status or "timeout" in phase:
            timeout_count += 1
        latency = row.get("latency_ms")
        if isinstance(latency, (int, float)):
            value = float(latency)
            latencies.append(value)
            if value >= 2000.0:
                slow_count += 1

    label_counts = _label_counts(labels)
    quality_grade = str(quality.get("grade") or "unknown").lower()
    quality_score = float(quality.get("score", 0.0) or 0.0)
    reasoning_count = len(reasoning)
    current_metrics = {
        "quality_score": quality_score,
        "parse_fail_rate": _safe_ratio(parse_fail_count, max(1, reasoning_count)),
        "timeout_rate": _safe_ratio(timeout_count, max(1, reasoning_count)),
        "slow_rate": _safe_ratio(slow_count, max(1, reasoning_count)),
    }
    golden_cmp = _golden_comparison(current=current_metrics, golden_sessions=list(golden_sessions or []))

    report = {
        "version": IMPROVEMENT_REPORT_VERSION,
        "generated_at_wall_s": time.time(),
        "session_dir": root,
        "scenario_tags": [str(x) for x in (scenario_tags or []) if str(x).strip()],
        "goal_tags": [str(x) for x in (goal_tags or []) if str(x).strip()],
        "runbook": str(runbook or ""),
        "quality": {
            "grade": quality_grade,
            "score": quality_score,
        },
        "summary": {
            "event_count": int((manifest or {}).get("event_count", 0) or 0),
            "reasoning_count": reasoning_count,
            "trace_count": len(traces),
            "weak_label_count": len(labels),
            "parse_fail_count": parse_fail_count,
            "timeout_count": timeout_count,
            "slow_count": slow_count,
            "latency_ms_p50": _latency_percentile(latencies, 50.0),
            "latency_ms_p95": _latency_percentile(latencies, 95.0),
        },
        "top_reasoning_signatures": _top_signatures(reasoning),
        "top_trace_failures": _top_trace_failures(traces),
        "failure_fingerprints": _failure_fingerprints(reasoning=reasoning, traces=traces, labels=labels),
        "label_breakdown": label_counts,
        "golden_comparison": golden_cmp,
        "recommendations": _recommendations(
            reasoning_total=reasoning_count,
            parse_fail_count=parse_fail_count,
            timeout_count=timeout_count,
            slow_count=slow_count,
            labels_count=len(labels),
            traces_count=len(traces),
            quality_grade=quality_grade,
        ),
    }
    return report


def write_improvement_report(session_dir: str, report: Mapping[str, Any], *, filename: str = "improvement_report.json") -> str:
    path = os.path.join(str(session_dir), filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(report), fh, separators=(",", ":"), ensure_ascii=True)
    return path


def load_improvement_report(path_or_dir: str) -> Optional[Dict[str, Any]]:
    path = str(path_or_dir)
    if os.path.isdir(path):
        path = os.path.join(path, "improvement_report.json")
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
