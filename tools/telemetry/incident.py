from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional

from .doctor import build_doctor_report, load_doctor_report
from .insights import build_improvement_report, load_improvement_report
from .quality import load_quality_report
from .schema_v3 import INCIDENT_REPORT_PATH
from .storage_sqlite import build_session_db, query_session_db, resolve_session_db_path


INCIDENT_MARKDOWN_PATH = "incident_report.md"


def _load_manifest(session_dir: str) -> Dict[str, Any]:
    path = os.path.join(str(session_dir), "manifest.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _severity_from_signals(
    *,
    quality_grade: str,
    doctor_grade: str,
    parse_fail_count: int,
    timeout_count: int,
    error_issues: int,
    warning_issues: int,
) -> str:
    if doctor_grade == "fail" or quality_grade == "fail" or error_issues > 0:
        return "critical"
    if parse_fail_count > 0 or timeout_count > 0:
        return "high"
    if warning_issues > 0 or quality_grade == "warn" or doctor_grade == "warn":
        return "medium"
    return "low"


def _top_recommendations(
    *,
    doctor_report: Mapping[str, Any],
    improvement_report: Mapping[str, Any],
    limit: int = 6,
) -> List[str]:
    out: List[str] = []
    doc_recs = doctor_report.get("recommendations")
    if isinstance(doc_recs, list):
        for item in doc_recs:
            text = str(item or "").strip()
            if text:
                out.append(text)
    imp_recs = improvement_report.get("recommendations")
    if isinstance(imp_recs, list):
        for row in imp_recs:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            action = str(row.get("action") or "").strip()
            if title and action:
                out.append(f"{title}: {action}")
            elif title:
                out.append(title)
    dedup: List[str] = []
    seen = set()
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    return dedup[: max(1, int(limit))]


def build_incident_report(
    session_dir: str,
    *,
    query: str = "kind:trace severity:error status:parse_fail status:timeout",
    limit: int = 8,
    quality_report: Optional[Mapping[str, Any]] = None,
    doctor_report: Optional[Mapping[str, Any]] = None,
    improvement_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    root = str(session_dir)
    manifest = _load_manifest(root)
    quality = dict(quality_report or load_quality_report(root) or {})
    doctor = dict(doctor_report or load_doctor_report(root) or {})
    improvement = dict(improvement_report or load_improvement_report(root) or {})

    if not doctor:
        doctor = build_doctor_report(root, manifest=manifest, quality_report=quality, improvement_report=improvement)
    if not improvement:
        improvement = build_improvement_report(root)

    summary = improvement.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    parse_fail_count = int(summary.get("parse_fail_count", 0) or 0)
    timeout_count = int(summary.get("timeout_count", 0) or 0)
    reasoning_count = int(summary.get("reasoning_count", 0) or 0)
    weak_label_count = int(summary.get("weak_label_count", 0) or 0)
    traces_count = int(summary.get("trace_count", 0) or 0)

    quality_grade = str(quality.get("grade") or "").lower()
    quality_score = quality.get("score")
    readiness = doctor.get("readiness")
    readiness = readiness if isinstance(readiness, dict) else {}
    doctor_grade = str(readiness.get("grade") or "").lower()
    doctor_score = readiness.get("score")
    doctor_summary = doctor.get("summary")
    doctor_summary = doctor_summary if isinstance(doctor_summary, dict) else {}
    error_issues = int(doctor_summary.get("error_count", 0) or 0)
    warning_issues = int(doctor_summary.get("warning_count", 0) or 0)

    db_path = resolve_session_db_path(root)
    if not os.path.exists(db_path):
        try:
            build_session_db(root, replace=False)
        except Exception:
            pass
    query_out: Dict[str, Any] = {"events": [], "traces": [], "reasoning": []}
    if os.path.exists(db_path):
        try:
            query_out = query_session_db(root, query=str(query or ""), limit=max(1, int(limit)))
        except Exception:
            query_out = {"events": [], "traces": [], "reasoning": []}

    issues = doctor.get("issues")
    issues = issues if isinstance(issues, list) else []
    severity = _severity_from_signals(
        quality_grade=quality_grade,
        doctor_grade=doctor_grade,
        parse_fail_count=parse_fail_count,
        timeout_count=timeout_count,
        error_issues=error_issues,
        warning_issues=warning_issues,
    )
    session_name = os.path.basename(root.rstrip("/")) or root
    title = (
        f"{session_name} incident ({severity}) "
        f"pf={parse_fail_count} timeout={timeout_count} traces={traces_count}"
    )
    report = {
        "version": 1,
        "generated_at_wall_s": time.time(),
        "session_dir": root,
        "session_name": session_name,
        "severity": severity,
        "title": title,
        "quality": {"grade": quality_grade or None, "score": quality_score},
        "doctor": {"grade": doctor_grade or None, "score": doctor_score},
        "summary": {
            "reasoning_count": reasoning_count,
            "trace_count": traces_count,
            "parse_fail_count": parse_fail_count,
            "timeout_count": timeout_count,
            "weak_label_count": weak_label_count,
            "error_issue_count": error_issues,
            "warning_issue_count": warning_issues,
        },
        "query": {"expr": str(query or ""), "limit": int(limit)},
        "top_traces": list(query_out.get("traces", []))[: max(1, int(limit))],
        "top_events": list(query_out.get("events", []))[: max(1, int(limit))],
        "issues": issues,
        "recommendations": _top_recommendations(doctor_report=doctor, improvement_report=improvement, limit=6),
    }
    return report


def render_incident_markdown(report: Mapping[str, Any]) -> str:
    title = str(report.get("title") or "Telemetry Incident")
    severity = str(report.get("severity") or "unknown")
    quality = report.get("quality")
    quality = quality if isinstance(quality, dict) else {}
    doctor = report.get("doctor")
    doctor = doctor if isinstance(doctor, dict) else {}
    summary = report.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    lines = [
        f"# {title}",
        "",
        f"- Severity: `{severity}`",
        f"- Quality: grade=`{quality.get('grade')}` score=`{quality.get('score')}`",
        f"- Doctor: grade=`{doctor.get('grade')}` score=`{doctor.get('score')}`",
        "",
        "## Summary",
        f"- parse_fail_count: {summary.get('parse_fail_count')}",
        f"- timeout_count: {summary.get('timeout_count')}",
        f"- trace_count: {summary.get('trace_count')}",
        f"- reasoning_count: {summary.get('reasoning_count')}",
        "",
        "## Issues",
    ]
    issues = report.get("issues")
    if isinstance(issues, list) and issues:
        for row in issues[:8]:
            if not isinstance(row, dict):
                continue
            lines.append(f"- [{row.get('severity')}] {row.get('code')}: {row.get('message')}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Recommendations")
    recs = report.get("recommendations")
    if isinstance(recs, list) and recs:
        for rec in recs[:8]:
            lines.append(f"- {rec}")
    else:
        lines.append("- none")
    return "\n".join(lines).strip() + "\n"


def write_incident_report(session_dir: str, report: Mapping[str, Any], *, filename: str = INCIDENT_REPORT_PATH) -> str:
    path = os.path.join(str(session_dir), str(filename))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(report), fh, separators=(",", ":"), ensure_ascii=True)
    return path


def write_incident_markdown(session_dir: str, report: Mapping[str, Any], *, filename: str = INCIDENT_MARKDOWN_PATH) -> str:
    path = os.path.join(str(session_dir), str(filename))
    text = render_incident_markdown(report)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def load_incident_report(path_or_dir: str) -> Optional[Dict[str, Any]]:
    path = str(path_or_dir)
    if os.path.isdir(path):
        path = os.path.join(path, INCIDENT_REPORT_PATH)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an incident report bundle from a telemetry session.")
    parser.add_argument("session_dir")
    parser.add_argument("--query", default="kind:trace severity:error status:parse_fail status:timeout")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output-json", default=INCIDENT_REPORT_PATH)
    parser.add_argument("--output-md", default=INCIDENT_MARKDOWN_PATH)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    session_dir = str(args.session_dir or "").strip()
    if not session_dir or not os.path.isdir(session_dir):
        raise SystemExit(f"session directory not found: {session_dir}")
    report = build_incident_report(session_dir, query=str(args.query or ""), limit=int(args.limit))
    json_path = write_incident_report(session_dir, report, filename=str(args.output_json or INCIDENT_REPORT_PATH))
    md_path = write_incident_markdown(session_dir, report, filename=str(args.output_md or INCIDENT_MARKDOWN_PATH))
    print(
        f"incident report: {json_path} markdown={md_path} severity={report.get('severity')} "
        f"issues={len(report.get('issues', []))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
