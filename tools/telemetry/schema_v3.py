from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional


TELEMETRY_SCHEMA_VERSION_V3 = 3

SESSION_DB_PATH = "session.db"
QUALITY_REPORT_PATH = "quality_report.json"
WEAK_LABELS_PATH = "labels.weak.jsonl"
DATASET_ROWS_PATH = "dataset_rows.jsonl"
IMPROVEMENT_REPORT_PATH = "improvement_report.json"
DOCTOR_REPORT_PATH = "doctor_report.json"
INCIDENT_REPORT_PATH = "incident_report.json"


def v3_artifact_paths() -> Dict[str, str]:
    return {
        "session_db_path": SESSION_DB_PATH,
        "quality_report_path": QUALITY_REPORT_PATH,
        "weak_labels_path": WEAK_LABELS_PATH,
        "dataset_rows_path": DATASET_ROWS_PATH,
        "improvement_report_path": IMPROVEMENT_REPORT_PATH,
        "doctor_report_path": DOCTOR_REPORT_PATH,
        "incident_report_path": INCIDENT_REPORT_PATH,
    }


def upgrade_manifest_v3(
    manifest: Mapping[str, Any],
    *,
    index_summary: Optional[Mapping[str, Any]] = None,
    quality_report: Optional[Mapping[str, Any]] = None,
    improvement_report: Optional[Mapping[str, Any]] = None,
    doctor_report: Optional[Mapping[str, Any]] = None,
    incident_report: Optional[Mapping[str, Any]] = None,
    weak_label_count: Optional[int] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(manifest)
    out["schema_version"] = max(int(out.get("schema_version", 0) or 0), TELEMETRY_SCHEMA_VERSION_V3)
    out["telemetry_version"] = "v3"
    out["updated_at_wall_s"] = time.time()

    for key, value in v3_artifact_paths().items():
        out[key] = value

    if index_summary is not None:
        out["index_summary"] = dict(index_summary)
        if "event_count" in index_summary:
            out["indexed_event_count"] = int(index_summary["event_count"])
        if "trace_count" in index_summary:
            out["indexed_trace_count"] = int(index_summary["trace_count"])
        if "reasoning_count" in index_summary:
            out["indexed_reasoning_count"] = int(index_summary["reasoning_count"])

    if quality_report is not None:
        score = quality_report.get("score")
        grade = quality_report.get("grade")
        if isinstance(score, (int, float)):
            out["quality_score"] = round(float(score), 2)
        if isinstance(grade, str):
            out["quality_grade"] = grade

    if improvement_report is not None:
        recs = improvement_report.get("recommendations")
        if isinstance(recs, list):
            out["improvement_recommendation_count"] = len(recs)
        summary = improvement_report.get("summary")
        if isinstance(summary, dict):
            out["improvement_summary"] = {
                "parse_fail_count": summary.get("parse_fail_count"),
                "timeout_count": summary.get("timeout_count"),
                "slow_count": summary.get("slow_count"),
                "weak_label_count": summary.get("weak_label_count"),
            }

    if doctor_report is not None:
        readiness = doctor_report.get("readiness")
        if isinstance(readiness, dict):
            score = readiness.get("score")
            grade = readiness.get("grade")
            if isinstance(score, (int, float)):
                out["doctor_readiness_score"] = round(float(score), 2)
            if isinstance(grade, str):
                out["doctor_readiness_grade"] = grade
        summary = doctor_report.get("summary")
        if isinstance(summary, dict):
            out["doctor_summary"] = {
                "error_count": summary.get("error_count"),
                "warning_count": summary.get("warning_count"),
            }

    if incident_report is not None:
        severity = incident_report.get("severity")
        title = incident_report.get("title")
        if isinstance(severity, str):
            out["incident_severity"] = severity
        if isinstance(title, str):
            out["incident_title"] = title
        issues = incident_report.get("issues")
        if isinstance(issues, list):
            out["incident_issue_count"] = len(issues)

    if weak_label_count is not None:
        out["weak_label_count"] = max(0, int(weak_label_count))

    return out
