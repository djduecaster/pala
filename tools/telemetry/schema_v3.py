from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional


TELEMETRY_SCHEMA_VERSION_V3 = 3

SESSION_DB_PATH = "session.db"
QUALITY_REPORT_PATH = "quality_report.json"
WEAK_LABELS_PATH = "labels.weak.jsonl"
DATASET_ROWS_PATH = "dataset_rows.jsonl"


def v3_artifact_paths() -> Dict[str, str]:
    return {
        "session_db_path": SESSION_DB_PATH,
        "quality_report_path": QUALITY_REPORT_PATH,
        "weak_labels_path": WEAK_LABELS_PATH,
        "dataset_rows_path": DATASET_ROWS_PATH,
    }


def upgrade_manifest_v3(
    manifest: Mapping[str, Any],
    *,
    index_summary: Optional[Mapping[str, Any]] = None,
    quality_report: Optional[Mapping[str, Any]] = None,
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

    if weak_label_count is not None:
        out["weak_label_count"] = max(0, int(weak_label_count))

    return out
