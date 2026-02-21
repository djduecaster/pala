from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .insights import load_improvement_report
from .quality import load_quality_report
from .schema_v3 import DOCTOR_REPORT_PATH


DOCTOR_REPORT_VERSION = 1


def _read_json(path: str) -> Optional[Dict[str, Any]]:
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


def _iter_event_lines(events_path: str) -> Iterable[Tuple[int, str]]:
    with open(events_path, "r", encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh):
            yield idx, line.rstrip("\n")


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _grade_from_score(score: float, *, error_count: int) -> str:
    if error_count > 0 or score < 60.0:
        return "fail"
    if score < 80.0:
        return "warn"
    return "pass"


def _make_issue(severity: str, code: str, message: str) -> Dict[str, str]:
    return {"severity": str(severity), "code": str(code), "message": str(message)}


def _recommendations_from_issues(issues: Sequence[Mapping[str, Any]]) -> List[str]:
    recs: List[str] = []
    codes = {str(item.get("code") or "") for item in issues}
    if "events_missing" in codes:
        recs.append("Capture did not write events.jsonl; validate capture directory and write permissions.")
    if "manifest_event_count_mismatch" in codes:
        recs.append("Rebuild telemetry artifacts with `tools.telemetry.migrate_session` to realign manifest counts.")
    if "invalid_event_json" in codes:
        recs.append("Repair or discard malformed events; telemetry parsers expect one valid JSON object per line.")
    if "missing_v3_artifact" in codes:
        recs.append("Run migration to generate full V3 artifacts (sqlite, quality, improvement, labels, doctor).")
    if "frame_ref_missing" in codes:
        recs.append("Re-capture frames or repair frame_ref links under session frames/ directory.")
    if "quality_fail" in codes:
        recs.append("Address major failure drivers before using this session for post-training.")
    if "low_label_density" in codes:
        recs.append("Increase capture duration or targeted failure scenarios to improve weak-label yield.")
    if not recs:
        recs.append("Session integrity looks healthy. Continue scaling scenario coverage and regression monitoring.")
    return recs


def build_doctor_report(
    session_dir: str,
    *,
    manifest: Optional[Mapping[str, Any]] = None,
    quality_report: Optional[Mapping[str, Any]] = None,
    improvement_report: Optional[Mapping[str, Any]] = None,
    index_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    root = str(session_dir)
    manifest_obj = dict(manifest or _read_json(os.path.join(root, "manifest.json")) or {})
    quality_obj = dict(quality_report or load_quality_report(root) or {})
    improvement_obj = dict(improvement_report or load_improvement_report(root) or {})

    required_files = [
        "events.jsonl",
        "manifest.json",
        "reasoning_index.json",
        "trace_index.json",
    ]
    optional_v3 = [
        "session.db",
        "quality_report.json",
        "improvement_report.json",
        "labels.weak.jsonl",
    ]
    present_required: Dict[str, bool] = {}
    present_v3: Dict[str, bool] = {}
    for name in required_files:
        present_required[name] = os.path.exists(os.path.join(root, name))
    for name in optional_v3:
        present_v3[name] = os.path.exists(os.path.join(root, name))

    issues: List[Dict[str, str]] = []
    if not present_required.get("events.jsonl", False):
        issues.append(_make_issue("error", "events_missing", "events.jsonl is missing"))

    event_count = 0
    invalid_json_count = 0
    frame_ref_missing_count = 0
    ts_regression_count = 0
    last_ts: Optional[float] = None
    source_counts: Dict[str, int] = {}

    events_path = os.path.join(root, "events.jsonl")
    if os.path.exists(events_path):
        for _, line in _iter_event_lines(events_path):
            if not line:
                continue
            event_count += 1
            try:
                msg = json.loads(line)
            except Exception:
                invalid_json_count += 1
                continue
            if not isinstance(msg, dict):
                invalid_json_count += 1
                continue
            source = str(msg.get("source") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            ts = _to_float(msg.get("ts_wall_s"))
            if ts is not None:
                if last_ts is not None and ts + 1e-6 < last_ts:
                    ts_regression_count += 1
                last_ts = ts
            payload = msg.get("payload")
            if source == "video_frame" and isinstance(payload, dict):
                frame_ref = payload.get("frame_ref")
                if isinstance(frame_ref, str) and frame_ref.strip():
                    frame_path = os.path.normpath(os.path.join(root, frame_ref))
                    root_abs = os.path.abspath(root)
                    frame_abs = os.path.abspath(frame_path)
                    if (not frame_abs.startswith(root_abs)) or (not os.path.exists(frame_abs)):
                        frame_ref_missing_count += 1

    if invalid_json_count > 0:
        issues.append(_make_issue("error", "invalid_event_json", f"{invalid_json_count} malformed event line(s) found"))
    if ts_regression_count > 0:
        issues.append(_make_issue("warning", "timestamp_regression", f"{ts_regression_count} event timestamp regressions found"))
    if frame_ref_missing_count > 0:
        issues.append(_make_issue("error", "frame_ref_missing", f"{frame_ref_missing_count} frame_ref entries point to missing files"))

    manifest_event_count = int(manifest_obj.get("event_count", 0) or 0)
    if manifest_event_count > 0 and event_count > 0 and manifest_event_count != event_count:
        issues.append(
            _make_issue(
                "warning",
                "manifest_event_count_mismatch",
                f"manifest event_count={manifest_event_count} differs from events.jsonl lines={event_count}",
            )
        )

    for name, exists in present_v3.items():
        if not exists:
            issues.append(_make_issue("warning", "missing_v3_artifact", f"missing {name}"))

    quality_grade = str(quality_obj.get("grade") or "").lower()
    if quality_grade == "fail":
        issues.append(_make_issue("warning", "quality_fail", "quality_report grade is fail"))

    summary = improvement_obj.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    weak_label_count = int(summary.get("weak_label_count", 0) or 0)
    reasoning_count = int(summary.get("reasoning_count", 0) or 0)
    if reasoning_count >= 20 and weak_label_count < max(5, int(0.05 * reasoning_count)):
        issues.append(_make_issue("warning", "low_label_density", "weak label yield is low relative to reasoning events"))

    idx_event_count = int((index_summary or {}).get("event_count", 0) or 0)
    if idx_event_count <= 0:
        idx_event_count = int(manifest_obj.get("indexed_event_count", 0) or 0)

    severity_weights = {"error": 22.0, "warning": 7.0, "info": 1.0}
    score = 100.0
    for item in issues:
        sev = str(item.get("severity") or "warning").lower()
        score -= severity_weights.get(sev, 7.0)
    score = max(0.0, min(100.0, score))
    error_count = sum(1 for item in issues if str(item.get("severity")) == "error")
    warning_count = sum(1 for item in issues if str(item.get("severity")) == "warning")
    grade = _grade_from_score(score, error_count=error_count)

    report = {
        "version": DOCTOR_REPORT_VERSION,
        "generated_at_wall_s": time.time(),
        "session_dir": root,
        "readiness": {
            "score": round(score, 2),
            "grade": grade,
        },
        "checks": {
            "required_files": present_required,
            "v3_artifacts": present_v3,
            "event_count_manifest": manifest_event_count,
            "event_count_lines": int(event_count),
            "event_count_indexed": int(idx_event_count),
            "invalid_json_count": int(invalid_json_count),
            "timestamp_regression_count": int(ts_regression_count),
            "frame_ref_missing_count": int(frame_ref_missing_count),
        },
        "metrics": {
            "source_counts": source_counts,
            "reasoning_count": int(reasoning_count),
            "weak_label_count": int(weak_label_count),
            "quality_grade": quality_grade or None,
        },
        "issues": issues,
        "summary": {
            "error_count": int(error_count),
            "warning_count": int(warning_count),
        },
        "recommendations": _recommendations_from_issues(issues),
    }
    return report


def write_doctor_report(session_dir: str, report: Mapping[str, Any], *, filename: str = DOCTOR_REPORT_PATH) -> str:
    path = os.path.join(str(session_dir), str(filename))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(report), fh, separators=(",", ":"), ensure_ascii=True)
    return path


def load_doctor_report(path_or_dir: str) -> Optional[Dict[str, Any]]:
    path = str(path_or_dir)
    if os.path.isdir(path):
        path = os.path.join(path, DOCTOR_REPORT_PATH)
    return _read_json(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run telemetry doctor checks on a session bundle.")
    parser.add_argument("session_dir", help="Session directory path.")
    parser.add_argument("--output", default=DOCTOR_REPORT_PATH, help="Output JSON report filename/path.")
    parser.add_argument("--gate", choices=["off", "warn", "strict"], default="warn", help="Failure gate level.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    session_dir = str(args.session_dir or "").strip()
    if not session_dir or not os.path.isdir(session_dir):
        raise SystemExit(f"session directory not found: {session_dir}")
    report = build_doctor_report(session_dir)
    output = str(args.output or DOCTOR_REPORT_PATH)
    if os.path.isabs(output):
        target_dir = os.path.dirname(output) or "."
        os.makedirs(target_dir, exist_ok=True)
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, separators=(",", ":"), ensure_ascii=True)
        out_path = output
    else:
        out_path = write_doctor_report(session_dir, report, filename=output)

    readiness = report.get("readiness") if isinstance(report, dict) else {}
    readiness = readiness if isinstance(readiness, dict) else {}
    grade = str(readiness.get("grade") or "unknown")
    score = readiness.get("score")
    summary = report.get("summary") if isinstance(report, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    print(
        f"doctor report: {out_path} grade={grade} score={score} "
        f"errors={summary.get('error_count', 0)} warnings={summary.get('warning_count', 0)}"
    )

    gate = str(args.gate or "warn")
    if gate == "off":
        return 0
    if gate == "warn":
        return 2 if grade == "fail" else 0
    return 0 if grade == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
