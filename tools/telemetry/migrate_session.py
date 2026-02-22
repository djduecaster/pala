from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from .dataset_export import export_dataset_rows
from .integrity import build_integrity_report, verify_integrity_report, write_integrity_report
from .labels import derive_weak_labels, write_labels_jsonl
from .quality import build_quality_report, write_quality_report
from .schema_v3 import (
    INTEGRITY_REPORT_PATH,
    QUALITY_REPORT_PATH,
    REASONING_TRACE_INDEX_PATH,
    SESSION_DB_PATH,
    WEAK_LABELS_PATH,
    upgrade_manifest_v3,
)
from .storage_sqlite import build_session_db
from .trace_join import build_reasoning_trace_index, write_reasoning_trace_index
from .trace_graph import load_trace_index, resolve_trace_index_path


def _load_manifest(session_dir: str) -> Dict[str, Any]:
    path = os.path.join(session_dir, "manifest.json")
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
    return [row for row in events if isinstance(row, dict)]


def _save_manifest(session_dir: str, manifest: Dict[str, Any]) -> None:
    path = os.path.join(session_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, separators=(",", ":"), ensure_ascii=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upgrade telemetry capture bundle to V3 artifacts.")
    parser.add_argument("session_dir", help="Capture bundle directory containing events.jsonl/manifest.json.")
    parser.add_argument("--no-sqlite", action="store_true", help="Skip sqlite index generation.")
    parser.add_argument("--export-dataset", action="store_true", help="Write dataset_rows.jsonl from weak labels.")
    parser.add_argument("--include-unlabeled", action="store_true", help="Include unlabeled rows in dataset export.")
    parser.add_argument("--label-min-confidence", type=float, default=0.6)
    parser.add_argument("--dataset-profile", choices=["fast", "strict", "hard_cases"], default="fast")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    session_dir = str(args.session_dir).strip()
    if not session_dir or not os.path.isdir(session_dir):
        raise SystemExit(f"session directory not found: {session_dir}")

    manifest = _load_manifest(session_dir)
    reasoning_index = _load_reasoning_index(session_dir)
    trace_index_path = resolve_trace_index_path(session_dir, manifest if manifest else None)
    traces = load_trace_index(trace_index_path) if os.path.exists(trace_index_path) else []

    index_summary: Dict[str, Any] = {}
    if not args.no_sqlite:
        index_summary = build_session_db(session_dir, db_path=os.path.join(session_dir, SESSION_DB_PATH), replace=True)
        print(f"indexed sqlite: {index_summary.get('db_path')} events={index_summary.get('event_count')}")

    labels = derive_weak_labels(reasoning_index=reasoning_index, traces=traces)
    labels_path = os.path.join(session_dir, WEAK_LABELS_PATH)
    weak_label_count = write_labels_jsonl(labels_path, labels)
    print(f"weak labels: {labels_path} rows={weak_label_count}")

    joined_index = build_reasoning_trace_index(session_dir, traces=traces, manifest=manifest)
    joined_path = write_reasoning_trace_index(session_dir, joined_index, filename=REASONING_TRACE_INDEX_PATH)
    print(f"reasoning trace index: {joined_path} rows={joined_index.get('row_count', 0)}")

    source_counts = index_summary.get("source_counts", {})
    if (not source_counts) and isinstance(manifest.get("source_counts"), dict):
        source_counts = dict(manifest.get("source_counts"))
    event_count = int(index_summary.get("event_count", manifest.get("event_count", 0) if manifest else 0))
    quality = build_quality_report(
        event_count=event_count,
        source_counts=source_counts if isinstance(source_counts, dict) else {},
        reasoning_index=reasoning_index,
        traces=traces,
    )
    quality_path = write_quality_report(session_dir, quality, filename=QUALITY_REPORT_PATH)
    print(f"quality report: {quality_path} grade={quality.get('grade')} score={quality.get('score')}")

    if args.export_dataset:
        result = export_dataset_rows(
            session_dir,
            include_unlabeled=bool(args.include_unlabeled),
            min_label_confidence=float(args.label_min_confidence),
            profile=str(args.dataset_profile),
        )
        print(f"dataset rows: {result.get('output_path')} rows={result.get('row_count')}")

    upgraded = upgrade_manifest_v3(
        manifest,
        index_summary=index_summary if index_summary else None,
        quality_report=quality,
        weak_label_count=weak_label_count,
    )
    try:
        report = build_integrity_report(session_dir)
        write_integrity_report(session_dir, report, filename=INTEGRITY_REPORT_PATH)
        verify = verify_integrity_report(session_dir)
        upgraded["integrity_ok"] = bool(verify.get("ok"))
        upgraded["integrity_checked_file_count"] = int(verify.get("checked_file_count", 0))
    except Exception as exc:
        upgraded.setdefault("v3_artifact_errors", []).append(f"integrity: {exc!r}")
    _save_manifest(session_dir, upgraded)
    print("manifest upgraded to telemetry v3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
