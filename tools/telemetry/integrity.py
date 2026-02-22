from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional


INTEGRITY_REPORT_PATH = "integrity.json"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_artifact_candidates() -> List[str]:
    return [
        "events.jsonl",
        "index.json",
        "reasoning_index.json",
        "trace_index.json",
        "reasoning_trace_index.json",
        "session.db",
        "quality_report.json",
        "labels.weak.jsonl",
        "dataset_rows.jsonl",
        "annotations.jsonl",
    ]


def build_integrity_report(
    session_dir: str,
    *,
    files: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    root = str(session_dir)
    rel_files = list(files) if files is not None else _default_artifact_candidates()
    out_files: List[Dict[str, Any]] = []
    for rel_path in rel_files:
        rel = str(rel_path).strip().replace("\\", "/")
        if not rel:
            continue
        abs_path = os.path.join(root, rel)
        if not os.path.exists(abs_path):
            continue
        if not os.path.isfile(abs_path):
            continue
        stat = os.stat(abs_path)
        out_files.append(
            {
                "path": rel,
                "sha256": _sha256_file(abs_path),
                "size_bytes": int(stat.st_size),
                "mtime_wall_s": float(stat.st_mtime),
            }
        )
    return {
        "version": 1,
        "generated_at_wall_s": time.time(),
        "files": out_files,
        "file_count": len(out_files),
    }


def write_integrity_report(
    session_dir: str,
    report: Mapping[str, Any],
    *,
    filename: str = INTEGRITY_REPORT_PATH,
) -> str:
    path = os.path.join(str(session_dir), str(filename))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(report), fh, separators=(",", ":"), ensure_ascii=True)
    return path


def load_integrity_report(path_or_dir: str) -> Optional[Dict[str, Any]]:
    path = str(path_or_dir)
    if os.path.isdir(path):
        path = os.path.join(path, INTEGRITY_REPORT_PATH)
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


def verify_integrity_report(path_or_dir: str) -> Dict[str, Any]:
    root = str(path_or_dir)
    report = load_integrity_report(root)
    if report is None:
        return {
            "ok": False,
            "error": "integrity_report_missing",
            "checked_file_count": 0,
            "missing": [],
            "mismatch": [],
        }
    if os.path.isfile(root):
        root = os.path.dirname(root)
    files = report.get("files")
    if not isinstance(files, list):
        files = []
    missing: List[str] = []
    mismatch: List[str] = []
    checked = 0
    for item in files:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("path") or "").strip()
        expected = str(item.get("sha256") or "").strip().lower()
        if not rel or not expected:
            continue
        abs_path = os.path.join(root, rel)
        if not os.path.exists(abs_path):
            missing.append(rel)
            continue
        if not os.path.isfile(abs_path):
            missing.append(rel)
            continue
        checked += 1
        try:
            got = _sha256_file(abs_path)
        except Exception:
            mismatch.append(rel)
            continue
        if got.lower() != expected:
            mismatch.append(rel)
    return {
        "ok": (len(missing) == 0 and len(mismatch) == 0),
        "checked_file_count": checked,
        "missing": missing,
        "mismatch": mismatch,
        "file_count": len(files),
        "generated_at_wall_s": report.get("generated_at_wall_s"),
    }

