from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Mapping


ANNOTATIONS_PATH = "annotations.jsonl"


def annotation_key(row: Mapping[str, Any]) -> str:
    created_raw = row.get("created_at_wall_s")
    if isinstance(created_raw, (int, float)):
        created = f"{float(created_raw):.6f}"
    else:
        created = str(created_raw or "")
    return "\x1f".join(
        [
            created,
            str(row.get("tag") or ""),
            str(row.get("trace_id") or ""),
            str(row.get("req_id") if row.get("req_id") is not None else ""),
            str(row.get("event_index") if row.get("event_index") is not None else ""),
            str(row.get("note") or ""),
            str(row.get("phase") or ""),
            str(row.get("status") or ""),
            str(row.get("source") or ""),
        ]
    )


def append_annotation(
    session_dir: str,
    annotation: Mapping[str, Any],
    *,
    filename: str = ANNOTATIONS_PATH,
) -> Dict[str, Any]:
    root = str(session_dir).strip()
    if not root:
        raise ValueError("session_dir is required")
    os.makedirs(root, exist_ok=True)
    row = dict(annotation)
    row.setdefault("created_at_wall_s", time.time())
    row.setdefault("version", 1)
    path = os.path.join(root, str(filename))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True))
        fh.write("\n")
    return row


def load_annotations(
    session_dir: str,
    *,
    filename: str = ANNOTATIONS_PATH,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    path = os.path.join(str(session_dir), str(filename))
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            out.append(obj)
    lim = int(limit)
    if lim <= 0:
        return out
    return out[-lim:]
