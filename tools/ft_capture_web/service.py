from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional

from tools.ft_capture.catalog import ScenarioCatalog, assign_split
from tools.ft_capture.schema import parse_expected_action_json
from tools.ft_capture.storage import TakeRecord, load_take_records, save_label


@dataclass(frozen=True)
class RecordView:
    token: str
    split: str
    record: TakeRecord
    clip_rel_url: str
    sampled_frame_urls: List[str]


@dataclass(frozen=True)
class RecordFilters:
    scenario: str = ""
    status: str = ""
    split: str = ""
    quality: str = ""



def make_token(record: TakeRecord) -> str:
    return f"{record.session_id}|{record.scenario_id}|{record.take_id}"



def parse_token(token: str) -> tuple[str, str, str]:
    parts = str(token or "").split("|")
    if len(parts) != 3:
        raise ValueError("invalid take token")
    return parts[0], parts[1], parts[2]



def _rel_url(path: str, *, dataset_root: str, mount_prefix: str) -> str:
    root = Path(dataset_root).expanduser().resolve()
    target = Path(path).expanduser().resolve()
    rel = target.relative_to(root)
    return f"{mount_prefix}/{rel.as_posix()}"



def _sampled_urls(record: TakeRecord, *, dataset_root: str, mount_prefix: str) -> List[str]:
    root = Path(record.sampled_frames_dir)
    if not root.exists():
        return []
    rows = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    rows.sort(key=lambda item: item.name)
    out: List[str] = []
    for item in rows:
        out.append(_rel_url(str(item), dataset_root=dataset_root, mount_prefix=mount_prefix))
    return out



def build_record_views(
    *,
    dataset_root: str,
    catalog: ScenarioCatalog,
    mount_prefix: str,
    filters: RecordFilters,
) -> List[RecordView]:
    all_records = load_take_records(dataset_root)
    out: List[RecordView] = []

    scenario_filter = str(filters.scenario or "").strip().lower()
    status_filter = str(filters.status or "").strip().lower()
    split_filter = str(filters.split or "").strip().lower()
    quality_filter = str(filters.quality or "").strip().lower()

    for record in all_records:
        split = assign_split(
            scenario_id=record.scenario_id,
            split_seed=catalog.split_seed,
            split_ratio=catalog.split_ratio,
        )
        status = str((record.label or {}).get("status", "")).strip().lower()
        quality = str((record.label or {}).get("quality_flag", "")).strip().lower()

        if scenario_filter and record.scenario_id != scenario_filter:
            continue
        if status_filter and status != status_filter:
            continue
        if split_filter and split != split_filter:
            continue
        if quality_filter and quality != quality_filter:
            continue

        out.append(
            RecordView(
                token=make_token(record),
                split=split,
                record=record,
                clip_rel_url=_rel_url(record.clip_path, dataset_root=dataset_root, mount_prefix=mount_prefix),
                sampled_frame_urls=_sampled_urls(record, dataset_root=dataset_root, mount_prefix=mount_prefix),
            )
        )

    return out



def find_record_view(views: List[RecordView], token: str) -> Optional[RecordView]:
    target = str(token or "").strip()
    if not target:
        return views[0] if views else None
    for row in views:
        if row.token == target:
            return row
    return views[0] if views else None



def update_label_from_form(
    *,
    record: TakeRecord,
    status: str,
    quality_flag: str,
    annotator: str,
    rationale_text: str,
    notes: str,
    expected_action_json: str,
) -> Dict[str, object]:
    payload = dict(record.label or {})
    payload["status"] = str(status or "").strip().lower() or "unlabeled"
    payload["quality_flag"] = str(quality_flag or "").strip().lower() or "usable"
    payload["annotator"] = str(annotator or "").strip()
    payload["rationale_text"] = str(rationale_text or "").strip()
    payload["notes"] = str(notes or "").strip()

    expected_action = None
    if payload["status"] == "labeled":
        expected_action = parse_expected_action_json(expected_action_json)
        payload["expected_action"] = {
            "intent": expected_action.intent,
            "primitive": expected_action.primitive,
            "command": dict(expected_action.command),
            "style": expected_action.style,
            "confidence": expected_action.confidence,
        }
    elif str(expected_action_json or "").strip():
        expected_action = parse_expected_action_json(expected_action_json)
        payload["expected_action"] = {
            "intent": expected_action.intent,
            "primitive": expected_action.primitive,
            "command": dict(expected_action.command),
            "style": expected_action.style,
            "confidence": expected_action.confidence,
        }
    else:
        payload["expected_action"] = None

    normalized = save_label(Path(record.take_dir), payload)
    return normalized



def scenario_choices(views: List[RecordView]) -> List[str]:
    out = sorted({item.record.scenario_id for item in views})
    return out



def record_by_token(dataset_root: str, token: str) -> Optional[TakeRecord]:
    try:
        session_id, scenario_id, take_id = parse_token(token)
    except ValueError:
        return None

    records = load_take_records(dataset_root)
    for record in records:
        if record.session_id == session_id and record.scenario_id == scenario_id and record.take_id == take_id:
            return record
    return None



def default_expected_action_json(record: TakeRecord) -> str:
    expected = (record.label or {}).get("expected_action")
    if not isinstance(expected, dict):
        return ""
    import json

    return json.dumps(expected, ensure_ascii=True, indent=2, sort_keys=True)
