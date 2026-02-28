from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List

from .catalog import ScenarioCatalog, assign_split
from .storage import TakeRecord, load_take_records


@dataclass(frozen=True)
class ExportResult:
    out_dir: str
    total_rows: int
    split_counts: Dict[str, int]
    scenario_counts: Dict[str, int]
    openai_jsonl_path: str
    index_jsonl_path: str
    manifest_path: str



def _iter_sampled_frames(sampled_frames_dir: str) -> List[str]:
    root = Path(sampled_frames_dir)
    if not root.exists():
        return []
    out = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    out.sort(key=lambda item: item.name)
    return [str(path.resolve()) for path in out]



def _is_usable_labeled_take(record: TakeRecord) -> bool:
    label = record.label or {}
    return (
        str(label.get("status", "")).strip().lower() == "labeled"
        and str(label.get("quality_flag", "")).strip().lower() == "usable"
        and isinstance(label.get("expected_action"), dict)
    )



def _build_user_text(record: TakeRecord, sampled_frames: List[str], split: str) -> str:
    label = record.label or {}
    manifest = record.take_manifest or {}
    scenario_desc = str(manifest.get("scenario_description", "")).strip()
    scenario_title = str(manifest.get("scenario_title", record.scenario_id)).strip()
    tags = manifest.get("scenario_tags")
    tags_csv = ", ".join(str(x) for x in tags) if isinstance(tags, list) else ""
    rationale = str(label.get("rationale_text", "")).strip()
    return (
        "Scenario clip context:\n"
        f"- scenario_id: {record.scenario_id}\n"
        f"- scenario_title: {scenario_title}\n"
        f"- scenario_description: {scenario_desc}\n"
        f"- tags: {tags_csv}\n"
        f"- split: {split}\n"
        f"- sampled_frames: {len(sampled_frames)}\n"
        f"- operator_rationale: {rationale}\n"
        "Use the visual evidence to produce the canonical expected action JSON."
    )



def _openai_row(record: TakeRecord, *, split: str) -> Dict[str, Any]:
    sampled_frames = _iter_sampled_frames(record.sampled_frames_dir)
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": _build_user_text(record, sampled_frames, split)}]
    for frame_path in sampled_frames:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"file://{frame_path}"},
            }
        )

    expected_action = dict(record.label.get("expected_action") or {})
    assistant_payload = json.dumps(expected_action, ensure_ascii=True, separators=(",", ":"))

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the PALA intent proposer ground-truth model. "
                    "Return exactly one canonical action JSON with intent, primitive, command, style, confidence."
                ),
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_payload},
        ],
        "metadata": {
            "session_id": record.session_id,
            "scenario_id": record.scenario_id,
            "take_id": record.take_id,
            "split": split,
            "clip_path": str(Path(record.clip_path).resolve()),
            "take_manifest_path": str(Path(record.take_manifest_path).resolve()),
            "label_path": str(Path(record.label_path).resolve()),
            "sampled_frame_count": len(sampled_frames),
            "rationale_text": str(record.label.get("rationale_text", "")),
        },
    }



def _index_row(record: TakeRecord, *, split: str) -> Dict[str, Any]:
    return {
        "session_id": record.session_id,
        "scenario_id": record.scenario_id,
        "take_id": record.take_id,
        "split": split,
        "created_at_utc": record.created_at_utc,
        "clip_path": record.clip_path,
        "sampled_frames_dir": record.sampled_frames_dir,
        "label_path": record.label_path,
        "expected_action": record.label.get("expected_action"),
        "rationale_text": record.label.get("rationale_text"),
    }



def export_openai_jsonl(
    *,
    dataset_root: str,
    catalog: ScenarioCatalog,
    out_dir: str,
) -> ExportResult:
    records = load_take_records(dataset_root)
    usable = [item for item in records if _is_usable_labeled_take(item)]

    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    openai_path = out_root / "dataset_openai.jsonl"
    index_path = out_root / "dataset_index.jsonl"
    manifest_path = out_root / "export_manifest.json"

    split_counts = {"train": 0, "val": 0, "test": 0}
    scenario_counts: Dict[str, int] = {}

    with openai_path.open("w", encoding="utf-8") as openai_fh, index_path.open("w", encoding="utf-8") as index_fh:
        for record in usable:
            split = assign_split(
                scenario_id=record.scenario_id,
                split_seed=catalog.split_seed,
                split_ratio=catalog.split_ratio,
            )
            split_counts[split] = split_counts.get(split, 0) + 1
            scenario_counts[record.scenario_id] = scenario_counts.get(record.scenario_id, 0) + 1

            openai_row = _openai_row(record, split=split)
            index_row = _index_row(record, split=split)

            openai_fh.write(json.dumps(openai_row, ensure_ascii=True, separators=(",", ":")))
            openai_fh.write("\n")
            index_fh.write(json.dumps(index_row, ensure_ascii=True, separators=(",", ":")))
            index_fh.write("\n")

    manifest = {
        "schema_version": 1,
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "catalog_path": catalog.source_path,
        "split_seed": catalog.split_seed,
        "split_ratio": dict(catalog.split_ratio),
        "total_rows": sum(split_counts.values()),
        "split_counts": split_counts,
        "scenario_counts": scenario_counts,
        "openai_jsonl": openai_path.name,
        "index_jsonl": index_path.name,
        "notes": [
            "image_url entries use file:// absolute paths and may require path remapping before remote upload",
            "rows include only label.status=labeled and label.quality_flag=usable",
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    return ExportResult(
        out_dir=str(out_root),
        total_rows=sum(split_counts.values()),
        split_counts=split_counts,
        scenario_counts=scenario_counts,
        openai_jsonl_path=str(openai_path),
        index_jsonl_path=str(index_path),
        manifest_path=str(manifest_path),
    )
