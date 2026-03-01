from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schema import default_label_record, normalize_label_payload, utc_now_iso


DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TakeRecord:
    dataset_root: str
    session_id: str
    scenario_id: str
    take_id: str
    take_dir: str
    clip_path: str
    sampled_frames_dir: str
    raw_frames_dir: str
    take_manifest_path: str
    label_path: str
    created_at_utc: str
    label: Dict[str, Any]
    take_manifest: Dict[str, Any]



def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")
    os.replace(str(tmp), str(path))



def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        with path.open("r", encoding="utf-8") as fh:
            decoded = json.load(fh)
    except Exception:
        return dict(default)
    if not isinstance(decoded, dict):
        return dict(default)
    return decoded



def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")



def ensure_session_dir(
    *,
    out_root: str,
    session_id: str,
    catalog_path: str,
) -> Path:
    root = Path(out_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    session_dir = root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = session_dir / "session_manifest.json"
    if not manifest_path.exists():
        _write_json_atomic(
            manifest_path,
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "session_id": session_id,
                "created_at_utc": utc_now_iso(),
                "catalog_path": str(Path(catalog_path).expanduser()),
                "catalog_snapshot_path": "catalog_snapshot.yaml",
                "takes": [],
            },
        )
    return session_dir



def update_session_manifest_take(session_dir: Path, *, scenario_id: str, take_id: str, take_rel_dir: str) -> None:
    path = session_dir / "session_manifest.json"
    manifest = _read_json(
        path,
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "session_id": session_dir.name,
            "created_at_utc": utc_now_iso(),
            "catalog_path": "",
            "catalog_snapshot_path": "catalog_snapshot.yaml",
            "takes": [],
        },
    )
    takes = manifest.get("takes")
    if not isinstance(takes, list):
        takes = []

    row = {
        "scenario_id": scenario_id,
        "take_id": take_id,
        "take_rel_dir": take_rel_dir,
    }
    exists = any(
        isinstance(item, dict)
        and str(item.get("scenario_id")) == scenario_id
        and str(item.get("take_id")) == take_id
        for item in takes
    )
    if not exists:
        takes.append(row)
    manifest["takes"] = takes
    manifest["updated_at_utc"] = utc_now_iso()
    _write_json_atomic(path, manifest)



def write_catalog_snapshot(session_dir: Path, *, source_catalog_path: str) -> None:
    src = Path(source_catalog_path).expanduser()
    dst = session_dir / "catalog_snapshot.yaml"
    if dst.exists():
        return
    if not src.exists():
        return
    try:
        text = src.read_text(encoding="utf-8")
    except OSError:
        return
    dst.write_text(text, encoding="utf-8")



def next_take_id(session_dir: Path, *, scenario_id: str) -> str:
    scenario_dir = session_dir / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    max_idx = 0
    for child in scenario_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("take_"):
            continue
        try:
            idx = int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        max_idx = max(max_idx, idx)
    return f"take_{max_idx + 1:04d}"



def create_take_layout(session_dir: Path, *, scenario_id: str, take_id: str) -> Path:
    take_dir = session_dir / scenario_id / take_id
    (take_dir / "raw_frames").mkdir(parents=True, exist_ok=True)
    (take_dir / "frames_1fps").mkdir(parents=True, exist_ok=True)
    return take_dir



def write_initial_label(take_dir: Path, *, label_template: Dict[str, Any]) -> Dict[str, Any]:
    base = default_label_record()
    if isinstance(label_template, dict):
        if "expected_decision" in label_template:
            base["expected_decision"] = label_template.get("expected_decision")
        elif "expected_action" in label_template:
            raise ValueError("label_template.expected_action is deprecated; use label_template.expected_decision")
        if "rationale_text" in label_template:
            base["rationale_text"] = str(label_template.get("rationale_text") or "").strip()
        if "notes" in label_template:
            base["notes"] = str(label_template.get("notes") or "").strip()
    normalized = normalize_label_payload(base)
    _write_json_atomic(take_dir / "label.json", normalized)
    return normalized



def load_label(take_dir: Path) -> Dict[str, Any]:
    payload = _read_json(take_dir / "label.json", default_label_record())
    return normalize_label_payload(payload)



def save_label(take_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_label_payload(payload)
    normalized["updated_at_utc"] = utc_now_iso()
    _write_json_atomic(take_dir / "label.json", normalized)
    return normalized



def write_take_manifest(take_dir: Path, payload: Dict[str, Any]) -> None:
    out = dict(payload)
    out.setdefault("schema_version", DATASET_SCHEMA_VERSION)
    out.setdefault("updated_at_utc", utc_now_iso())
    _write_json_atomic(take_dir / "take_manifest.json", out)



def read_take_manifest(take_dir: Path) -> Dict[str, Any]:
    return _read_json(take_dir / "take_manifest.json", {})



def iter_take_dirs(dataset_root: str) -> Iterable[Path]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        return []

    out: List[Path] = []
    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir():
            continue
        if not (session_dir / "session_manifest.json").exists():
            continue
        for scenario_dir in sorted(session_dir.iterdir()):
            if not scenario_dir.is_dir() or scenario_dir.name.startswith("."):
                continue
            if scenario_dir.name in {"raw_frames", "frames_1fps"}:
                continue
            for take_dir in sorted(scenario_dir.iterdir()):
                if not take_dir.is_dir() or not take_dir.name.startswith("take_"):
                    continue
                if not (take_dir / "take_manifest.json").exists():
                    continue
                out.append(take_dir)
    return out



def load_take_records(dataset_root: str) -> List[TakeRecord]:
    root = Path(dataset_root).expanduser().resolve()
    records: List[TakeRecord] = []
    for take_dir in iter_take_dirs(str(root)):
        session_id = take_dir.parents[1].name
        scenario_id = take_dir.parents[0].name
        take_id = take_dir.name

        take_manifest = read_take_manifest(take_dir)
        label = load_label(take_dir)

        created_at = str(take_manifest.get("created_at_utc") or "")
        if not created_at:
            created_at = str(label.get("updated_at_utc") or "")
        if not created_at:
            created_at = "1970-01-01T00:00:00+00:00"

        clip_path = str((take_dir / "clip.mp4").resolve())
        sampled_dir = str((take_dir / "frames_1fps").resolve())
        raw_dir = str((take_dir / "raw_frames").resolve())

        records.append(
            TakeRecord(
                dataset_root=str(root),
                session_id=session_id,
                scenario_id=scenario_id,
                take_id=take_id,
                take_dir=str(take_dir.resolve()),
                clip_path=clip_path,
                sampled_frames_dir=sampled_dir,
                raw_frames_dir=raw_dir,
                take_manifest_path=str((take_dir / "take_manifest.json").resolve()),
                label_path=str((take_dir / "label.json").resolve()),
                created_at_utc=created_at,
                label=label,
                take_manifest=take_manifest,
            )
        )

    records.sort(key=lambda item: item.created_at_utc, reverse=True)
    return records
