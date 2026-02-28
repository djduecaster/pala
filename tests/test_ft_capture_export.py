from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import numpy as np

from tools.ft_capture.catalog import load_catalog
from tools.ft_capture.export import export_openai_jsonl
from tools.ft_capture.storage import (
    create_take_layout,
    ensure_session_dir,
    save_label,
    write_take_manifest,
)


CATALOG_YAML = """
version: 1
split_seed: test-seed
split_ratio:
  train: 0.8
  val: 0.1
  test: 0.1
scenarios:
  - id: alpha_case
    title: Alpha
    description: Alpha scenario
    countdown_s: 5
    duration_s: 5
    sample_fps: 1
    tags: [alpha]
    label_template: {}
"""


def _write_dummy_jpeg(path: Path) -> None:
    arr = np.zeros((20, 30, 3), dtype=np.uint8)
    arr[:, :, 0] = 180
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG", quality=70)



def test_export_openai_jsonl_filters_labeled_usable(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(CATALOG_YAML, encoding="utf-8")
    catalog = load_catalog(str(catalog_path))

    dataset_root = tmp_path / "dataset"
    session_dir = ensure_session_dir(
        out_root=str(dataset_root),
        session_id="session_001",
        catalog_path=str(catalog_path),
    )

    # take_0001: usable+labeled -> included
    take1 = create_take_layout(session_dir, scenario_id="alpha_case", take_id="take_0001")
    _write_dummy_jpeg(take1 / "raw_frames" / "frame_000001.jpg")
    _write_dummy_jpeg(take1 / "frames_1fps" / "frame_0001.jpg")
    (take1 / "clip.mp4").write_bytes(b"dummy")
    write_take_manifest(
        take1,
        {
            "session_id": "session_001",
            "scenario_id": "alpha_case",
            "take_id": "take_0001",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
        },
    )
    save_label(
        take1,
        {
            "status": "labeled",
            "quality_flag": "usable",
            "expected_action": {
                "intent": "track_user",
                "primitive": "orient_to_zone",
                "command": {"zone": "center", "amp_rad": 0.2, "rate_rad_s": 1.4},
                "style": "focused",
                "confidence": 0.9,
            },
            "rationale_text": "user entered frame",
            "annotator": "tester",
            "notes": "",
        },
    )

    # take_0002: discarded -> excluded
    take2 = create_take_layout(session_dir, scenario_id="alpha_case", take_id="take_0002")
    _write_dummy_jpeg(take2 / "raw_frames" / "frame_000001.jpg")
    _write_dummy_jpeg(take2 / "frames_1fps" / "frame_0001.jpg")
    (take2 / "clip.mp4").write_bytes(b"dummy")
    write_take_manifest(
        take2,
        {
            "session_id": "session_001",
            "scenario_id": "alpha_case",
            "take_id": "take_0002",
            "created_at_utc": "2026-01-01T00:00:10+00:00",
        },
    )
    save_label(
        take2,
        {
            "status": "discarded",
            "quality_flag": "discard",
            "expected_action": None,
            "rationale_text": "bad framing",
            "annotator": "tester",
            "notes": "",
        },
    )

    out_dir = tmp_path / "export"
    result = export_openai_jsonl(dataset_root=str(dataset_root), catalog=catalog, out_dir=str(out_dir))
    assert result.total_rows == 1

    rows = [json.loads(line) for line in (out_dir / "dataset_openai.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["metadata"]["take_id"] == "take_0001"

    manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_rows"] == 1
