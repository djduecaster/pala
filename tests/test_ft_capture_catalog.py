from __future__ import annotations

from pathlib import Path

from tools.ft_capture.catalog import assign_split, load_catalog


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
    operator_setup_notes: notes
    countdown_s: 5
    duration_s: 5
    sample_fps: 1
    tags: [alpha]
    label_template: {}
  - id: beta_case
    title: Beta
    description: Beta scenario
    countdown_s: 3
    duration_s: 4
    sample_fps: 2
    tags: [beta]
    label_template: {}
"""


def test_load_catalog_and_split_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(CATALOG_YAML, encoding="utf-8")

    catalog = load_catalog(str(path))
    assert catalog.version == 1
    assert set(catalog.scenarios.keys()) == {"alpha_case", "beta_case"}

    first = assign_split(scenario_id="alpha_case", split_seed=catalog.split_seed, split_ratio=catalog.split_ratio)
    second = assign_split(scenario_id="alpha_case", split_seed=catalog.split_seed, split_ratio=catalog.split_ratio)
    assert first == second
    assert first in {"train", "val", "test"}
