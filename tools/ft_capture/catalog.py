from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

from .schema import decision_to_dict, scenario_id_is_valid, validate_expected_decision

_ALLOWED_SPLITS = ("train", "val", "test")
_ALLOWED_LABEL_TEMPLATE_KEYS = {"expected_decision", "rationale_text", "notes"}


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    title: str
    description: str
    operator_setup_notes: str
    countdown_s: float
    duration_s: float
    sample_fps: float
    tags: List[str]
    label_template: Dict[str, Any]


@dataclass(frozen=True)
class ScenarioCatalog:
    version: int
    split_seed: str
    split_ratio: Dict[str, float]
    scenarios: Dict[str, ScenarioDefinition]
    source_path: str


def _fail(path: str, message: str) -> None:
    raise ValueError(f"catalog error at '{path}': {message}")


def _as_float(value: Any, path: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        _fail(path, f"expected number, got {type(value).__name__}")


def _as_tags(raw: Any, path: str) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        _fail(path, "expected list of strings")
    out: List[str] = []
    for i, item in enumerate(raw):
        token = str(item).strip().lower()
        if not token:
            _fail(f"{path}[{i}]", "empty tag")
        out.append(token)
    return out


def _normalize_split_ratio(raw: Any, path: str) -> Dict[str, float]:
    if not isinstance(raw, Mapping):
        _fail(path, "expected mapping")

    values: Dict[str, float] = {}
    for key in _ALLOWED_SPLITS:
        if key not in raw:
            _fail(path, f"missing key '{key}'")
        val = _as_float(raw[key], f"{path}.{key}")
        if val < 0.0:
            _fail(f"{path}.{key}", "must be >= 0")
        values[key] = val

    total = values["train"] + values["val"] + values["test"]
    if total <= 0.0:
        _fail(path, "split_ratio sum must be > 0")

    return {name: values[name] / total for name in _ALLOWED_SPLITS}


def _normalize_label_template(raw: Mapping[str, Any], path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    unknown = [str(key) for key in raw.keys() if str(key) not in _ALLOWED_LABEL_TEMPLATE_KEYS]
    if unknown:
        _fail(path, f"unknown keys: {sorted(unknown)}")

    if "expected_decision" in raw:
        decision_raw = raw.get("expected_decision")
        if not isinstance(decision_raw, Mapping):
            _fail(f"{path}.expected_decision", "expected object")
        decision = validate_expected_decision(decision_raw)
        out["expected_decision"] = decision_to_dict(decision)

    if "rationale_text" in raw:
        out["rationale_text"] = " ".join(str(raw.get("rationale_text") or "").split()).strip()

    if "notes" in raw:
        out["notes"] = str(raw.get("notes") or "").strip()

    return out


def load_catalog(path: str) -> ScenarioCatalog:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"scenario catalog not found: {path}")

    with path_obj.open("r", encoding="utf-8") as fh:
        decoded = yaml.safe_load(fh) or {}
    if not isinstance(decoded, Mapping):
        _fail("root", "expected mapping")

    version = int(decoded.get("version", 1))
    if version != 1:
        _fail("version", "only version=1 is supported")

    split_seed = str(decoded.get("split_seed", "pala-ft-v1")).strip() or "pala-ft-v1"
    split_ratio = _normalize_split_ratio(
        decoded.get("split_ratio", {"train": 0.8, "val": 0.1, "test": 0.1}),
        "split_ratio",
    )

    raw_scenarios = decoded.get("scenarios", [])
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        _fail("scenarios", "expected non-empty list")

    scenarios: Dict[str, ScenarioDefinition] = {}
    for idx, raw in enumerate(raw_scenarios):
        base_path = f"scenarios[{idx}]"
        if not isinstance(raw, Mapping):
            _fail(base_path, "expected mapping")

        sid = str(raw.get("id", "")).strip().lower()
        if not scenario_id_is_valid(sid):
            _fail(f"{base_path}.id", "must match ^[a-z0-9][a-z0-9_-]{2,63}$")
        if sid in scenarios:
            _fail(f"{base_path}.id", f"duplicate scenario id '{sid}'")

        title = str(raw.get("title", "")).strip()
        description = str(raw.get("description", "")).strip()
        if not title:
            _fail(f"{base_path}.title", "required")
        if not description:
            _fail(f"{base_path}.description", "required")

        notes = str(raw.get("operator_setup_notes", "")).strip()
        countdown_s = _as_float(raw.get("countdown_s", 5.0), f"{base_path}.countdown_s")
        duration_s = _as_float(raw.get("duration_s", 5.0), f"{base_path}.duration_s")
        sample_fps = _as_float(raw.get("sample_fps", 1.0), f"{base_path}.sample_fps")

        if countdown_s < 0.0:
            _fail(f"{base_path}.countdown_s", "must be >= 0")
        if duration_s <= 0.0:
            _fail(f"{base_path}.duration_s", "must be > 0")
        if sample_fps <= 0.0:
            _fail(f"{base_path}.sample_fps", "must be > 0")

        tags = _as_tags(raw.get("tags", []), f"{base_path}.tags")

        label_template_raw = raw.get("label_template", {})
        if not isinstance(label_template_raw, Mapping):
            _fail(f"{base_path}.label_template", "expected mapping")
        if "expected_action" in label_template_raw:
            _fail(
                f"{base_path}.label_template.expected_action",
                "deprecated; use label_template.expected_decision",
            )
        label_template = _normalize_label_template(label_template_raw, f"{base_path}.label_template")

        scenarios[sid] = ScenarioDefinition(
            scenario_id=sid,
            title=title,
            description=description,
            operator_setup_notes=notes,
            countdown_s=float(countdown_s),
            duration_s=float(duration_s),
            sample_fps=float(sample_fps),
            tags=tags,
            label_template=label_template,
        )

    return ScenarioCatalog(
        version=version,
        split_seed=split_seed,
        split_ratio=split_ratio,
        scenarios=scenarios,
        source_path=os.path.abspath(path_obj),
    )


def list_scenarios(catalog: ScenarioCatalog) -> List[ScenarioDefinition]:
    return [catalog.scenarios[key] for key in sorted(catalog.scenarios.keys())]


def resolve_scenario(catalog: ScenarioCatalog, scenario_id: str) -> ScenarioDefinition:
    key = str(scenario_id or "").strip().lower()
    if key not in catalog.scenarios:
        valid = ", ".join(sorted(catalog.scenarios.keys()))
        raise ValueError(f"unknown scenario '{scenario_id}'. valid: {valid}")
    return catalog.scenarios[key]


def assign_split(*, scenario_id: str, split_seed: str, split_ratio: Mapping[str, float]) -> str:
    token = f"{split_seed}:{scenario_id}".encode("utf-8", errors="strict")
    digest = hashlib.sha256(token).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    frac = value / float(2**64)

    train = float(split_ratio.get("train", 0.8))
    val = float(split_ratio.get("val", 0.1))
    test = float(split_ratio.get("test", 0.1))
    total = train + val + test
    if total <= 0.0:
        train, val, test = 0.8, 0.1, 0.1
        total = 1.0

    train /= total
    val /= total

    if frac < train:
        return "train"
    if frac < (train + val):
        return "val"
    return "test"
