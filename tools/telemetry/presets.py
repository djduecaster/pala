from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Mapping, Optional


DEFAULT_PRESETS_PATH = os.path.join(os.path.dirname(__file__), "presets.yaml")


@dataclass(frozen=True)
class ViewerPreset:
    name: str
    description: str
    packs: List[str]
    panels: List[str]
    query: str = ""
    quality_gate: str = "off"
    index_mode: str = "auto"
    no_video: bool = False


def _default_presets() -> Dict[str, ViewerPreset]:
    presets = [
        ViewerPreset(
            name="baseline",
            description="Lean daily telemetry view with trace and reasoning focus.",
            packs=["reasoning_live"],
            panels=["summary", "trace_list", "reasoning_stream", "alignment", "quality", "video"],
            quality_gate="warn",
            index_mode="auto",
        ),
        ViewerPreset(
            name="headless-debug",
            description="No-video debugging for SSH-only triage.",
            packs=["behavior_v2_debug"],
            panels=["summary", "trace_list", "trace_detail", "reasoning_stream", "query", "quality", "logs", "transport"],
            quality_gate="warn",
            no_video=True,
        ),
        ViewerPreset(
            name="posttrain-curation",
            description="Failure-focused curation for dataset building/export.",
            packs=["behavior_v2_debug"],
            panels=["summary", "trace_list", "trace_detail", "alignment", "query", "quality", "annotations"],
            query="kind:joined severity:error|warning status:parse_fail|timeout sort:severity",
            quality_gate="strict",
            no_video=True,
        ),
        ViewerPreset(
            name="demo",
            description="Demo-friendly telemetry with video and compact reasoning.",
            packs=["demo_overview"],
            panels=["summary", "video", "trace_list", "reasoning_stream", "quality"],
            quality_gate="off",
        ),
    ]
    return {p.name: p for p in presets}


def _load_yaml_or_json(text: str) -> Optional[Any]:
    body = text.strip()
    if not body:
        return None
    try:
        return json.loads(body)
    except Exception:
        pass
    try:
        import yaml  # type: ignore

        return yaml.safe_load(body)
    except Exception:
        return None


def _coerce_preset(obj: Mapping[str, Any]) -> Optional[ViewerPreset]:
    name = str(obj.get("name") or "").strip()
    if not name:
        return None
    desc = str(obj.get("description") or "").strip() or "custom preset"
    packs = [str(v).strip() for v in (obj.get("packs") or []) if str(v).strip()]
    panels = [str(v).strip() for v in (obj.get("panels") or []) if str(v).strip()]
    if not packs:
        packs = ["reasoning_live"]
    if not panels:
        panels = ["summary", "trace_list", "trace_detail", "reasoning_stream", "query", "quality"]
    return ViewerPreset(
        name=name,
        description=desc,
        packs=packs,
        panels=panels,
        query=str(obj.get("query") or "").strip(),
        quality_gate=str(obj.get("quality_gate") or "off").strip() or "off",
        index_mode=str(obj.get("index_mode") or "auto").strip() or "auto",
        no_video=bool(obj.get("no_video")),
    )


def load_viewer_presets(path: str = "") -> Dict[str, ViewerPreset]:
    out = _default_presets()
    cfg_path = str(path).strip() or DEFAULT_PRESETS_PATH
    if not os.path.exists(cfg_path):
        return out
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            decoded = _load_yaml_or_json(fh.read())
    except Exception:
        return out
    items: List[Any] = []
    if isinstance(decoded, list):
        items = decoded
    elif isinstance(decoded, dict):
        raw = decoded.get("presets")
        if isinstance(raw, list):
            items = raw
    for item in items:
        if not isinstance(item, dict):
            continue
        preset = _coerce_preset(item)
        if preset is None:
            continue
        out[preset.name] = preset
    return out
