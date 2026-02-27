from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class SignalPack:
    name: str
    description: str
    sources: Tuple[str, ...]
    journal_filters: Tuple[str, ...] = ()


@dataclass
class ResolvedPacks:
    names: List[str]
    sources: Set[str]
    journal_filters: List[str]


PACKS: Dict[str, SignalPack] = {
    "reasoning_live": SignalPack(
        name="reasoning_live",
        description="Live orchestrator reasoning stream for on-call/demo monitoring.",
        sources=(
            "agent",
            "transport_stats",
            "timeline_log",
            "actions_log",
            "behavior_env_log",
            "behavior_planner_log",
            "behavior_reasoning_log",
            "video",
            "video_frame",
            "tegrastats",
        ),
        journal_filters=(r"(orchestrator|cosmos|reasoning|parse_fail|timeout|inflight|fallback)",),
    ),
    "reasoning_failures": SignalPack(
        name="reasoning_failures",
        description="Failure-centric reasoning and transport diagnostics.",
        sources=(
            "agent",
            "transport_stats",
            "timeline_log",
            "actions_log",
            "behavior_env_log",
            "behavior_planner_log",
            "behavior_reasoning_log",
            "journal",
        ),
        journal_filters=(r"(fail|error|invalid|timeout|fallback|traceback)",),
    ),
    "demo_overview": SignalPack(
        name="demo_overview",
        description="Balanced demo view with behavior context and system health.",
        sources=(
            "agent",
            "transport_stats",
            "timeline_log",
            "actions_log",
            "behavior_env_log",
            "behavior_planner_log",
            "video",
            "video_frame",
            "perception_log",
            "tegrastats",
        ),
        journal_filters=(r"(orchestrator|reasoning|camera|deepstream|error)",),
    ),
    "runtime_core": SignalPack(
        name="runtime_core",
        description="Primary runtime loop visibility and health.",
        sources=("agent", "transport_stats", "perception_log", "actions_log", "video", "video_frame", "tegrastats"),
        journal_filters=(r"(error|timeout|deadman|panic|traceback)",),
    ),
    "perception_debug": SignalPack(
        name="perception_debug",
        description="Perception and camera diagnostics.",
        sources=("perception_log", "video", "video_frame", "journal"),
        journal_filters=(r"(deepstream|nvinfer|gstreamer|gst|detector|camera|source_error)",),
    ),
    "planner_debug": SignalPack(
        name="planner_debug",
        description="Planner action outcomes and request lifecycle traces.",
        sources=("actions_log", "timeline_log", "journal"),
        journal_filters=(r"(planner|orchestrator|cosmos|reasoning|parse_fail|inflight)",),
    ),
    "memory_debug": SignalPack(
        name="memory_debug",
        description="Long-horizon memory and transcript distillation.",
        sources=("memory_log", "timeline_log", "actions_log"),
        journal_filters=(r"(memory|digest|summary_event|timeline)",),
    ),
    "hardware_safety": SignalPack(
        name="hardware_safety",
        description="Thermals, deadman-like symptoms, and hardware-adjacent warnings.",
        sources=("tegrastats", "journal", "actions_log", "agent", "transport_stats"),
        journal_filters=(r"(deadman|servo|hardware|overtemp|error|timeout)",),
    ),
    "cosmos_io": SignalPack(
        name="cosmos_io",
        description="Cosmos/planner I/O, response quality, and reasoning probes.",
        sources=("timeline_log", "memory_log", "journal", "actions_log", "behavior_env_log", "behavior_planner_log", "behavior_reasoning_log"),
        journal_filters=(r"(cosmos|reasoning|probe|response|parse|fallback)",),
    ),
    "behavior_v2_debug": SignalPack(
        name="behavior_v2_debug",
        description="BehaviorV2 environment/planner/reasoning traces with perception context.",
        sources=(
            "perception_log",
            "behavior_env_log",
            "behavior_planner_log",
            "behavior_reasoning_log",
            "timeline_log",
            "actions_log",
            "video_frame",
            "agent",
            "transport_stats",
        ),
        journal_filters=(r"(behavior|planner|env|reasoning|parse_fail|timeout)",),
    ),
}


def list_packs() -> List[SignalPack]:
    return [PACKS[name] for name in sorted(PACKS)]


def list_pack_names() -> List[str]:
    return [pack.name for pack in list_packs()]


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def resolve_packs(names: Sequence[str] | None) -> ResolvedPacks:
    selected = [str(x).strip() for x in (names or []) if str(x).strip()]
    if not selected:
        selected = ["runtime_core"]
    if any(name == "all" for name in selected):
        selected = list_pack_names()
    selected = _dedupe_keep_order(selected)

    missing = [name for name in selected if name not in PACKS]
    if missing:
        known = ", ".join(list_pack_names())
        raise ValueError(f"unknown pack(s): {', '.join(missing)}; known: {known}")

    sources: Set[str] = set()
    journal_filters: List[str] = []
    for name in selected:
        pack = PACKS[name]
        sources.update(pack.sources)
        journal_filters.extend(pack.journal_filters)
    return ResolvedPacks(
        names=selected,
        sources=sources,
        journal_filters=_dedupe_keep_order(journal_filters),
    )


def apply_pack_overrides(resolved: ResolvedPacks, overrides: Sequence[str] | None) -> ResolvedPacks:
    out = ResolvedPacks(
        names=list(resolved.names),
        sources=set(resolved.sources),
        journal_filters=list(resolved.journal_filters),
    )
    for raw in overrides or ():
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"invalid pack override '{text}'; expected key=value")
        key, value = text.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "include_sources":
            for src in [x.strip() for x in value.split(",") if x.strip()]:
                out.sources.add(src)
            continue
        if key == "exclude_sources":
            for src in [x.strip() for x in value.split(",") if x.strip()]:
                out.sources.discard(src)
            continue
        if key == "add_journal":
            if value:
                out.journal_filters.append(value)
            continue
        if key == "set_journal":
            out.journal_filters = [value] if value else []
            continue
        raise ValueError(f"unknown pack override key: '{key}'")
    out.journal_filters = _dedupe_keep_order(out.journal_filters)
    return out
