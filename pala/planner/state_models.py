from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SceneSummary:
    timestamp_monotonic_s: float
    person_present: bool
    zone_hint: Optional[str]
    primary_person_conf: Optional[float]
    activity_hint: Optional[str]
    uncertainty_flags: list[str] = field(default_factory=list)
    frame_age_ms: Optional[float] = None


@dataclass
class SessionMemory:
    interaction_state: str
    task_hypothesis: Optional[str]
    last_transition_s: float
    staleness_ms: float
    recent_intents: list[str] = field(default_factory=list)


@dataclass
class OrchestratorDecision:
    intent: str
    style: str
    primitive_hint: Optional[str]
    target_zone: Optional[str]
    confidence: float
    rationale: str
    source: str = "local"
