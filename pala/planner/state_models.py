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
class OrchestratorDecision:
    intent: str
    style: str
    primitive_hint: Optional[str]
    target_zone: Optional[str]
    confidence: float
    rationale: str
    source: str = "local"
