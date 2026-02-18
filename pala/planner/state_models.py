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
class ObservationPacket:
    timestamp_monotonic_s: float
    person_present: bool
    zone_hint: Optional[str]
    primary_person_conf: Optional[float]
    frame_age_ms: Optional[float]
    activity_hint: Optional[str]
    uncertainty_flags: list[str] = field(default_factory=list)
    zone_stable_s: float = 0.0
    zone_transitions_recent: int = 0
    control_active_primitive: Optional[str] = None
    control_active_age_s: Optional[float] = None


@dataclass
class InteractionBelief:
    timestamp_monotonic_s: float
    state: str
    confidence: float
    last_seen_zone: Optional[str]
    person_last_seen_s: Optional[float]
    reason: str
    uncertainty_flags: list[str] = field(default_factory=list)


@dataclass
class OrchestratorDecision:
    target_state: str
    intent: str
    style: str
    primitive_hint: Optional[str]
    target_zone: Optional[str]
    allow_interrupt: bool
    urgency: str
    confidence: float
    rationale: str
    source: str = "local"
