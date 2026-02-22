from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


IntentName = Literal[
    "idle_presence",
    "acknowledge_presence",
    "track_user",
    "scan_environment",
    "react_to_change",
    "reset_pose",
    "affirmation",
]
PrimitiveName = Literal["hold", "home", "breath", "glance", "nod", "orient_to_zone"]
StyleName = Literal["calm", "curious", "focused"]
RiskLevel = Literal["low", "medium", "high"]
ProposalSource = Literal["remote", "idle_engine"]


@dataclass(frozen=True)
class IntentProposal:
    intent: IntentName
    primitive: PrimitiveName
    command: Dict[str, Any]
    style: StyleName
    score: float
    confidence: float
    urgency: float
    risk: RiskLevel
    allow_interrupt: bool
    min_dwell_ms: Optional[int] = None
    max_duration_ms: Optional[int] = None
    evidence: List[str] = field(default_factory=list)
    rationale_short: str = ""


@dataclass(frozen=True)
class ProposerResponse:
    schema_version: str
    proposals: List[IntentProposal]
    notes_short: str = ""


@dataclass(frozen=True)
class ProposalCandidate:
    proposal: IntentProposal
    source: ProposalSource


@dataclass
class GovernedCandidate:
    candidate: ProposalCandidate
    valid: bool
    reject_reason: Optional[str]
    utility: float = 0.0


@dataclass(frozen=True)
class EnvSummary:
    scene: str
    events: str
    hypotheses: str
    summary_short: str
    delta_score: float
    features: Dict[str, Any] = field(default_factory=dict)


def clamp01(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def clamp_float(value: Any, *, lo: float, hi: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if numeric < lo:
        return lo
    if numeric > hi:
        return hi
    return numeric


def clamp_int(value: Any, *, lo: int, hi: int, default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    if numeric < lo:
        return lo
    if numeric > hi:
        return hi
    return numeric
