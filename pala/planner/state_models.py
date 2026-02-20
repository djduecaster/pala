from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Optional


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


@dataclass
class SceneSummary:
    ts_wall_s: float
    ts_mono_s: float
    scene_state: str
    person_present: bool
    zone_hint: Optional[str]
    notable_changes: list[str]
    activity_hint: Optional[str]
    uncertainty: list[str]
    confidence: float
    rationale: str
    source: str = "remote"

    def to_payload(self) -> dict[str, Any]:
        return {
            "scene_state": self.scene_state,
            "person_present": self.person_present,
            "zone_hint": self.zone_hint,
            "notable_changes": list(self.notable_changes),
            "activity_hint": self.activity_hint,
            "uncertainty": list(self.uncertainty),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "source": self.source,
        }
