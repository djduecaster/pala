from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .types import IntentProposal


@dataclass
class IdleEngineConfig:
    idle_after_s: float = 6.0
    glance_after_s: float = 10.0


class IdleEngine:
    """Deterministic low-amplitude baseline so the lamp never appears dead."""

    def __init__(self, config: IdleEngineConfig | None = None):
        self._cfg = config or IdleEngineConfig()

    def propose(self, *, no_commit_s: float, zone_hint: str, tick_index: int) -> List[IntentProposal]:
        proposals: List[IntentProposal] = []

        breath_score = 0.12 if no_commit_s < self._cfg.idle_after_s else 0.30
        proposals.append(
            IntentProposal(
                intent="idle_presence",
                primitive="breath",
                command={"amp_rad": 0.06, "period_s": 6.5, "rate_rad_s": 1.0},
                style="calm",
                score=breath_score,
                confidence=0.6,
                urgency=0.10,
                risk="low",
                allow_interrupt=True,
                evidence=["idle:heartbeat"],
                rationale_short="maintain subtle breathing presence",
            )
        )

        if no_commit_s >= self._cfg.glance_after_s:
            direction = "left" if (tick_index % 2 == 0) else "right"
            proposals.append(
                IntentProposal(
                    intent="scan_environment",
                    primitive="glance",
                    command={"direction": direction, "amp_rad": 0.24, "duration_s": 0.55, "rate_rad_s": 1.5},
                    style="curious",
                    score=0.36,
                    confidence=0.58,
                    urgency=0.18,
                    risk="low",
                    allow_interrupt=True,
                    evidence=["idle:micro_scan"],
                    rationale_short="perform low-amplitude micro-scan to stay socially alive",
                )
            )

            if zone_hint in {"left", "center", "right"}:
                proposals.append(
                    IntentProposal(
                        intent="track_user",
                        primitive="orient_to_zone",
                        command={"zone": zone_hint, "amp_rad": 0.22, "rate_rad_s": 1.2},
                        style="focused",
                        score=0.38,
                        confidence=0.55,
                        urgency=0.20,
                        risk="low",
                        allow_interrupt=True,
                        evidence=[f"idle:zone:{zone_hint}"],
                        rationale_short="gently re-center attention toward likely user zone",
                    )
                )

        return proposals
