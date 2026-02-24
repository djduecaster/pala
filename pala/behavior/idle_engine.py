from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .decision_types import BehaviorMode
from .types import IntentProposal


@dataclass
class IdleEngineConfig:
    idle_after_s: float = 6.0
    glance_after_s: float = 8.0


class IdleEngine:
    """Deterministic baseline behaviors that keep motion alive under low confidence."""

    def __init__(self, config: IdleEngineConfig | None = None):
        self._cfg = config or IdleEngineConfig()

    def propose(
        self,
        *,
        mode: BehaviorMode,
        no_commit_s: float,
        zone_hint: str | None,
        tick_index: int,
        signals: Dict[str, float | bool | None],
    ) -> List[IntentProposal]:
        proposals: List[IntentProposal] = []

        if mode == BehaviorMode.RECOVER_RESET:
            proposals.append(
                IntentProposal(
                    intent="reset_pose",
                    primitive="home",
                    command={"rate_rad_s": 1.2},
                    style="calm",
                    score=0.62,
                    confidence=0.70,
                    urgency=0.55,
                    risk="low",
                    allow_interrupt=False,
                    min_dwell_ms=800,
                    evidence=["idle:recover"],
                    rationale_short="recover to safe neutral pose",
                )
            )
            return proposals

        breath_score = 0.18 if no_commit_s < self._cfg.idle_after_s else 0.30
        proposals.append(
            IntentProposal(
                intent="idle_presence",
                primitive="breath",
                command={"amp_rad": 0.06, "period_s": 6.5, "rate_rad_s": 1.0},
                style="calm",
                score=breath_score,
                confidence=0.62,
                urgency=0.10,
                risk="low",
                allow_interrupt=True,
                evidence=["idle:heartbeat"],
                rationale_short="maintain subtle breathing presence",
            )
        )

        if no_commit_s >= self._cfg.glance_after_s or mode in {BehaviorMode.SCAN_EXPLORE, BehaviorMode.ACKNOWLEDGE}:
            direction = "left" if (tick_index % 2 == 0) else "right"
            proposals.append(
                IntentProposal(
                    intent="scan_environment",
                    primitive="glance",
                    command={"direction": direction, "amp_rad": 0.24, "duration_s": 0.55, "rate_rad_s": 1.5},
                    style="curious",
                    score=0.42,
                    confidence=0.60,
                    urgency=0.24,
                    risk="low",
                    allow_interrupt=True,
                    evidence=["idle:micro_scan"],
                    rationale_short="perform micro-scan for scene changes",
                )
            )

        if zone_hint in {"left", "center", "right"} and mode in {BehaviorMode.ENGAGE_TRACK, BehaviorMode.ACKNOWLEDGE}:
            proposals.append(
                IntentProposal(
                    intent="track_user",
                    primitive="orient_to_zone",
                    command={"zone": zone_hint, "amp_rad": 0.22, "rate_rad_s": 1.3},
                    style="focused",
                    score=0.46,
                    confidence=0.62,
                    urgency=0.30,
                    risk="low",
                    allow_interrupt=True,
                    evidence=[f"idle:zone:{zone_hint}"],
                    rationale_short="gently orient toward likely user zone",
                )
            )

        # Always keep a deterministic safe fallback candidate.
        proposals.append(
            IntentProposal(
                intent="idle_presence",
                primitive="hold",
                command={},
                style="calm",
                score=0.16,
                confidence=0.70,
                urgency=0.05,
                risk="low",
                allow_interrupt=True,
                evidence=["idle:fallback"],
                rationale_short="safe hold fallback",
            )
        )

        return proposals
