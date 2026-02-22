from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .types import GovernedCandidate, ProposalCandidate


@dataclass
class GovernorConfig:
    block_high_risk: bool = True
    block_home_unless_reset_pose: bool = True


class Governor:
    """Deterministic validation + utility priors for model proposals."""

    def __init__(self, config: Optional[GovernorConfig] = None):
        self._cfg = config or GovernorConfig()

    def evaluate(self, candidates: Iterable[ProposalCandidate]) -> List[GovernedCandidate]:
        out: List[GovernedCandidate] = []
        for cand in candidates:
            proposal = cand.proposal

            if proposal.risk == "high" and self._cfg.block_high_risk:
                out.append(GovernedCandidate(candidate=cand, valid=False, reject_reason="risk_high_blocked"))
                continue

            if (
                proposal.primitive == "home"
                and self._cfg.block_home_unless_reset_pose
                and proposal.intent != "reset_pose"
            ):
                out.append(GovernedCandidate(candidate=cand, valid=False, reject_reason="home_blocked"))
                continue

            if proposal.primitive == "orient_to_zone":
                zone = proposal.command.get("zone")
                if zone not in {"left", "center", "right"}:
                    out.append(GovernedCandidate(candidate=cand, valid=False, reject_reason="invalid_zone"))
                    continue

            # Base utility from model confidence and urgency. Arbiter applies hysteresis/switch rules.
            utility = (
                proposal.score * (0.55 + 0.45 * proposal.confidence)
                + (0.2 * proposal.urgency)
                + _source_bias(cand.source)
                + _risk_penalty(proposal.risk)
            )
            out.append(
                GovernedCandidate(
                    candidate=cand,
                    valid=True,
                    reject_reason=None,
                    utility=max(0.0, min(1.5, utility)),
                )
            )
        return out


def _source_bias(source: str) -> float:
    # Keep remote authority primary while allowing low-cost idle heartbeat proposals.
    if source == "remote":
        return 0.05
    if source == "idle_engine":
        return 0.0
    return 0.0


def _risk_penalty(risk: str) -> float:
    if risk == "low":
        return 0.0
    if risk == "medium":
        return -0.08
    return -0.25
