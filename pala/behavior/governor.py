from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional

from .decision_types import BehaviorMode
from .types import GovernedCandidate, ProposalCandidate


@dataclass
class GovernorConfig:
    block_high_risk: bool = True
    block_home_unless_reset_pose: bool = True
    stale_penalty_per_s: float = 0.05
    stale_expire_s: float = 14.0


class Governor:
    """Deterministic validation + mode-aware utility priors for proposals."""

    def __init__(self, config: Optional[GovernorConfig] = None):
        self._cfg = config or GovernorConfig()

    def evaluate(
        self,
        candidates: Iterable[ProposalCandidate],
        *,
        mode: BehaviorMode,
        signals: Mapping[str, object],
    ) -> List[GovernedCandidate]:
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

            reason = _validate_command(proposal.primitive, proposal.command)
            if reason is not None:
                out.append(GovernedCandidate(candidate=cand, valid=False, reject_reason=reason))
                continue

            age_s = max(0.0, float(cand.age_s))
            if age_s > self._cfg.stale_expire_s:
                out.append(GovernedCandidate(candidate=cand, valid=False, reject_reason="stale_expired"))
                continue

            utility = (
                proposal.score * (0.50 + 0.50 * proposal.confidence)
                + 0.18 * proposal.urgency
                + _source_bias(cand.source)
                + _risk_penalty(proposal.risk)
                + _mode_affinity(mode, proposal.primitive)
                + _presence_boost(signals, proposal.primitive)
                - (self._cfg.stale_penalty_per_s * age_s)
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


def _validate_command(primitive: str, command: Mapping[str, object]) -> Optional[str]:
    if primitive == "orient_to_zone":
        zone = command.get("zone")
        if zone not in {"left", "center", "right"}:
            return "invalid_zone"
    if primitive == "glance":
        direction = command.get("direction")
        if direction not in {"left", "right", "up", "down"}:
            return "invalid_direction"
    return None


def _source_bias(source: str) -> float:
    if source == "remote":
        return 0.06
    if source == "idle_engine":
        return 0.0
    return 0.0


def _risk_penalty(risk: str) -> float:
    if risk == "low":
        return 0.0
    if risk == "medium":
        return -0.08
    return -0.25


def _mode_affinity(mode: BehaviorMode, primitive: str) -> float:
    table = {
        BehaviorMode.IDLE_PRESENCE: {"breath": 0.08, "hold": 0.03, "glance": 0.02, "orient_to_zone": -0.02},
        BehaviorMode.ENGAGE_TRACK: {"orient_to_zone": 0.15, "nod": 0.05, "breath": -0.03, "hold": -0.08},
        BehaviorMode.ACKNOWLEDGE: {"nod": 0.14, "glance": 0.10, "orient_to_zone": 0.05, "hold": -0.06},
        BehaviorMode.SCAN_EXPLORE: {"glance": 0.14, "orient_to_zone": 0.06, "breath": -0.02},
        BehaviorMode.RECOVER_RESET: {"home": 0.18, "hold": 0.02},
    }
    return table.get(mode, {}).get(primitive, 0.0)


def _presence_boost(signals: Mapping[str, object], primitive: str) -> float:
    person_present = bool(signals.get("person_present", False))
    if not person_present:
        return 0.0
    if primitive in {"orient_to_zone", "nod", "glance"}:
        return 0.06
    if primitive in {"hold", "breath"}:
        return -0.04
    return 0.0
