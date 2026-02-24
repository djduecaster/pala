from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any, List, Mapping, Optional, Tuple

from ..types import ActionPlan
from .decision_types import BehaviorMode
from .types import GovernedCandidate


@dataclass
class ArbiterConfig:
    min_dwell_s: float = 1.0
    base_margin: float = 0.08
    idle_after_s: float = 5.0
    terminal_retrigger_s: float = 1.4
    max_same_primitive_streak: int = 3


@dataclass
class ArbiterResult:
    decision: str
    reason: str
    chosen: Optional[GovernedCandidate]
    best_utility: float
    threshold: float
    effective_current: float
    margin: float


class Arbiter:
    """Deterministic commit logic with hysteresis, diversity, and idle de-stick."""

    def __init__(self, config: Optional[ArbiterConfig] = None):
        self._cfg = config or ArbiterConfig()

    def select(
        self,
        *,
        candidates: List[GovernedCandidate],
        current_action: ActionPlan,
        current_utility: float,
        action_age_s: float,
        no_commit_s: float,
        recent_switches: int,
        planner_open_breaker: bool,
        same_primitive_streak: int,
        mode: BehaviorMode,
    ) -> ArbiterResult:
        valids = [item for item in candidates if item.valid]
        if not valids:
            return ArbiterResult(
                decision="keep_current",
                reason="no_valid_candidates",
                chosen=None,
                best_utility=0.0,
                threshold=0.0,
                effective_current=0.0,
                margin=0.0,
            )

        current_sig = _action_signature(current_action)
        best = max(valids, key=lambda item: item.utility)
        best_sig = _proposal_signature(best)

        if same_primitive_streak >= self._cfg.max_same_primitive_streak:
            alternate = _best_non_matching(valids, current_action.primitive.value)
            if alternate is not None and alternate.utility >= max(0.05, best.utility - 0.12):
                best = alternate
                best_sig = _proposal_signature(best)

        if best_sig == current_sig:
            if _is_terminal_primitive(current_action.primitive.value) and action_age_s >= self._cfg.terminal_retrigger_s:
                return ArbiterResult(
                    decision="commit",
                    reason="same_signature_retrigger",
                    chosen=best,
                    best_utility=best.utility,
                    threshold=0.0,
                    effective_current=0.0,
                    margin=0.0,
                )
            return ArbiterResult(
                decision="keep_current",
                reason="same_signature",
                chosen=best,
                best_utility=best.utility,
                threshold=0.0,
                effective_current=0.0,
                margin=0.0,
            )

        min_dwell_s = self._cfg.min_dwell_s
        if best.candidate.proposal.min_dwell_ms is not None:
            min_dwell_s = max(0.0, float(best.candidate.proposal.min_dwell_ms) / 1000.0)
        if action_age_s < min_dwell_s and not best.candidate.proposal.allow_interrupt:
            return ArbiterResult(
                decision="keep_current",
                reason="min_dwell_not_met",
                chosen=best,
                best_utility=best.utility,
                threshold=0.0,
                effective_current=0.0,
                margin=0.0,
            )

        effective_current = self._decayed_current_utility(
            current_action=current_action,
            current_utility=current_utility,
            action_age_s=action_age_s,
            no_commit_s=no_commit_s,
            mode=mode,
        )
        margin = self._margin(recent_switches=recent_switches, no_commit_s=no_commit_s, mode=mode)
        threshold = max(0.0, effective_current + margin)

        if planner_open_breaker and best.candidate.source == "remote":
            threshold += 0.04

        if best.utility >= threshold:
            return ArbiterResult(
                decision="commit",
                reason="utility_beats_threshold",
                chosen=best,
                best_utility=best.utility,
                threshold=threshold,
                effective_current=effective_current,
                margin=margin,
            )

        return ArbiterResult(
            decision="keep_current",
            reason="utility_below_threshold",
            chosen=best,
            best_utility=best.utility,
            threshold=threshold,
            effective_current=effective_current,
            margin=margin,
        )

    def _margin(self, *, recent_switches: int, no_commit_s: float, mode: BehaviorMode) -> float:
        margin = self._cfg.base_margin + (0.02 * max(0, recent_switches))
        if no_commit_s >= self._cfg.idle_after_s:
            margin = max(0.02, margin - 0.04)
        if mode == BehaviorMode.ACKNOWLEDGE:
            margin = max(0.02, margin - 0.03)
        if mode == BehaviorMode.RECOVER_RESET:
            margin += 0.03
        return margin

    def _decayed_current_utility(
        self,
        *,
        current_action: ActionPlan,
        current_utility: float,
        action_age_s: float,
        no_commit_s: float,
        mode: BehaviorMode,
    ) -> float:
        utility = max(0.0, float(current_utility))
        primitive = current_action.primitive.value

        if primitive in {"hold", "breath"}:
            if action_age_s >= 4.0:
                utility *= 0.70
            if action_age_s >= 8.0:
                utility *= 0.45

        if mode in {BehaviorMode.SCAN_EXPLORE, BehaviorMode.ENGAGE_TRACK} and primitive in {"hold", "breath"}:
            utility *= 0.75

        if no_commit_s >= self._cfg.idle_after_s:
            utility *= 0.80

        if _is_terminal_primitive(primitive) and action_age_s >= self._cfg.terminal_retrigger_s:
            utility *= 0.55

        return max(0.0, min(1.5, utility))


def _action_signature(action: ActionPlan) -> Tuple[str, str, str]:
    return (action.primitive.value, _command_signature(action.command), action.style)


def _proposal_signature(item: GovernedCandidate) -> Tuple[str, str, str]:
    proposal = item.candidate.proposal
    return (proposal.primitive, _command_signature(proposal.command), proposal.style)


def _command_signature(command: Any) -> str:
    if is_dataclass(command):
        data = {k: getattr(command, k) for k in command.__dataclass_fields__.keys()}  # type: ignore[attr-defined]
        return _mapping_signature(data)
    if isinstance(command, Mapping):
        return _mapping_signature(command)
    return repr(command)


def _mapping_signature(mapping: Mapping[str, Any]) -> str:
    return repr(sorted((str(key), mapping[key]) for key in mapping.keys()))


def _is_terminal_primitive(primitive: str) -> bool:
    return primitive in {"glance", "nod", "orient_to_zone", "home"}


def _best_non_matching(valids: List[GovernedCandidate], primitive: str) -> Optional[GovernedCandidate]:
    non_matching = [item for item in valids if item.candidate.proposal.primitive != primitive]
    if not non_matching:
        return None
    return max(non_matching, key=lambda item: item.utility)
