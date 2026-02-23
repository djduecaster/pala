from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any, List, Mapping, Optional, Tuple

from ..types import ActionPlan
from .types import GovernedCandidate


@dataclass
class ArbiterConfig:
    min_dwell_s: float = 1.2
    base_margin: float = 0.10
    idle_after_s: float = 6.0
    terminal_retrigger_s: float = 2.0
    takeover_no_signal_streak: int = 2
    takeover_no_commit_s: float = 2.0


@dataclass
class ArbiterResult:
    decision: str
    reason: str
    chosen: Optional[GovernedCandidate]
    best_utility: float


class Arbiter:
    """Deterministic commit logic with hysteresis and anti-collapse bias."""

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
        last_intent: str,
        recent_switches: int,
        planner_open_breaker: bool,
        planner_no_signal_streak: int,
        perception_degraded: bool,
    ) -> ArbiterResult:
        valids = [item for item in candidates if item.valid]
        if not valids:
            return ArbiterResult(
                decision="keep_current",
                reason="no_valid_candidates",
                chosen=None,
                best_utility=0.0,
            )

        current_sig = _action_signature(current_action)
        for item in valids:
            item.utility = self._adjusted_utility(
                item=item,
                current_signature=current_sig,
                no_commit_s=no_commit_s,
                last_intent=last_intent,
                planner_open_breaker=planner_open_breaker,
                perception_degraded=perception_degraded,
            )

        best = max(valids, key=lambda item: item.utility)
        best_sig = _proposal_signature(best)

        takeover = self._select_anti_collapse_takeover(
            valids=valids,
            current_signature=current_sig,
            current_action=current_action,
            action_age_s=action_age_s,
            no_commit_s=no_commit_s,
            planner_open_breaker=planner_open_breaker,
            planner_no_signal_streak=planner_no_signal_streak,
        )
        if takeover is not None:
            return ArbiterResult(
                decision="commit",
                reason="anti_collapse_takeover",
                chosen=takeover,
                best_utility=takeover.utility,
            )

        if best_sig == current_sig:
            # Terminal gestures should be re-triggerable after a short cooldown,
            # otherwise behavior can appear frozen after a one-shot action completes.
            if not _is_terminal_primitive(current_action.primitive.value) or action_age_s < self._cfg.terminal_retrigger_s:
                return ArbiterResult(
                    decision="keep_current",
                    reason="same_signature",
                    chosen=best,
                    best_utility=best.utility,
                )
            return ArbiterResult(
                decision="commit",
                reason="same_signature_retrigger",
                chosen=best,
                best_utility=best.utility,
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
            )

        margin = self._cfg.base_margin + (0.05 * max(0, recent_switches))
        if no_commit_s >= self._cfg.idle_after_s:
            margin = max(0.02, margin - 0.04)

        effective_current = self._decayed_current_utility(
            current_action=current_action,
            current_utility=current_utility,
            action_age_s=action_age_s,
            no_commit_s=no_commit_s,
        )
        threshold = max(0.0, float(effective_current) + margin)
        if best.utility >= threshold:
            return ArbiterResult(
                decision="commit",
                reason="utility_beats_threshold",
                chosen=best,
                best_utility=best.utility,
            )

        return ArbiterResult(
            decision="keep_current",
            reason="utility_below_threshold",
            chosen=best,
            best_utility=best.utility,
        )

    def _adjusted_utility(
        self,
        *,
        item: GovernedCandidate,
        current_signature: Tuple[str, str, str],
        no_commit_s: float,
        last_intent: str,
        planner_open_breaker: bool,
        perception_degraded: bool,
    ) -> float:
        proposal = item.candidate.proposal
        utility = float(item.utility)

        if proposal.intent == last_intent:
            utility -= 0.08
        else:
            utility += 0.04

        if _proposal_signature(item) == current_signature:
            utility -= 0.14

        if planner_open_breaker and item.candidate.source == "remote":
            utility -= 0.06

        if perception_degraded and item.candidate.source == "remote":
            utility -= 0.04

        if no_commit_s >= self._cfg.idle_after_s:
            if proposal.primitive == "hold":
                utility -= 0.12
            elif item.candidate.source == "idle_engine":
                utility += 0.10
            else:
                utility += 0.05

        return max(0.0, min(1.5, utility))

    def _decayed_current_utility(
        self,
        *,
        current_action: ActionPlan,
        current_utility: float,
        action_age_s: float,
        no_commit_s: float,
    ) -> float:
        utility = max(0.0, float(current_utility))
        primitive = current_action.primitive.value

        # Idle primitives should not remain sticky for long windows.
        if primitive in {"hold", "breath"}:
            if action_age_s >= 8.0:
                utility *= 0.55
            if action_age_s >= 15.0:
                utility *= 0.35

        if no_commit_s >= self._cfg.idle_after_s:
            utility *= 0.75
        if _is_terminal_primitive(primitive) and action_age_s >= self._cfg.terminal_retrigger_s:
            utility *= 0.45
        return max(0.0, min(1.5, utility))

    def _select_anti_collapse_takeover(
        self,
        *,
        valids: List[GovernedCandidate],
        current_signature: Tuple[str, str, str],
        current_action: ActionPlan,
        action_age_s: float,
        no_commit_s: float,
        planner_open_breaker: bool,
        planner_no_signal_streak: int,
    ) -> Optional[GovernedCandidate]:
        required_no_commit_s = max(0.2, self._cfg.takeover_no_commit_s)
        takeover_due = False
        if planner_open_breaker and no_commit_s >= required_no_commit_s:
            takeover_due = True
        elif (
            not planner_open_breaker
            and planner_no_signal_streak >= max(1, self._cfg.takeover_no_signal_streak)
            and no_commit_s >= (required_no_commit_s + 1.0)
        ):
            takeover_due = True
        if not takeover_due:
            return None

        # Prefer non-hold idle engine proposals to break no-op loops quickly.
        ordered = sorted(
            valids,
            key=lambda item: (
                item.candidate.source != "idle_engine",
                item.candidate.proposal.primitive == "hold",
                -item.utility,
            ),
        )
        for item in ordered:
            sig = _proposal_signature(item)
            if sig != current_signature:
                return item
            if _is_terminal_primitive(current_action.primitive.value) and action_age_s >= self._cfg.terminal_retrigger_s:
                return item
        return None


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
