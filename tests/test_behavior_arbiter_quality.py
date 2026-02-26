from __future__ import annotations

import pytest

from pala.behavior.arbiter import Arbiter, ArbiterConfig, _best_non_matching, _command_signature
from pala.behavior.decision_types import BehaviorMode
from pala.behavior.types import GovernedCandidate, IntentProposal, ProposalCandidate
from pala.types import ActionPlan, HoldCommand, PrimitiveKind


def _proposal(**overrides):
    payload = {
        "intent": "track_user",
        "primitive": "orient_to_zone",
        "command": {"zone": "left", "amp_rad": 0.2, "rate_rad_s": 1.2},
        "style": "focused",
        "score": 0.8,
        "confidence": 0.7,
        "urgency": 0.4,
        "risk": "low",
        "allow_interrupt": True,
        "evidence": ["frame:latest"],
        "rationale_short": "track user",
    }
    payload.update(overrides)
    return IntentProposal(**payload)


def _governed(*, proposal: IntentProposal, valid: bool = True, utility: float = 0.8):
    return GovernedCandidate(
        candidate=ProposalCandidate(proposal=proposal, source="remote"),
        valid=valid,
        reject_reason=None if valid else "invalid",
        utility=utility,
    )


def test_arbiter_returns_keep_current_when_no_valid_candidates():
    arb = Arbiter()
    current = ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=0.2, style="calm")
    result = arb.select(
        candidates=[_governed(proposal=_proposal(), valid=False)],
        current_action=current,
        current_utility=0.4,
        action_age_s=1.0,
        no_commit_s=1.0,
        recent_switches=0,
        planner_open_breaker=False,
        same_primitive_streak=1,
        mode=BehaviorMode.IDLE_PRESENCE,
    )
    assert result.decision == "keep_current"
    assert result.reason == "no_valid_candidates"


def test_arbiter_commits_when_utility_beats_threshold():
    arb = Arbiter(ArbiterConfig(min_dwell_s=0.0, base_margin=0.01, idle_after_s=1.0))
    current = ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=0.2, style="calm")
    result = arb.select(
        candidates=[_governed(proposal=_proposal(), utility=1.2)],
        current_action=current,
        current_utility=0.2,
        action_age_s=2.0,
        no_commit_s=2.0,
        recent_switches=0,
        planner_open_breaker=False,
        same_primitive_streak=1,
        mode=BehaviorMode.ENGAGE_TRACK,
    )
    assert result.decision == "commit"
    assert result.chosen is not None
    assert result.reason == "utility_beats_threshold"


def test_arbiter_limits_repeated_same_primitive_with_alternate_choice():
    arb = Arbiter(ArbiterConfig(min_dwell_s=0.0, base_margin=0.0))
    current = ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command={"amp_rad": 0.08, "period_s": 7.0, "rate_rad_s": 1.0},
        confidence=0.6,
        style="calm",
    )
    same = _governed(
        proposal=_proposal(
            primitive="breath",
            intent="idle_presence",
            command={"amp_rad": 0.08, "period_s": 7.0, "rate_rad_s": 1.0},
            style="calm",
        ),
        utility=1.0,
    )
    alt = _governed(
        proposal=_proposal(
            primitive="glance",
            intent="scan_environment",
            command={"direction": "right", "amp_rad": 0.2, "duration_s": 0.5, "rate_rad_s": 1.4},
            style="curious",
        ),
        utility=0.95,
    )
    result = arb.select(
        candidates=[same, alt],
        current_action=current,
        current_utility=0.9,
        action_age_s=6.0,
        no_commit_s=6.0,
        recent_switches=0,
        planner_open_breaker=False,
        same_primitive_streak=4,
        mode=BehaviorMode.SCAN_EXPLORE,
    )
    assert result.decision == "commit"
    assert result.reason == "utility_beats_threshold"
    assert result.chosen is alt
    assert result.chosen.candidate.proposal.primitive == "glance"


def test_arbiter_uses_candidate_min_dwell_override_when_interrupts_are_disallowed():
    arb = Arbiter(ArbiterConfig(min_dwell_s=0.0, base_margin=0.0))
    current = ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=0.4, style="calm")
    slow_candidate = _governed(
        proposal=_proposal(allow_interrupt=False, min_dwell_ms=5000),
        utility=1.2,
    )

    result = arb.select(
        candidates=[slow_candidate],
        current_action=current,
        current_utility=0.2,
        action_age_s=1.0,
        no_commit_s=0.1,
        recent_switches=0,
        planner_open_breaker=False,
        same_primitive_streak=1,
        mode=BehaviorMode.IDLE_PRESENCE,
    )
    assert result.decision == "keep_current"
    assert result.reason == "min_dwell_not_met"


def test_arbiter_applies_orient_cooldown_to_reduce_same_zone_chatter():
    arb = Arbiter(ArbiterConfig(min_dwell_s=0.0, base_margin=0.0, orient_cooldown_s=2.0))
    current = ActionPlan(
        primitive=PrimitiveKind.ORIENT_TO_ZONE,
        command={"zone": "left", "amp_rad": 0.2, "rate_rad_s": 1.2},
        confidence=0.7,
        style="focused",
    )
    candidate = _governed(
        proposal=_proposal(
            primitive="orient_to_zone",
            command={"zone": "left", "amp_rad": 0.24, "rate_rad_s": 1.2},
        ),
        utility=1.1,
    )

    early = arb.select(
        candidates=[candidate],
        current_action=current,
        current_utility=0.5,
        action_age_s=0.5,
        no_commit_s=0.5,
        recent_switches=0,
        planner_open_breaker=False,
        same_primitive_streak=1,
        mode=BehaviorMode.ENGAGE_TRACK,
    )
    assert early.decision == "keep_current"
    assert early.reason == "orient_cooldown"

    late = arb.select(
        candidates=[candidate],
        current_action=current,
        current_utility=0.5,
        action_age_s=2.3,
        no_commit_s=2.3,
        recent_switches=0,
        planner_open_breaker=False,
        same_primitive_streak=1,
        mode=BehaviorMode.ENGAGE_TRACK,
    )
    assert late.decision == "commit"
    assert late.reason == "utility_beats_threshold"


def test_arbiter_allows_orient_zone_switch_during_cooldown_window():
    arb = Arbiter(ArbiterConfig(min_dwell_s=0.0, base_margin=0.0, orient_cooldown_s=2.0))
    current = ActionPlan(
        primitive=PrimitiveKind.ORIENT_TO_ZONE,
        command={"zone": "left", "amp_rad": 0.2, "rate_rad_s": 1.2},
        confidence=0.7,
        style="focused",
    )
    candidate = _governed(
        proposal=_proposal(
            primitive="orient_to_zone",
            command={"zone": "right", "amp_rad": 0.2, "rate_rad_s": 1.2},
        ),
        utility=1.1,
    )

    result = arb.select(
        candidates=[candidate],
        current_action=current,
        current_utility=0.5,
        action_age_s=0.5,
        no_commit_s=0.5,
        recent_switches=0,
        planner_open_breaker=False,
        same_primitive_streak=1,
        mode=BehaviorMode.ENGAGE_TRACK,
    )
    assert result.decision == "commit"
    assert result.reason == "utility_beats_threshold"


def test_arbiter_planner_breaker_adds_remote_threshold_penalty():
    arb = Arbiter(ArbiterConfig(min_dwell_s=0.0, base_margin=0.0, idle_after_s=30.0))
    current = ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=0.5, style="calm")
    candidate = _governed(proposal=_proposal(), utility=0.42)

    no_breaker = arb.select(
        candidates=[candidate],
        current_action=current,
        current_utility=0.4,
        action_age_s=1.0,
        no_commit_s=0.0,
        recent_switches=0,
        planner_open_breaker=False,
        same_primitive_streak=1,
        mode=BehaviorMode.IDLE_PRESENCE,
    )
    with_breaker = arb.select(
        candidates=[candidate],
        current_action=current,
        current_utility=0.4,
        action_age_s=1.0,
        no_commit_s=0.0,
        recent_switches=0,
        planner_open_breaker=True,
        same_primitive_streak=1,
        mode=BehaviorMode.IDLE_PRESENCE,
    )

    assert no_breaker.decision == "commit"
    assert with_breaker.decision == "keep_current"
    assert with_breaker.threshold == pytest.approx(no_breaker.threshold + 0.04)


def test_arbiter_decay_and_margin_cover_long_age_and_terminal_branches():
    arb = Arbiter(ArbiterConfig(base_margin=0.08, idle_after_s=5.0, terminal_retrigger_s=1.0))
    hold_action = ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=0.5, style="calm")
    hold_decay = arb._decayed_current_utility(  # noqa: SLF001
        current_action=hold_action,
        current_utility=1.0,
        action_age_s=9.0,
        no_commit_s=6.0,
        mode=BehaviorMode.SCAN_EXPLORE,
    )
    assert hold_decay == pytest.approx(1.0 * 0.70 * 0.45 * 0.75 * 0.80)

    terminal_action = ActionPlan(
        primitive=PrimitiveKind.GLANCE,
        command={"direction": "left", "amp_rad": 0.2, "duration_s": 0.5, "rate_rad_s": 1.2},
        confidence=0.5,
        style="curious",
    )
    terminal_decay = arb._decayed_current_utility(  # noqa: SLF001
        current_action=terminal_action,
        current_utility=1.0,
        action_age_s=2.0,
        no_commit_s=0.0,
        mode=BehaviorMode.IDLE_PRESENCE,
    )
    assert terminal_decay == pytest.approx(0.55)

    assert arb._margin(recent_switches=0, no_commit_s=0.0, mode=BehaviorMode.ACKNOWLEDGE) == pytest.approx(0.05)  # noqa: SLF001
    assert arb._margin(recent_switches=0, no_commit_s=0.0, mode=BehaviorMode.RECOVER_RESET) == pytest.approx(0.11)  # noqa: SLF001


def test_arbiter_helper_fallbacks_for_no_alternate_and_non_mapping_commands():
    same_only = _governed(
        proposal=_proposal(
            intent="idle_presence",
            primitive="hold",
            command={},
            style="calm",
        ),
        utility=0.6,
    )
    assert _best_non_matching([same_only], "hold") is None  # noqa: SLF001
    assert _command_signature("raw-command") == "'raw-command'"  # noqa: SLF001
