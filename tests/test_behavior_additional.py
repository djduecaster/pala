from __future__ import annotations

from pala.behavior import (
    ActionCompiler,
    Arbiter,
    ArbiterConfig,
    EnvSummarizer,
    Governor,
    IntentProposal,
    IntentProposer,
    ProposalCandidate,
)
from pala.behavior.decision_types import BehaviorMode
from pala.types import ActionPlan, HoldCommand, PrimitiveKind


def test_latest_only_bookkeeping_helpers():
    env = EnvSummarizer()
    planner = IntentProposer()

    assert env.submit_or_replace({"tick": 1}) is True
    assert env.submit_or_replace({"tick": 2}) is False
    assert env.take_latest_pending() == {"tick": 2}

    assert planner.submit_or_replace({"tick": 1}) is True
    assert planner.submit_or_replace({"tick": 2}) is False
    assert planner.take_latest_pending() == {"tick": 2}


def test_governor_blocks_high_risk_proposals():
    gov = Governor()
    candidates = [
        ProposalCandidate(
            proposal=IntentProposal(
                intent="scan_environment",
                primitive="glance",
                command={"direction": "left", "amp_rad": 0.2, "duration_s": 0.5, "rate_rad_s": 1.3},
                style="curious",
                score=0.7,
                confidence=0.7,
                urgency=0.4,
                risk="high",
                allow_interrupt=True,
                evidence=[],
                rationale_short="unsafe in test",
            ),
            source="remote",
        )
    ]

    out = gov.evaluate(candidates, mode=BehaviorMode.SCAN_EXPLORE, signals={"person_present": True})
    assert len(out) == 1
    assert out[0].valid is False
    assert out[0].reject_reason == "risk_high_blocked"


def test_action_compiler_and_arbiter_keep_current_on_same_signature():
    compiler = ActionCompiler()
    arbiter = Arbiter(ArbiterConfig(min_dwell_s=0.0, base_margin=0.01, idle_after_s=2.0))

    current = ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command={"amp_rad": 0.06, "period_s": 6.5, "rate_rad_s": 1.0},
        confidence=0.6,
        style="calm",
    )
    proposal = IntentProposal(
        intent="idle_presence",
        primitive="breath",
        command={"amp_rad": 0.06, "period_s": 6.5, "rate_rad_s": 1.0},
        style="calm",
        score=0.8,
        confidence=0.8,
        urgency=0.2,
        risk="low",
        allow_interrupt=True,
        evidence=[],
        rationale_short="same action",
    )
    governed = Governor().evaluate(
        [ProposalCandidate(proposal=proposal, source="remote")],
        mode=BehaviorMode.IDLE_PRESENCE,
        signals={"person_present": False},
    )

    result = arbiter.select(
        candidates=governed,
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
    assert result.reason == "same_signature"

    compile_result = compiler.compile(proposal)
    assert compile_result.action is not None
    assert compile_result.error is None


def test_proposer_and_env_parser_capture_diagnostic_errors():
    planner = IntentProposer()
    env = EnvSummarizer()

    planner.submit_or_replace({"tick": 1})
    env.submit_or_replace({"tick": 1})

    assert planner.complete_request("{bad json") is None
    assert env.complete_request("{bad json") is None
    assert (planner.last_parse_error or "").startswith("json_decode:")
    assert (env.last_parse_error or "").startswith("json_decode:")
