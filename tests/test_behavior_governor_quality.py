from __future__ import annotations

from pala.behavior.decision_types import BehaviorMode
from pala.behavior.governor import (
    Governor,
    GovernorConfig,
    _allowed_primitives_for_mode,
    _presence_boost,
    _risk_penalty,
    _source_bias,
    _validate_command,
)
from pala.behavior.types import IntentProposal, ProposalCandidate


def _proposal(**overrides):
    payload = {
        "intent": "track_user",
        "primitive": "orient_to_zone",
        "command": {"zone": "left", "amp_rad": 0.25, "rate_rad_s": 1.2},
        "style": "focused",
        "score": 0.75,
        "confidence": 0.7,
        "urgency": 0.3,
        "risk": "low",
        "allow_interrupt": True,
        "evidence": ["frame:latest"],
        "rationale_short": "keep tracking user",
    }
    payload.update(overrides)
    return IntentProposal(**payload)


def _candidate(*, source: str = "remote", age_s: float = 0.0, **proposal_overrides):
    return ProposalCandidate(  # type: ignore[arg-type]
        proposal=_proposal(**proposal_overrides),
        source=source,
        age_s=age_s,
    )


def test_governor_rejects_stale_candidates_and_invalid_glance_direction():
    gov = Governor(GovernorConfig(block_high_risk=False, stale_expire_s=1.0))
    stale = _candidate(age_s=1.2)
    bad_glance = _candidate(
        primitive="glance",
        intent="scan_environment",
        style="curious",
        command={"direction": "forward", "amp_rad": 0.2, "duration_s": 0.5, "rate_rad_s": 1.2},
    )

    out = gov.evaluate(
        [stale, bad_glance],
        mode=BehaviorMode.SCAN_EXPLORE,
        signals={"person_present": False},
    )

    assert out[0].valid is False
    assert out[0].reject_reason == "stale_expired"
    assert out[1].valid is False
    assert out[1].reject_reason == "invalid_direction"


def test_governor_applies_high_risk_penalty_when_high_risk_not_blocked():
    gov = Governor(GovernorConfig(block_high_risk=False))
    low = _candidate(risk="low", source="remote")
    high = _candidate(risk="high", source="partner")

    out = gov.evaluate(
        [low, high],
        mode=BehaviorMode.ENGAGE_TRACK,
        signals={"person_present": False},
    )

    assert out[0].valid is True
    assert out[1].valid is True
    assert out[1].utility < out[0].utility


def test_governor_helper_branches_for_presence_and_fallbacks():
    assert _validate_command("glance", {"direction": "left"}) is None  # noqa: SLF001
    assert _validate_command("glance", {"direction": "diagonal"}) == "invalid_direction"  # noqa: SLF001

    assert _source_bias("remote") == 0.06  # noqa: SLF001
    assert _source_bias("idle_engine") == 0.0  # noqa: SLF001
    assert _source_bias("partner") == 0.0  # noqa: SLF001

    assert _risk_penalty("low") == 0.0  # noqa: SLF001
    assert _risk_penalty("medium") == -0.08  # noqa: SLF001
    assert _risk_penalty("high") == -0.25  # noqa: SLF001

    assert _presence_boost({"person_present": False}, "glance") == 0.0  # noqa: SLF001
    assert _presence_boost({"person_present": True}, "nod") == 0.06  # noqa: SLF001
    assert _presence_boost({"person_present": True}, "hold") == -0.04  # noqa: SLF001
    assert _presence_boost({"person_present": True}, "home") == 0.0  # noqa: SLF001


def test_governor_rejects_mode_disallowed_primitives():
    gov = Governor(GovernorConfig(block_high_risk=False))
    candidate = _candidate(primitive="home", intent="reset_pose", command={"rate_rad_s": 1.2})
    out = gov.evaluate([candidate], mode=BehaviorMode.SCAN_EXPLORE, signals={"person_present": False})
    assert out[0].valid is False
    assert out[0].reject_reason == "mode_disallowed"

    recover = gov.evaluate([candidate], mode=BehaviorMode.RECOVER_RESET, signals={"person_present": False})
    assert recover[0].valid is True

    assert "home" in _allowed_primitives_for_mode(BehaviorMode.RECOVER_RESET)  # noqa: SLF001
