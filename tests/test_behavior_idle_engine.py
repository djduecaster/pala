from __future__ import annotations

from pala.behavior.decision_types import BehaviorMode
from pala.behavior.idle_engine import IdleEngine, IdleEngineConfig


def _signals():
    return {"person_present": True, "activity_level": 0.4, "novelty": 0.5}


def test_idle_engine_recover_mode_returns_only_home_reset():
    engine = IdleEngine(IdleEngineConfig(idle_after_s=2.0, glance_after_s=3.0))
    proposals = engine.propose(
        mode=BehaviorMode.RECOVER_RESET,
        no_commit_s=99.0,
        zone_hint="left",
        tick_index=1,
        signals=_signals(),
    )
    assert len(proposals) == 1
    only = proposals[0]
    assert only.intent == "reset_pose"
    assert only.primitive == "home"
    assert only.allow_interrupt is False
    assert only.min_dwell_ms == 800


def test_idle_engine_idle_mode_emits_breath_and_hold_with_low_no_commit():
    engine = IdleEngine(IdleEngineConfig(idle_after_s=6.0, glance_after_s=8.0))
    proposals = engine.propose(
        mode=BehaviorMode.IDLE_PRESENCE,
        no_commit_s=1.0,
        zone_hint="left",
        tick_index=2,
        signals=_signals(),
    )
    primitives = [p.primitive for p in proposals]
    assert primitives == ["breath", "hold"]
    assert proposals[0].score == 0.18
    assert proposals[-1].intent == "idle_presence"


def test_idle_engine_glance_direction_and_orient_zone_paths():
    engine = IdleEngine(IdleEngineConfig(idle_after_s=2.0, glance_after_s=4.0))

    # Odd tick with large no_commit should add right glance due timeout condition.
    timeout_proposals = engine.propose(
        mode=BehaviorMode.IDLE_PRESENCE,
        no_commit_s=5.0,
        zone_hint=None,
        tick_index=3,
        signals=_signals(),
    )
    glance = [p for p in timeout_proposals if p.primitive == "glance"][0]
    assert glance.command["direction"] == "right"
    assert timeout_proposals[0].score == 0.30

    # Scan mode should add glance even without timeout.
    scan_proposals = engine.propose(
        mode=BehaviorMode.SCAN_EXPLORE,
        no_commit_s=0.5,
        zone_hint=None,
        tick_index=2,
        signals=_signals(),
    )
    scan_glance = [p for p in scan_proposals if p.primitive == "glance"][0]
    assert scan_glance.command["direction"] == "left"

    # Engage mode with valid zone hint should add orient_to_zone proposal.
    engage_proposals = engine.propose(
        mode=BehaviorMode.ENGAGE_TRACK,
        no_commit_s=0.5,
        zone_hint="right",
        tick_index=4,
        signals=_signals(),
    )
    orient = [p for p in engage_proposals if p.primitive == "orient_to_zone"][0]
    assert orient.command["zone"] == "right"
    assert orient.evidence == ["idle:zone:right"]
