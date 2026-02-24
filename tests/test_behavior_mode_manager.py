from __future__ import annotations

from pala.behavior.decision_types import BehaviorMode, ModeSignals
from pala.behavior.mode_manager import ModeManager, ModeManagerConfig


def _signals(
    *,
    person_present: bool = False,
    person_conf: float = 0.0,
    activity_level: float = 0.0,
    novelty: float = 0.0,
    env_delta: float = 0.0,
    planner_open_breaker: bool = False,
    perception_degraded: bool = False,
) -> ModeSignals:
    return ModeSignals(
        person_present=person_present,
        person_conf=person_conf,
        activity_level=activity_level,
        novelty=novelty,
        env_delta=env_delta,
        planner_open_breaker=planner_open_breaker,
        perception_degraded=perception_degraded,
    )


def test_mode_manager_reset_and_health_dominated_transition():
    mgr = ModeManager()
    mgr.reset(now_mono_s=3.0)
    assert mgr.snapshot.mode == BehaviorMode.IDLE_PRESENCE
    assert mgr.snapshot.entered_mono_s == 3.0
    assert mgr.snapshot.reason == "reset"

    decision = mgr.update(
        now_mono_s=4.0,
        signals=_signals(planner_open_breaker=True),
    )
    assert decision.next_mode == BehaviorMode.RECOVER_RESET
    assert decision.reason == "health_degraded"
    assert decision.transitioned is True


def test_mode_manager_person_presence_paths_and_hysteresis():
    cfg = ModeManagerConfig(min_mode_dwell_s=0.5, engage_person_conf=0.45, disengage_person_conf=0.2, novelty_for_ack=0.4)
    mgr = ModeManager(cfg)

    # Starts idle, high confidence + novelty triggers ACK.
    d1 = mgr.update(now_mono_s=1.0, signals=_signals(person_present=True, person_conf=0.8, novelty=0.8))
    assert d1.next_mode == BehaviorMode.ACKNOWLEDGE
    assert d1.reason == "presence_novelty_ack"

    # Immediate switch is blocked by min dwell.
    d2 = mgr.update(now_mono_s=1.2, signals=_signals(person_present=True, person_conf=0.8, novelty=0.1))
    assert d2.next_mode == BehaviorMode.ACKNOWLEDGE
    assert d2.reason == "min_dwell_hold"
    assert d2.transitioned is False

    # After dwell passes, lower novelty should engage tracking.
    d3 = mgr.update(now_mono_s=1.8, signals=_signals(person_present=True, person_conf=0.8, novelty=0.1))
    assert d3.next_mode == BehaviorMode.ENGAGE_TRACK
    assert d3.reason == "presence_track"

    # Presence drop below disengage threshold transitions away from engage.
    d4 = mgr.update(now_mono_s=3.0, signals=_signals(person_present=True, person_conf=0.1, activity_level=0.4))
    assert d4.next_mode == BehaviorMode.SCAN_EXPLORE
    assert d4.reason == "disengage_presence_drop"


def test_mode_manager_ack_fallthrough_and_idle_scan_paths():
    mgr = ModeManager(ModeManagerConfig(min_mode_dwell_s=0.0, activity_for_scan=0.3))

    # First enter ACK mode.
    mgr.update(now_mono_s=1.0, signals=_signals(person_present=True, person_conf=0.9, novelty=0.9))
    assert mgr.snapshot.mode == BehaviorMode.ACKNOWLEDGE

    # Losing person from ACK falls through to scan/idle decision.
    d1 = mgr.update(now_mono_s=2.0, signals=_signals(person_present=False, activity_level=0.1))
    assert d1.next_mode == BehaviorMode.IDLE_PRESENCE
    assert d1.reason == "ack_to_idle_or_scan"

    d2 = mgr.update(now_mono_s=3.0, signals=_signals(person_present=False, activity_level=0.7))
    assert d2.next_mode == BehaviorMode.SCAN_EXPLORE
    assert d2.reason == "activity_scan"

    d3 = mgr.update(now_mono_s=4.0, signals=_signals(person_present=False, env_delta=0.4))
    assert d3.next_mode == BehaviorMode.SCAN_EXPLORE
    assert d3.reason == "activity_scan"
