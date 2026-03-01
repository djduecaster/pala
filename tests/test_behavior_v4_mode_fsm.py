from __future__ import annotations

from pala.behavior.mode_fsm_v4 import MacroMode, ModeFsmV4, ModeSignalsV4


def test_mode_fsm_boot_to_idle_on_startup_complete():
    fsm = ModeFsmV4()
    fsm.reset(now_mono_s=0.0)
    transition = fsm.update(
        now_mono_s=1.0,
        signals=ModeSignalsV4(startup_complete=True),
    )
    assert transition.transitioned is True
    assert transition.next_mode == MacroMode.IDLE_PRESENCE
    assert transition.reason == "startup_complete"


def test_mode_fsm_holds_mode_until_min_dwell():
    fsm = ModeFsmV4()
    fsm.reset(now_mono_s=0.0)
    _ = fsm.update(now_mono_s=1.0, signals=ModeSignalsV4(startup_complete=True))
    transition = fsm.update(
        now_mono_s=1.1,
        signals=ModeSignalsV4(person_present=True, person_conf=0.9, startup_complete=True),
    )
    assert transition.next_mode == MacroMode.IDLE_PRESENCE
    assert transition.reason == "min_mode_dwell_hold"


def test_mode_fsm_transitions_to_social_on_presence():
    fsm = ModeFsmV4()
    fsm.reset(now_mono_s=0.0)
    _ = fsm.update(now_mono_s=1.0, signals=ModeSignalsV4(startup_complete=True))
    transition = fsm.update(
        now_mono_s=2.5,
        signals=ModeSignalsV4(person_present=True, person_conf=0.8, startup_complete=True),
    )
    assert transition.transitioned is True
    assert transition.next_mode == MacroMode.SOCIAL_INTERACT
    assert transition.reason == "person_present_engage"


def test_mode_fsm_health_preempts_to_recover():
    fsm = ModeFsmV4()
    fsm.reset(now_mono_s=0.0)
    transition = fsm.update(
        now_mono_s=0.2,
        signals=ModeSignalsV4(health_degraded=True),
    )
    assert transition.transitioned is True
    assert transition.next_mode == MacroMode.RECOVER_RESET
    assert transition.reason == "health_degraded"


def test_mode_fsm_force_mode_transition():
    fsm = ModeFsmV4()
    fsm.reset(now_mono_s=0.0)
    _ = fsm.update(now_mono_s=1.0, signals=ModeSignalsV4(startup_complete=True))
    transition = fsm.force_mode(now_mono_s=1.1, next_mode=MacroMode.SEARCH_ASSIST, reason="external_force")
    assert transition.transitioned is True
    assert transition.previous_mode == MacroMode.IDLE_PRESENCE
    assert transition.next_mode == MacroMode.SEARCH_ASSIST
    assert transition.reason == "external_force"


def test_mode_fsm_home_request_transitions_to_return_home():
    fsm = ModeFsmV4()
    fsm.reset(now_mono_s=0.0)
    _ = fsm.update(now_mono_s=1.0, signals=ModeSignalsV4(startup_complete=True))
    transition = fsm.update(
        now_mono_s=2.4,
        signals=ModeSignalsV4(home_requested=True, startup_complete=True),
    )
    assert transition.transitioned is True
    assert transition.next_mode == MacroMode.RETURN_HOME
    assert transition.reason == "home_requested"


def test_mode_fsm_return_home_completes_to_idle():
    fsm = ModeFsmV4()
    fsm.reset(now_mono_s=0.0)
    _ = fsm.update(now_mono_s=1.0, signals=ModeSignalsV4(startup_complete=True))
    _ = fsm.force_mode(now_mono_s=2.0, next_mode=MacroMode.RETURN_HOME, reason="test")
    transition = fsm.update(
        now_mono_s=2.3,
        signals=ModeSignalsV4(home_completed=True, startup_complete=True),
    )
    assert transition.transitioned is True
    assert transition.next_mode == MacroMode.IDLE_PRESENCE
    assert transition.reason == "home_complete"
