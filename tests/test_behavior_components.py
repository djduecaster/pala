from __future__ import annotations

from pala.behavior.action_governor import ActionGovernor, ActionGovernorConfig
from pala.behavior.models import SceneObservation
from pala.behavior.scene_interpreter import SceneInterpreter
from pala.control.primitives import PrimitiveKind, HoldCommand, BreathCommand, OrientToZoneCommand
from pala.types import ActionPlan, BBoxNorm, PerceptionState


class _Clock:
    def __init__(self, values):
        self._values = list(values)
        self._last = float(self._values[-1]) if self._values else 0.0

    def __call__(self):
        if self._values:
            self._last = float(self._values.pop(0))
            return self._last
        self._last += 0.1
        return self._last


def test_scene_interpreter_emits_presence_and_zone_events():
    clock = _Clock([0.0, 0.1, 0.2, 0.3])
    interp = SceneInterpreter(clock=clock)
    none_state = PerceptionState(timestamp_monotonic_s=0.0, primary_person=None)
    left = PerceptionState(
        timestamp_monotonic_s=0.1,
        primary_person=BBoxNorm(cx=0.2, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.9,
    )
    right = PerceptionState(
        timestamp_monotonic_s=0.2,
        primary_person=BBoxNorm(cx=0.8, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.9,
    )

    a = interp.observe(none_state)
    b = interp.observe(left)
    c = interp.observe(right)
    d = interp.observe(none_state)

    assert a.event == "no_person"
    assert b.event == "person_entered"
    assert c.event == "zone_changed"
    assert d.event == "person_exited"


def test_action_governor_forces_interrupt_when_switching_from_hold():
    clock = _Clock([0.0, 0.1])
    gov = ActionGovernor(clock=clock)
    obs = SceneObservation(
        ts_mono_s=0.0,
        person_present=True,
        person_conf=0.9,
        zone="left",
        zone_changed=False,
        zone_dwell_s=0.0,
        activity_hint=None,
        event="person_present",
    )

    hold = ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=0.5, cancel_current=False)
    orient = ActionPlan(
        primitive=PrimitiveKind.ORIENT_TO_ZONE,
        command=OrientToZoneCommand(zone="left"),
        confidence=0.7,
        cancel_current=False,
    )
    first = gov.apply(hold, obs)
    second = gov.apply(orient, obs)
    assert first.cancel_current is False
    assert second.cancel_current is True


def test_action_governor_refreshes_after_long_breath():
    clock = _Clock([0.0, 4.0])
    gov = ActionGovernor(cfg=ActionGovernorConfig(max_hold_s=2.0, max_breath_s=1.0), clock=clock)
    obs = SceneObservation(
        ts_mono_s=0.0,
        person_present=True,
        person_conf=0.9,
        zone="right",
        zone_changed=False,
        zone_dwell_s=0.0,
        activity_hint=None,
        event="person_present",
    )
    breath = ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command=BreathCommand(amp_rad=0.06, period_s=6.0, rate_rad_s=1.0),
        confidence=0.4,
        cancel_current=False,
    )
    first = gov.apply(breath, obs)
    second = gov.apply(breath, obs)

    assert first.primitive == PrimitiveKind.BREATH
    assert second.primitive == PrimitiveKind.ORIENT_TO_ZONE
    assert isinstance(second.command, OrientToZoneCommand)
    assert second.command.zone == "right"
    assert second.cancel_current is True

