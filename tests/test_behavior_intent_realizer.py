from __future__ import annotations

from pala.behavior.action_realizer import ActionRealizer
from pala.behavior.intent_planner import IntentPlanner
from pala.behavior.intents import BehaviorIntent
from pala.behavior.models import SceneObservation
from pala.behavior.scene_memory import SceneMemory
from pala.control.primitives import PrimitiveKind, HoldCommand, GlanceCommand, OrientToZoneCommand
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


def _obs(t: float, present: bool, event: str, zone: str | None) -> SceneObservation:
    return SceneObservation(
        ts_mono_s=t,
        person_present=present,
        person_conf=0.9 if present else None,
        zone=zone,
        zone_changed=(event == "zone_changed"),
        zone_dwell_s=0.0,
        activity_hint=None,
        event=event,
    )


def test_intent_planner_selects_reacquire_when_person_recently_lost():
    clock = _Clock([0.0, 0.5, 1.0])
    planner = IntentPlanner(clock=clock)
    mem = SceneMemory(recently_seen_s=2.0)
    mem.update(_obs(0.0, True, "person_present", "left"))
    snap = mem.update(_obs(1.0, False, "person_exited", None))
    proposed = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.5,
        explanation="remote uncertain",
        style="calm",
    )
    intent = planner.plan(proposed=proposed, obs=_obs(1.0, False, "person_exited", None), memory=snap)
    assert intent.mode == "reacquire"


def test_intent_planner_does_not_collapse_to_idle_on_hold_when_person_present():
    clock = _Clock([0.0, 0.5])
    planner = IntentPlanner(clock=clock)
    mem = SceneMemory(recently_seen_s=2.0)
    snap = mem.update(_obs(0.0, True, "person_present", "center"))
    proposed = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.9,
        explanation="remote calm hold",
        style="calm",
    )
    intent = planner.plan(proposed=proposed, obs=_obs(0.0, True, "person_present", "center"), memory=snap)
    assert intent.mode in {"engage", "track", "assist"}


def test_intent_planner_accepts_explicit_remote_mode_override():
    clock = _Clock([0.0, 0.1])
    planner = IntentPlanner(clock=clock)
    mem = SceneMemory(recently_seen_s=2.0)
    snap = mem.update(_obs(0.0, True, "person_present", "center"))
    proposed = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.95,
        explanation="target_state=reacquire reason=test",
        style="curious",
    )
    intent = planner.plan(proposed=proposed, obs=_obs(0.0, True, "person_present", "center"), memory=snap)
    assert intent.mode == "reacquire"


def test_intent_planner_prefers_explicit_remote_target_zone():
    clock = _Clock([0.0, 0.1])
    planner = IntentPlanner(clock=clock)
    mem = SceneMemory(recently_seen_s=2.0)
    snap = mem.update(_obs(0.0, True, "person_present", "center"))
    proposed = ActionPlan(
        primitive=PrimitiveKind.ORIENT_TO_ZONE,
        command=OrientToZoneCommand(zone="left"),
        confidence=0.85,
        explanation="target_state=tracking",
        style="curious",
    )
    intent = planner.plan(proposed=proposed, obs=_obs(0.0, True, "person_present", "center"), memory=snap)
    assert intent.target_zone == "left"


def test_action_realizer_runs_reacquire_script():
    clock = _Clock([0.0, 0.7, 1.5, 2.3])
    realizer = ActionRealizer(clock=clock)
    intent = BehaviorIntent(mode="reacquire", target_zone="left", confidence=0.6, reason="test", allow_interrupt=True)
    proposed = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.4,
        explanation="remote hold",
        style="calm",
    )
    mem = SceneMemory(recently_seen_s=2.0)
    snap = mem.update(_obs(0.0, False, "person_exited", None))
    a = realizer.realize(intent=intent, proposed=proposed, st=None, obs=_obs(0.0, False, "person_exited", None), memory=snap)
    b = realizer.realize(intent=intent, proposed=proposed, st=None, obs=_obs(0.7, False, "no_person", None), memory=snap)
    assert a.primitive == PrimitiveKind.GLANCE
    assert b.primitive == PrimitiveKind.GLANCE
    assert isinstance(a.command, GlanceCommand)


def test_action_realizer_gaze_deadband_holds_when_target_stable():
    clock = _Clock([0.0, 0.1, 0.2, 0.3])
    realizer = ActionRealizer(clock=clock)
    intent = BehaviorIntent(mode="track", target_zone="center", confidence=0.7, reason="track", allow_interrupt=True)
    proposed = ActionPlan(
        primitive=PrimitiveKind.ORIENT_TO_ZONE,
        command=OrientToZoneCommand(zone="center"),
        confidence=0.6,
        explanation="track",
        style="curious",
    )
    st = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.52, cy=0.49, w=0.2, h=0.4),
        primary_person_conf=0.9,
    )
    mem = SceneMemory(recently_seen_s=2.0)
    snap = mem.update(_obs(0.0, True, "person_present", "center"))
    first = realizer.realize(intent=intent, proposed=proposed, st=st, obs=_obs(0.0, True, "person_present", "center"), memory=snap)
    second = realizer.realize(intent=intent, proposed=proposed, st=st, obs=_obs(0.1, True, "person_present", "center"), memory=snap)
    assert first.primitive == PrimitiveKind.GAZE_TO
    assert second.primitive in {PrimitiveKind.ORIENT_TO_ZONE, PrimitiveKind.GAZE_TO}


def test_action_realizer_emits_reaffirm_motion_when_tracking_stalls():
    clock = _Clock([0.0, 1.0, 2.4])
    realizer = ActionRealizer(clock=clock)
    intent = BehaviorIntent(mode="track", target_zone="center", confidence=0.7, reason="track", allow_interrupt=True)
    proposed = ActionPlan(
        primitive=PrimitiveKind.ORIENT_TO_ZONE,
        command=OrientToZoneCommand(zone="center"),
        confidence=0.6,
        explanation="track",
        style="curious",
    )
    st = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.5, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.9,
    )
    mem = SceneMemory(recently_seen_s=2.0)
    snap = mem.update(_obs(0.0, True, "person_present", "center"))
    _ = realizer.realize(intent=intent, proposed=proposed, st=st, obs=_obs(0.0, True, "person_present", "center"), memory=snap)
    _ = realizer.realize(intent=intent, proposed=proposed, st=st, obs=_obs(1.0, True, "person_present", "center"), memory=snap)
    third = realizer.realize(intent=intent, proposed=proposed, st=st, obs=_obs(2.4, True, "person_present", "center"), memory=snap)
    assert third.primitive in {PrimitiveKind.GLANCE, PrimitiveKind.ORIENT_TO_ZONE, PrimitiveKind.GAZE_TO}


def test_action_realizer_passthroughs_high_conf_remote_once_then_breaks_repetition():
    clock = _Clock([0.0, 0.2, 0.4])
    realizer = ActionRealizer(clock=clock)
    intent = BehaviorIntent(
        mode="track",
        target_zone="left",
        confidence=0.8,
        reason="track",
        allow_interrupt=True,
        urgency="medium",
        act_now=True,
    )
    proposed = ActionPlan(
        primitive=PrimitiveKind.GLANCE,
        command=GlanceCommand(direction="left", duration_s=0.4),
        confidence=0.92,
        explanation="target_state=tracking remote action",
        style="curious",
        cancel_current=True,
    )
    st = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.7, cy=0.45, w=0.2, h=0.4),
        primary_person_conf=0.9,
    )
    mem = SceneMemory(recently_seen_s=2.0)
    snap = mem.update(_obs(0.0, True, "person_present", "right"))
    first = realizer.realize(intent=intent, proposed=proposed, st=st, obs=_obs(0.0, True, "person_present", "right"), memory=snap)
    second = realizer.realize(intent=intent, proposed=proposed, st=st, obs=_obs(0.2, True, "person_present", "right"), memory=snap)
    assert first.primitive == PrimitiveKind.GLANCE
    assert second.primitive in {PrimitiveKind.GAZE_TO, PrimitiveKind.ORIENT_TO_ZONE, PrimitiveKind.HOLD, PrimitiveKind.GLANCE}
    assert not (second.primitive == PrimitiveKind.GLANCE and second.explanation == first.explanation)
