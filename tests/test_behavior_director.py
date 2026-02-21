from __future__ import annotations

from pala.behavior.director import BehaviorDirector, BehaviorDirectorConfig
from pala.behavior.models import SceneObservation
from pala.behavior.scene_memory import SceneMemory
from pala.control.primitives import PrimitiveKind, HoldCommand, GlanceCommand
from pala.types import ActionPlan, BBoxNorm, BreathCommand, PerceptionState


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


def _obs(*, t: float, present: bool, event: str, zone: str | None) -> SceneObservation:
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


def test_scene_memory_tracks_recent_presence_and_zone():
    mem = SceneMemory(recently_seen_s=2.0)
    s1 = mem.update(_obs(t=0.0, present=True, event="person_entered", zone="left"))
    s2 = mem.update(_obs(t=1.0, present=True, event="zone_changed", zone="right"))
    s3 = mem.update(_obs(t=2.5, present=False, event="person_exited", zone=None))
    assert s1.person_recently_seen is True
    assert s2.likely_zone in {"left", "right"}
    assert s2.zone_transitions_recent >= 1
    assert s3.person_recently_seen is True
    assert s3.last_seen_age_s is not None


def test_director_uses_assist_intent_from_remote_explanation():
    clock = _Clock([0.0, 0.1])
    director = BehaviorDirector(clock=clock)
    st = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.7, cy=0.4, w=0.2, h=0.4),
        primary_person_conf=0.9,
    )
    proposed = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.5,
        explanation="assist_with_task user is focused",
        style="focused",
    )
    mem = SceneMemory(recently_seen_s=2.0)
    snap = mem.update(_obs(t=0.0, present=True, event="person_present", zone="right"))
    out = director.realize(proposed, st, snap and _obs(t=0.0, present=True, event="person_present", zone="right"), snap)
    assert out.primitive == PrimitiveKind.GAZE_TO
    assert out.style == "focused"


def test_director_commitment_keeps_mode_stable_without_hard_events():
    clock = _Clock([0.0, 0.2, 0.4, 0.6, 0.8])
    director = BehaviorDirector(
        BehaviorDirectorConfig(
            enable_acknowledge=False,
            mode_track_s=2.0,
        ),
        clock=clock,
    )
    mem = SceneMemory(recently_seen_s=2.0)
    st = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.55, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.9,
    )
    obs = _obs(t=0.0, present=True, event="person_present", zone="center")
    snap = mem.update(obs)

    # First proposed action hints tracking, second tries to idle, director should keep tracking mode.
    a1 = ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command=BreathCommand(amp_rad=0.05, period_s=6.0, rate_rad_s=1.0),
        confidence=0.4,
        explanation="track_transition",
        style="curious",
    )
    a2 = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.3,
        explanation="idle",
        style="calm",
    )
    out1 = director.realize(a1, st, obs, snap)
    out2 = director.realize(a2, st, obs, snap)
    assert out1.primitive in {PrimitiveKind.GAZE_TO, PrimitiveKind.ORIENT_TO_ZONE}
    assert out2.primitive in {PrimitiveKind.GAZE_TO, PrimitiveKind.ORIENT_TO_ZONE}


def test_director_blend_prefers_mode_over_low_conf_remote_hold():
    clock = _Clock([0.0, 0.1, 0.2])
    director = BehaviorDirector(
        BehaviorDirectorConfig(
            enable_acknowledge=False,
            blend_remote_weight=0.1,
        ),
        clock=clock,
    )
    mem = SceneMemory(recently_seen_s=2.0)
    st = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.7, cy=0.45, w=0.2, h=0.4),
        primary_person_conf=0.9,
    )
    obs = _obs(t=0.0, present=True, event="person_present", zone="right")
    snap = mem.update(obs)
    proposed = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.2,
        explanation="low confidence hold",
        style="calm",
    )
    out = director.realize(proposed, st, obs, snap)
    assert out.primitive in {PrimitiveKind.GAZE_TO, PrimitiveKind.ORIENT_TO_ZONE}


def test_director_high_conf_remote_override():
    clock = _Clock([0.0, 0.1])
    director = BehaviorDirector(
        BehaviorDirectorConfig(
            enable_acknowledge=False,
            remote_high_conf_override=0.8,
        ),
        clock=clock,
    )
    mem = SceneMemory(recently_seen_s=2.0)
    st = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.52, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.9,
    )
    obs = _obs(t=0.0, present=True, event="person_present", zone="center")
    snap = mem.update(obs)
    proposed = ActionPlan(
        primitive=PrimitiveKind.GLANCE,
        command=GlanceCommand(direction="left"),
        confidence=0.95,
        explanation="remote high confidence interrupt",
        style="curious",
        cancel_current=True,
    )
    out = director.realize(proposed, st, obs, snap)
    assert out.primitive == PrimitiveKind.GLANCE
