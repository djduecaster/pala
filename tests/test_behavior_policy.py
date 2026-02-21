from __future__ import annotations

from dataclasses import dataclass

from pala.behavior.policy import BehaviorPolicy
from pala.types import PerceptionState, BBoxNorm, ActionPlan
from pala.control.primitives import PrimitiveKind, BreathCommand, GlanceCommand, HoldCommand
from pala.types import OrientToZoneCommand


@dataclass
class _Planner:
    last: ActionPlan | None = None

    def plan(self, _st: PerceptionState) -> ActionPlan:
        self.last = ActionPlan(
            primitive=PrimitiveKind.BREATH,
            command=BreathCommand(amp_rad=0.05, period_s=6.0, rate_rad_s=1.0),
            confidence=0.1,
        )
        return self.last


class _Time:
    def __init__(self, values):
        self._values = list(values)
        self._last = float(self._values[-1]) if self._values else 0.0

    def __call__(self):
        if self._values:
            self._last = float(self._values.pop(0))
            return self._last
        self._last += 0.1
        return self._last


@dataclass
class _RemoteOwnedPlanner:
    owns_semantic_behavior: bool = True
    calls: int = 0

    def plan(self, _st: PerceptionState) -> ActionPlan:
        self.calls += 1
        return ActionPlan(
            primitive=PrimitiveKind.BREATH,
            command=BreathCommand(amp_rad=0.05, period_s=6.0, rate_rad_s=1.0),
            confidence=0.5,
        )


@dataclass
class _RemoteHoldPlanner:
    owns_semantic_behavior: bool = True

    def plan(self, _st: PerceptionState) -> ActionPlan:
        return ActionPlan(
            primitive=PrimitiveKind.HOLD,
            command=HoldCommand(),
            confidence=0.4,
            cancel_current=False,
        )


@dataclass
class _RemoteBreathPlanner:
    owns_semantic_behavior: bool = True

    def plan(self, _st: PerceptionState) -> ActionPlan:
        return ActionPlan(
            primitive=PrimitiveKind.BREATH,
            command=BreathCommand(amp_rad=0.05, period_s=6.0, rate_rad_s=1.0),
            confidence=0.4,
            cancel_current=False,
            style="curious",
        )


@dataclass
class _HoldThenOrientPlanner:
    owns_semantic_behavior: bool = True
    calls: int = 0

    def plan(self, _st: PerceptionState) -> ActionPlan:
        self.calls += 1
        if self.calls == 1:
            return ActionPlan(
                primitive=PrimitiveKind.HOLD,
                command=HoldCommand(),
                confidence=0.4,
                cancel_current=False,
            )
        return ActionPlan(
            primitive=PrimitiveKind.ORIENT_TO_ZONE,
            command=OrientToZoneCommand(zone="left"),
            confidence=0.7,
            cancel_current=False,
        )


def test_behavior_passthrough_when_director_disabled():
    fake_time = _Time([0.0, 0.1, 0.2, 0.3, 1.3, 1.4, 1.5])
    planner = _Planner()
    policy = BehaviorPolicy(
        planner=planner,
        dwell_s=1.0,
        cooldown_s=0.5,
        director_enabled=False,
        fusion_enable_ack=False,
        fusion_enable_gaze=False,
        fusion_enable_reacquire=False,
        clock=fake_time,
    )
    st = PerceptionState(timestamp_monotonic_s=0.0, primary_person=BBoxNorm(cx=0.1, cy=0.5, w=0.2, h=0.4))

    action = policy.step(st)
    assert action.primitive == PrimitiveKind.BREATH

    action = policy.step(st)
    assert action.primitive == PrimitiveKind.BREATH


def test_behavior_forces_interrupt_when_switching_off_persistent_action():
    fake_time = _Time([0.0, 0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    planner = _HoldThenOrientPlanner()
    policy = BehaviorPolicy(
        planner=planner,
        dwell_s=10.0,
        cooldown_s=10.0,
        director_enabled=False,
        fusion_enable_ack=False,
        fusion_enable_gaze=False,
        fusion_enable_reacquire=False,
        clock=fake_time,
    )
    st = PerceptionState(timestamp_monotonic_s=0.0, primary_person=BBoxNorm(cx=0.5, cy=0.5, w=0.2, h=0.4))

    first = policy.step(st)
    second = policy.step(st)
    assert first.cancel_current is False
    assert second.cancel_current is True
    assert second.primitive == PrimitiveKind.ORIENT_TO_ZONE


def test_behavior_skips_dwell_semantics_when_planner_owns_semantics():
    fake_time = _Time([0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2])
    planner = _RemoteOwnedPlanner()
    policy = BehaviorPolicy(
        planner=planner,
        dwell_s=1.0,
        cooldown_s=0.5,
        director_enabled=False,
        fusion_enable_ack=False,
        fusion_enable_gaze=False,
        fusion_enable_reacquire=False,
        clock=fake_time,
    )
    st = PerceptionState(timestamp_monotonic_s=0.0, primary_person=BBoxNorm(cx=0.1, cy=0.5, w=0.2, h=0.4))

    first = policy.step(st)
    second = policy.step(st)
    assert first.primitive == PrimitiveKind.BREATH
    assert second.primitive == PrimitiveKind.BREATH
    assert planner.calls == 2


def test_behavior_repeated_signature_reuses_action_id_for_persistent_primitives():
    planner = _Planner()
    policy = BehaviorPolicy(planner=planner, dwell_s=1.0, cooldown_s=0.5)

    first = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.8,
        style="curious",
        cancel_current=False,
    )
    second = ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.8,
        style="curious",
        cancel_current=False,
    )

    out1 = policy._arbitrate(first)
    out2 = policy._arbitrate(second)
    assert out1.action_id == out2.action_id
    assert out2.cancel_current is False


def test_behavior_repeated_signature_refreshes_action_id_for_terminating_primitives():
    planner = _Planner()
    policy = BehaviorPolicy(planner=planner, dwell_s=1.0, cooldown_s=0.5)

    first = ActionPlan(
        primitive=PrimitiveKind.GLANCE,
        command=GlanceCommand(direction="left", duration_s=0.4),
        confidence=0.8,
        style="curious",
        cancel_current=True,
    )
    second = ActionPlan(
        primitive=PrimitiveKind.GLANCE,
        command=GlanceCommand(direction="left", duration_s=0.4),
        confidence=0.8,
        style="curious",
        cancel_current=True,
    )

    out1 = policy._arbitrate(first)
    out2 = policy._arbitrate(second)
    assert out1.action_id != out2.action_id


def test_behavior_refreshes_after_hold_collapse():
    fake_time = _Time([0.0, 0.1, 0.2, 3.0, 3.1, 3.2, 3.3, 6.0, 6.1, 6.2])
    planner = _RemoteHoldPlanner()
    policy = BehaviorPolicy(
        planner=planner,
        dwell_s=10.0,
        cooldown_s=10.0,
        max_hold_s=1.0,
        director_enabled=False,
        fusion_enable_ack=False,
        fusion_enable_gaze=False,
        fusion_enable_reacquire=False,
        clock=fake_time,
    )
    st = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.8, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.9,
    )

    first = policy.step(st)
    second = policy.step(st)
    assert first.primitive == PrimitiveKind.HOLD
    assert second.primitive == PrimitiveKind.ORIENT_TO_ZONE
    assert second.cancel_current is True
    assert isinstance(second.command, OrientToZoneCommand)
    assert second.command.zone == "right"


def test_behavior_zone_change_nudges_tracking():
    fake_time = _Time([0.0, 0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    planner = _RemoteHoldPlanner()
    policy = BehaviorPolicy(
        planner=planner,
        dwell_s=10.0,
        cooldown_s=10.0,
        fusion_enable_ack=False,
        fusion_enable_gaze=False,
        fusion_enable_reacquire=False,
        clock=fake_time,
    )
    left = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.1, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.9,
        debug={"zone_hint": "left"},
    )
    right = PerceptionState(
        timestamp_monotonic_s=0.4,
        primary_person=BBoxNorm(cx=0.9, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.9,
        debug={"zone_hint": "right"},
    )

    policy.step(left)
    out = policy.step(right)
    assert out.primitive == PrimitiveKind.ORIENT_TO_ZONE
    assert out.cancel_current is True
    assert isinstance(out.command, OrientToZoneCommand)
    assert out.command.zone == "right"


def test_behavior_remote_owned_path_bypasses_local_mode_realizer():
    fake_time = _Time([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    planner = _RemoteBreathPlanner()
    policy = BehaviorPolicy(
        planner=planner,
        fusion_enable_ack=False,
        fusion_enable_gaze=True,
        fusion_enable_reacquire=False,
        clock=fake_time,
    )
    st = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.85, cy=0.35, w=0.2, h=0.4),
        primary_person_conf=0.9,
    )
    out = policy.step(st)
    assert out.primitive == PrimitiveKind.BREATH
    assert out.cancel_current is False


def test_behavior_remote_owned_path_keeps_remote_action_on_exit():
    fake_time = _Time([0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.5, 1.6, 1.7])
    planner = _RemoteBreathPlanner()
    policy = BehaviorPolicy(
        planner=planner,
        fusion_enable_ack=False,
        fusion_enable_gaze=False,
        fusion_enable_reacquire=True,
        clock=fake_time,
    )
    present = PerceptionState(
        timestamp_monotonic_s=0.0,
        primary_person=BBoxNorm(cx=0.1, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.9,
        debug={"zone_hint": "left"},
    )
    absent = PerceptionState(
        timestamp_monotonic_s=1.0,
        primary_person=None,
        primary_person_conf=None,
    )
    policy.step(present)
    out = policy.step(absent)
    assert out.primitive == PrimitiveKind.BREATH
    assert out.cancel_current is False
