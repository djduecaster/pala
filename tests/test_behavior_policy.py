from __future__ import annotations

from dataclasses import dataclass

from pala.behavior.policy import BehaviorPolicy
from pala.types import PerceptionState, BBoxNorm, ActionPlan
from pala.control.primitives import PrimitiveKind, BreathCommand, HoldCommand


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


@dataclass
class _InterruptPlanner:
    calls: int = 0

    def plan(self, _st: PerceptionState) -> ActionPlan:
        self.calls += 1
        return ActionPlan(
            primitive=PrimitiveKind.BREATH if self.calls == 1 else PrimitiveKind.HOLD,
            command=BreathCommand(amp_rad=0.05, period_s=6.0, rate_rad_s=1.0) if self.calls == 1 else HoldCommand(),
            confidence=0.2,
            cancel_current=False,
        )


class _Time:
    def __init__(self, values):
        self._values = list(values)

    def __call__(self):
        if not self._values:
            raise RuntimeError("monotonic called too many times in test")
        return self._values.pop(0)


def test_behavior_zone_dwell(monkeypatch):
    fake_time = _Time([0.0, 0.1, 1.2, 1.3, 1.8])
    monkeypatch.setattr("pala.behavior.policy.time.monotonic", fake_time)

    planner = _Planner()
    policy = BehaviorPolicy(planner=planner, dwell_s=1.0, cooldown_s=0.5)
    st = PerceptionState(timestamp_monotonic_s=0.0, primary_person=BBoxNorm(cx=0.1, cy=0.5, w=0.2, h=0.4))

    action = policy.step(st)
    assert action.primitive == PrimitiveKind.BREATH

    action = policy.step(st)
    assert action.primitive == PrimitiveKind.GLANCE
    assert action.command.direction == "left"


def test_behavior_respects_non_interrupting_planner_action(monkeypatch):
    fake_time = _Time([0.0, 0.1, 0.2])
    monkeypatch.setattr("pala.behavior.policy.time.monotonic", fake_time)

    planner = _InterruptPlanner()
    policy = BehaviorPolicy(planner=planner, dwell_s=10.0, cooldown_s=10.0)
    st = PerceptionState(timestamp_monotonic_s=0.0, primary_person=BBoxNorm(cx=0.5, cy=0.5, w=0.2, h=0.4))

    first = policy.step(st)
    second = policy.step(st)
    assert first.cancel_current is False
    assert second.cancel_current is False
