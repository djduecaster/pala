from __future__ import annotations

from dataclasses import dataclass

from pala.behavior.policy import BehaviorPolicy
from pala.types import PerceptionState, BBoxNorm, ActionPlan


@dataclass
class _Planner:
    last: ActionPlan | None = None

    def plan(self, _st: PerceptionState) -> ActionPlan:
        self.last = ActionPlan(primitive="planner_idle", params={}, confidence=0.1)
        return self.last


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
    assert action.primitive == "planner_idle"

    action = policy.step(st)
    assert action.primitive == "glance_left"

    action = policy.step(st)
    assert action.primitive == "planner_idle"

    action = policy.step(st)
    assert action.primitive == "glance_left"
