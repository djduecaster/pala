from pala.behavior import HoldBehaviorPolicy
from pala.types import PerceptionState, PrimitiveKind


def test_reset_policy_preserves_one_hold_action():
    policy = HoldBehaviorPolicy()
    state = PerceptionState(timestamp_monotonic_s=1.0, frame_id=1, is_new_frame=True)

    first = policy.step(state)
    second = policy.step(state)

    assert first is second
    assert first.primitive == PrimitiveKind.HOLD
    assert first.explanation == "behavior_reset_hold"
