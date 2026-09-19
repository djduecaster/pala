import math

from tools.primitive_sim.social_scene import build_trace


def test_desk_scene_is_bounded_and_returns_to_rest():
    trace = build_trace()
    samples = trace['samples']
    assert {s['scene']['id'] for s in samples} == {'rest', 'notice', 'greet', 'excited', 'settle', 'breathing', 'breathing_final', 'point_left', 'point_right'}
    assert not {s['status'] for s in samples} & {'rejected', 'timed_out', 'canceled'}
    for sample in samples:
        assert sample['joint_angles_rad'][4] == 0
        assert all(lo <= v <= hi for v, (lo, hi) in zip(sample['joint_angles_rad'], trace['metadata']['joint_limits_rad']))
    assert all(abs(a - math.radians(b)) < 1e-6 for a, b in zip(samples[-1]['joint_angles_rad'], [0, -40, 25, 0, 0]))
    notice = [s for s in samples if s['scene']['id'] == 'notice']
    assert max(abs(s['joint_angles_rad'][3]) for s in notice) <= math.radians(3) + 1e-6
