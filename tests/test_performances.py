"""Manual performance timing, request gating, and calibration regressions."""
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from pala.config import load_config
from pala.control.performances import PerformanceLibrary, PerformanceSequencer
from pala.behavior.manual import ManualBehaviorPolicy
from pala.utils.manual_console import ManualConsole


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def setup():
    cfg = load_config('config/robot.yaml')
    clock = Clock()
    return cfg, PerformanceLibrary('config/performances.json', cfg), clock, PerformanceSequencer(cfg, clock=clock)


def finish(seq, plan, clock, cfg):
    previous = seq.current
    holds = {}
    dt = 1 / cfg.loop_rates.control_hz
    for _ in range(15000):
        clock.now += dt
        cmd, report = seq.tick(plan, dt)
        assert cmd.enable
        assert report.status != 'failed'
        assert cmd.joint_angles_rad[-1] == 0  # weak pitch3 stays fixed
        assert all(lo - 1e-6 <= v <= hi + 1e-6 for v, (lo, hi) in zip(cmd.joint_angles_rad, cfg.joint_limits_rad))
        assert max(abs(a-b) for a, b in zip(previous, cmd.joint_angles_rad)) <= math.radians(max(s.rate_deg_s for s in plan.steps)) * dt + 1e-9
        previous = tuple(cmd.joint_angles_rad)
        if report.phase == 'hold':
            holds.setdefault(report.step, clock.now)
        if report.status == 'complete':
            assert previous == pytest.approx(plan.steps[-1].target_rad, abs=1e-6)
            return report, holds
    pytest.fail('Performance did not finish')


def test_full_interaction_and_repeat_hold(setup):
    cfg, lib, clock, seq = setup
    for name in ('startup', 'greet', 'attend', 'settle', 'demo', 'shutdown'):
        plan = lib.plan(name)
        report, holds = finish(seq, plan, clock, cfg)
        assert len(holds) == len(plan.steps)
        for _ in range(10):
            clock.now += .02
            cmd, again = seq.tick(plan, .02)
            assert again.status == 'complete'
            assert cmd.joint_angles_rad == pytest.approx(report.angles_rad)
    assert seq.current == pytest.approx([0] * 5)


def test_short_pause_is_not_behavior_rate_quantized(setup):
    cfg, lib, clock, seq = setup
    plan = lib.plan('greet')
    # Isolate a quarter-second pause without travel.
    step = replace(plan.steps[1], target_rad=(0.,) * 5)
    plan = replace(plan, steps=(step,))
    report, holds = finish(seq, plan, clock, cfg)
    duration = report.timestamp_monotonic_s - holds[1]
    assert .25 <= duration <= .25 + 1 / cfg.loop_rates.control_hz + 1e-9


def test_shutdown_preempts_from_commanded_position(setup):
    cfg, lib, clock, seq = setup
    plan = lib.plan('greet')
    for _ in range(80):
        clock.now += .0125
        seq.tick(plan, .0125)
    before = seq.current
    with pytest.raises(RuntimeError, match='interrupt'):
        seq.tick(lib.plan('settle'), .0125)
    shutdown = lib.plan('shutdown')
    clock.now += .0125
    cmd, _ = seq.tick(shutdown, .0125)
    assert max(abs(a-b) for a, b in zip(before, cmd.joint_angles_rad)) <= math.radians(8) * .0125 + 1e-9
    finish(seq, shutdown, clock, cfg)


def test_stalled_sequence_fails_and_disables(setup):
    cfg, lib, clock, seq = setup
    plan = lib.plan('greet')
    seq.tick(plan, .01)
    clock.now += 100
    cmd, report = seq.tick(plan, 100)
    assert not cmd.enable
    assert report.status == 'failed'
    cmd, report = seq.tick(plan, .01)
    assert not cmd.enable


@pytest.mark.parametrize('mutation', ['mapping', 'end_pose', 'nan', 'rate', 'pitch_limit'])
def test_reject_invalid_library(setup, tmp_path, mutation):
    cfg, lib, *_ = setup
    raw = json.loads(json.dumps(lib.raw))
    if mutation == 'mapping':
        cfg.servo_calibration['per_joint']['pitch2']['angle_offset'] = 160
    elif mutation == 'end_pose':
        raw['performances']['shutdown']['end_state'] = 'rest'
    elif mutation == 'nan':
        raw['poses_deg']['rest'][1] = float('nan')
    elif mutation == 'rate':
        raw['performances']['greet']['steps'][0]['rate_deg_s'] = 0
    else:
        raw['poses_deg']['attention'][2] = 66
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        PerformanceLibrary(path, cfg)


def test_policy_rejects_busy_duplicate_and_stale_reports(setup):
    cfg, lib, clock, seq = setup
    policy = ManualBehaviorPolicy(lib)
    assert not policy.request('greet')[0]
    startup = policy.step(None)
    report, _ = finish(seq, startup, clock, cfg)
    policy.step(report)
    assert policy.status()['state'] == 'rest'
    assert policy.request('greet')[0]
    assert not policy.request('greet')[0]
    policy.step(report)  # old completion cannot complete the newly accepted request
    assert policy.status()['busy']
    report, _ = finish(seq, policy.step(None), clock, cfg)
    policy.step(report)
    assert not policy.request('greet')[0]
    assert policy.request('settle')[0]
    report, _ = finish(seq, policy.step(None), clock, cfg)
    policy.step(report)
    assert policy.request('greet')[0]
    assert policy.request('shutdown')[0]
    assert not policy.request('greet')[0]
    report, _ = finish(seq, policy.step(None), clock, cfg)
    policy.step(report)
    assert policy.status()['finished']


def test_console_nonblocking_lines_and_eof():
    reader, writer = os.pipe()
    with os.fdopen(reader) as stream:
        console = ManualConsole(stream)
        assert console.poll() == []
        os.write(writer, b'greet\nstat')
        assert console.poll() == ['greet']
        os.write(writer, b'us\nshutdown\n')
        assert console.poll() == ['status', 'shutdown']
        os.close(writer)
        assert console.poll() == ['shutdown']
        assert console.poll() == []


def test_camera_rest_is_shared_by_startup_settling_and_demo(setup):
    cfg, lib, clock, seq = setup
    expected = tuple(math.radians(v) for v in [0, -40, 25, 0, 0])
    assert lib.poses['rest'] == expected
    assert lib.raw['performances']['settle']['steps'][-1]['pose'] == 'rest'
    for name in ('startup', 'settle', 'demo'):
        assert lib.plan(name).steps[-1].target_rad == expected
    assert lib.plan('shutdown').steps[-1].target_rad == (0.,) * 5
