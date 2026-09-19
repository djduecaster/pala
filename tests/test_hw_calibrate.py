"""Calibration guards must reject bad commands before any servo write."""
import math
import pytest
from pala.config import load_config
from tools import hw_calibrate as cal


class FakeServo:
    def __init__(self):
        self.writes = []

    def set_angles(self, values):
        self.writes.append(list(values))


@pytest.mark.parametrize('smoothing', [True, False])
@pytest.mark.parametrize('value', [2.0, -2.0, float('nan'), float('inf')])
def test_invalid_target_never_writes_or_changes_estimate(smoothing, value):
    servo = FakeServo()
    current = [0.0, 0.0]
    with pytest.raises(ValueError):
        cal._apply_command(servo, current, [0.1, value],
                           joint_limits_rad=[[-1, 1], [-1, 1]],
                           smoothing=smoothing, slew_rate_deg_s=5, interp_dt_s=.02)
    assert servo.writes == []
    assert current == [0.0, 0.0]


@pytest.mark.parametrize('smoothing', [True, False])
def test_valid_boundary_reaches_target_within_limits(smoothing, monkeypatch):
    monkeypatch.setattr(cal.time, 'sleep', lambda _: None)
    servo = FakeServo()
    current = [0.0]
    cal._apply_command(servo, current, [1.0], joint_limits_rad=[[-1, 1]],
                       smoothing=smoothing, slew_rate_deg_s=100, interp_dt_s=.02)
    assert servo.writes[-1] == [1.0]
    assert all(-1 <= frame[0] <= 1 for frame in servo.writes)


@pytest.mark.parametrize('args', [
    ['--joint', 'pitch2', '--deg', '66'],
    ['--joint', 'pitch2', '--rad', '-1'],
    ['--joint', 'yaw', '--deg', 'nan'],
    ['--neutral', '--joint', 'pitch2', '--deg', '100'],
    ['--repl', '--slew-rate-deg-s', 'nan'],
])
def test_invalid_cli_rejected_before_hardware_creation(monkeypatch, args):
    monkeypatch.setattr('sys.argv', ['hw_calibrate', '--enable', *args])
    monkeypatch.setattr(cal, '_build_servo', lambda _: pytest.fail('hardware constructed'))
    with pytest.raises(SystemExit) as exc:
        cal.main()
    assert exc.value.code == 2


def test_repl_rejects_bad_target_and_accepts_next_command(monkeypatch):
    cfg = load_config('config/robot.yaml')
    servo = FakeServo()
    commands = iter(['pitch2 66', 'yaw nan', 'pitch2 10', 'q'])
    monkeypatch.setattr('builtins.input', lambda _: next(commands))
    current = [0.0] * 5
    assert cal._run_repl(servo, cfg.joint_names, cfg.servo_calibration['per_joint'],
                         current, joint_limits_rad=cfg.joint_limits_rad,
                         smoothing=False, slew_rate_deg_s=5, interp_dt_s=.02) == 0
    assert len(servo.writes) == 1
    assert servo.writes[0][2] == pytest.approx(math.radians(10))
