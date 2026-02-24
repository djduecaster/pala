from __future__ import annotations

import math
import sys
import types

import pytest

from pala.hardware.servo import DummyServo, PCA9685Servo, ServoCalibration, ServoInterface


def test_servo_interface_methods_raise_not_implemented():
    base = ServoInterface()
    with pytest.raises(NotImplementedError):
        base.set_angles([0.1])
    with pytest.raises(NotImplementedError):
        base.enable(True)
    with pytest.raises(NotImplementedError):
        base.shutdown()


def test_dummy_servo_logs_only_on_state_change(monkeypatch):
    seen = []
    monkeypatch.setattr("pala.hardware.servo.logger.info", lambda msg, *args: seen.append(msg % args if args else msg))
    monkeypatch.setattr("pala.hardware.servo.time.sleep", lambda _s: None)

    servo = DummyServo(log_every=2)
    servo.set_angles([0.1])
    servo.set_angles([0.2])  # logs every second call
    servo.enable(True)  # already enabled; should not log
    servo.enable(False)  # transition; should log
    servo.shutdown()  # already disabled; should not log again

    assert any("dummy servo angles(rad)" in line for line in seen)
    assert seen.count("dummy servo disabled") == 1


def test_pca9685_servo_wrapper_forwards_config_and_converts_radians(monkeypatch):
    captured = {}

    class _FakeBackend:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            captured["angles"] = None
            captured["enabled"] = []
            captured["shutdown"] = False

        def set_angles_deg(self, angles_deg):
            captured["angles"] = list(angles_deg)

        def enable(self, on):
            captured["enabled"].append(on)

        def shutdown(self):
            captured["shutdown"] = True

    fake_module = types.SimpleNamespace(PCA9685Servo=_FakeBackend)
    monkeypatch.setitem(sys.modules, "pala.hardware.servo_pca9685", fake_module)

    calibration = ServoCalibration(
        bus_number=1,
        address=0x40,
        frequency=50,
        channels=[0, 1],
        min_pulse_us=[500.0, 500.0],
        max_pulse_us=[2500.0, 2500.0],
        angle_scales=[1.0, 1.0],
        angle_offsets_deg=[0.0, 0.0],
        reverses=[False, True],
    )

    servo = PCA9685Servo(calibration)
    servo.set_angles([0.0, math.pi / 2.0])
    servo.enable(False)
    servo.shutdown()

    assert captured["init"]["bus_number"] == 1
    assert captured["init"]["address"] == 0x40
    assert captured["init"]["frequency_hz"] == 50
    assert captured["init"]["channels"] == [0, 1]
    assert captured["angles"] == [0.0, 90.0]
    assert captured["enabled"] == [False]
    assert captured["shutdown"] is True
