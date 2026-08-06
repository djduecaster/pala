from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import pala.main as pala_main
from pala.config.load import CameraConfig, LoggingConfig, LoopRates, RobotConfig


def _cfg(*, mode: str) -> RobotConfig:
    return RobotConfig(
        mode=mode,
        loop_rates=LoopRates(perception_hz=20, behavior_hz=3, control_hz=80, hardware_hz=120),
        deadman_timeout_ms=250,
        joint_names=["yaw", "pitch1", "pitch2", "roll", "pitch3"],
        joint_limits_rad=[[-1.0, 1.0] for _ in range(5)],
        servo_calibration={},
        logging=LoggingConfig(enabled=False, perception_jsonl=None, actions_jsonl=None),
        camera=CameraConfig(device="/dev/video0", width=640, height=480, fps=30, pipeline=None),
    )


def _valid_servo_calibration(joint_names: list[str]) -> dict:
    return {
        "bus_number": 1,
        "address": 0x41,
        "frequency": 60,
        "channels": list(range(len(joint_names))),
        "per_joint": {
            name: {
                "min_pulse": 500.0,
                "max_pulse": 2500.0,
                "angle_scale": 1.0,
                "angle_offset": 0.0,
                "reverse": False,
            }
            for name in joint_names
        },
    }


def test_build_servo_returns_dummy_for_non_jetson_mode():
    assert pala_main._build_servo(_cfg(mode="dev")).__class__.__name__ == "DummyServo"


def test_build_servo_requires_calibration_for_jetson_full():
    cfg = _cfg(mode="jetson_full")
    with pytest.raises(ValueError, match="servo_calibration is required"):
        pala_main._build_servo(cfg)


def test_build_servo_rejects_channel_length_mismatch():
    cfg = _cfg(mode="jetson_full")
    cfg.servo_calibration = _valid_servo_calibration(cfg.joint_names)
    cfg.servo_calibration["channels"] = [0, 1]
    with pytest.raises(ValueError, match="channels must match joint_names length"):
        pala_main._build_servo(cfg)


def test_build_servo_maps_calibration_to_backend(monkeypatch):
    captured = {}

    class _FakeServo:
        def __init__(self, calibration):
            captured["calibration"] = calibration

    monkeypatch.setattr(pala_main, "PCA9685Servo", _FakeServo)
    cfg = _cfg(mode="jetson_full")
    cfg.servo_calibration = _valid_servo_calibration(cfg.joint_names)

    pala_main._build_servo(cfg)
    calibration = captured["calibration"]
    assert calibration.bus_number == 1
    assert calibration.channels == [0, 1, 2, 3, 4]


def test_build_frame_source_returns_dummy_for_dev_mode():
    assert pala_main._build_frame_source(_cfg(mode="dev")).__class__.__name__ == "DummyFrameSource"


@pytest.mark.parametrize("mode", ["jetson_perception", "jetson_full"])
def test_build_frame_source_jetson_modes_use_gstreamer_camera(monkeypatch, mode):
    captured = {}

    class _FakeCamera:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def shutdown(self):
            return None

    monkeypatch.setitem(sys.modules, "pala.hardware.camera_gst", SimpleNamespace(GStreamerCamera=_FakeCamera))

    source = pala_main._build_frame_source(_cfg(mode=mode))
    assert source.__class__.__name__ == "CameraFrameSource"
    assert captured["kwargs"]["device"] == "/dev/video0"


def test_build_preview_tap_uses_defaults_when_missing_config():
    tap = pala_main._build_preview_tap(SimpleNamespace())
    assert tap._enabled is False
    assert tap._max_width == 640
    assert tap._max_height == 360
