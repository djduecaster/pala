from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from pala.config.load import CameraConfig, CosmosConfig, DeepStreamConfig, LoggingConfig, LoopRates, RobotConfig
import pala.main as pala_main


def _cfg(*, mode: str, detector: str, cosmos_enabled: bool) -> RobotConfig:
    return RobotConfig(
        mode=mode,
        detector=detector,
        loop_rates=LoopRates(perception_hz=20, behavior_hz=3, control_hz=80, hardware_hz=120),
        deadman_timeout_ms=250,
        joint_names=["yaw", "pitch1", "pitch2", "roll", "pitch3"],
        joint_limits_rad=[[-1.0, 1.0] for _ in range(5)],
        servo_calibration={},
        logging=LoggingConfig(enabled=False, perception_jsonl=None, actions_jsonl=None),
        camera=CameraConfig(device="/dev/video0", width=640, height=480, fps=30, pipeline=None),
        deepstream=DeepStreamConfig(config_path="ds.txt", person_class_id=7, conf_threshold=0.25),
        cosmos=CosmosConfig(enabled=cosmos_enabled),
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
    cfg = _cfg(mode="dev", detector="dummy", cosmos_enabled=False)
    servo = pala_main._build_servo(cfg)
    assert servo.__class__.__name__ == "DummyServo"


def test_build_servo_requires_calibration_for_jetson_full():
    cfg = _cfg(mode="jetson_full", detector="deepstream", cosmos_enabled=False)
    cfg.servo_calibration = {}
    with pytest.raises(ValueError, match="servo_calibration is required"):
        pala_main._build_servo(cfg)


def test_build_servo_rejects_channel_length_mismatch():
    cfg = _cfg(mode="jetson_full", detector="deepstream", cosmos_enabled=False)
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

    cfg = _cfg(mode="jetson_full", detector="deepstream", cosmos_enabled=False)
    cfg.servo_calibration = _valid_servo_calibration(cfg.joint_names)

    servo = pala_main._build_servo(cfg)
    assert servo.__class__.__name__ == "_FakeServo"
    calibration = captured["calibration"]
    assert calibration.bus_number == 1
    assert calibration.address == 0x41
    assert calibration.frequency == 60
    assert calibration.channels == [0, 1, 2, 3, 4]
    assert calibration.min_pulse_us == [500.0] * 5
    assert calibration.max_pulse_us == [2500.0] * 5


def test_build_detector_routes_explicit_backends(monkeypatch):
    captured = {}

    class _Dummy:
        pass

    class _Jetson:
        pass

    class _DeepStream:
        def __init__(self, **kwargs):
            captured["deepstream_kwargs"] = kwargs

    monkeypatch.setattr(pala_main, "DummyDetector", _Dummy)
    monkeypatch.setattr(pala_main, "JetsonDetector", _Jetson)
    monkeypatch.setattr(pala_main, "DeepStreamDetector", _DeepStream)

    cfg = _cfg(mode="dev", detector="dummy", cosmos_enabled=False)
    assert isinstance(pala_main._build_detector(cfg), _Dummy)

    cfg.detector = "jetson"
    assert isinstance(pala_main._build_detector(cfg), _Jetson)

    cfg.detector = "deepstream"
    det = pala_main._build_detector(cfg)
    assert isinstance(det, _DeepStream)
    assert captured["deepstream_kwargs"] == {
        "config_path": "ds.txt",
        "person_class_id": 7,
        "conf_threshold": 0.25,
    }


def test_build_detector_rejects_unknown_backend():
    cfg = _cfg(mode="dev", detector="unknown", cosmos_enabled=False)
    with pytest.raises(ValueError, match="Unknown detector backend"):
        pala_main._build_detector(cfg)


def test_build_detector_deepstream_preflight_failure_uses_noop_detector(monkeypatch):
    class _DeepStream:
        def __init__(self, **_kwargs):
            raise AssertionError("DeepStreamDetector should not be constructed when preflight fails")

    monkeypatch.setattr(pala_main, "DeepStreamDetector", _DeepStream)
    monkeypatch.setattr(pala_main, "_deepstream_preflight_error", lambda: "missing_pyds")

    cfg = _cfg(mode="jetson_full", detector="deepstream", cosmos_enabled=False)
    det = pala_main._build_detector(cfg)
    assert det.__class__.__name__ == "_NoopDetector"
    assert det.detect(None) == []


def test_build_detector_uses_mode_default_when_detector_unspecified(monkeypatch):
    class _Dummy:
        pass

    class _Jetson:
        pass

    monkeypatch.setattr(pala_main, "DummyDetector", _Dummy)
    monkeypatch.setattr(pala_main, "JetsonDetector", _Jetson)

    cfg = _cfg(mode="dev", detector="", cosmos_enabled=False)
    assert isinstance(pala_main._build_detector(cfg), _Dummy)

    cfg.mode = "jetson_full"
    assert isinstance(pala_main._build_detector(cfg), _Jetson)


def test_build_frame_source_returns_dummy_for_non_jetson_mode():
    cfg = _cfg(mode="dev", detector="dummy", cosmos_enabled=False)
    src = pala_main._build_frame_source(cfg)
    assert src.__class__.__name__ == "DummyFrameSource"


def test_build_frame_source_jetson_uses_gstreamer_camera(monkeypatch):
    captured = {}

    class _FakeCamera:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def get_frame(self):
            raise AssertionError("not used in this test")

        def shutdown(self):
            return None

    monkeypatch.setitem(sys.modules, "pala.hardware.camera_gst", SimpleNamespace(GStreamerCamera=_FakeCamera))

    cfg = _cfg(mode="jetson_full", detector="deepstream", cosmos_enabled=False)
    src = pala_main._build_frame_source(cfg)
    assert src.__class__.__name__ == "CameraFrameSource"
    assert captured["kwargs"]["device"] == "/dev/video0"
    assert captured["kwargs"]["width"] == 640
    assert captured["kwargs"]["height"] == 480
    assert captured["kwargs"]["fps"] == 30


def test_build_preview_tap_uses_defaults_when_missing_config():
    tap = pala_main._build_preview_tap(SimpleNamespace())
    assert tap._enabled is False
    assert tap._max_width == 640
    assert tap._max_height == 360
