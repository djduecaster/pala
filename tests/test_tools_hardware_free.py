from __future__ import annotations

import sys

import pytest

from pala.config.load import CameraConfig, LoggingConfig, LoopRates, RobotConfig


def _jetson_config() -> RobotConfig:
    return RobotConfig(
        mode="jetson_full",
        loop_rates=LoopRates(20.0, 3.0, 80.0, 120.0),
        deadman_timeout_ms=250,
        joint_names=["yaw", "pitch1", "pitch2", "roll", "pitch3"],
        joint_limits_rad=[[-1.0, 1.0] for _ in range(5)],
        servo_calibration={},
        logging=LoggingConfig(False, None, None),
        camera=CameraConfig("/dev/video0", 640, 480, 30, None),
    )


def test_expressive_dry_run_never_builds_servo(monkeypatch):
    import tools.expressive_movement_demo as demo

    cfg = _jetson_config()
    monkeypatch.setattr(demo, "load_config", lambda _path: cfg)
    monkeypatch.setattr(demo, "build_demo_segments", lambda _cfg, _args: [])
    monkeypatch.setattr(demo, "_build_servo", lambda _cfg: pytest.fail("real servo builder called"))
    assert demo.main(["--dry-run", "--no-neutral-start", "--no-neutral-end"]) == 0


def test_primitive_validator_dry_run_never_builds_servo(monkeypatch):
    import tools.validate_primitives as validator

    cfg = _jetson_config()
    monkeypatch.setattr(validator, "load_config", lambda _path: cfg)
    monkeypatch.setattr(validator, "_suite_segments", lambda _args, _cfg: [])
    monkeypatch.setattr(validator, "_build_servo", lambda _cfg: pytest.fail("real servo builder called"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_primitives", "--dry-run", "--no-log", "--no-neutral-start", "--no-neutral-end"],
    )
    assert validator.main() == 0
