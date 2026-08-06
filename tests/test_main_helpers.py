from __future__ import annotations

from types import SimpleNamespace

import pytest

import pala.main as pala_main
from pala.config.load import CameraConfig, LoggingConfig, LoopRates, RobotConfig
from pala.types import HardwareCommand


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


def test_build_preview_extra_none_command_returns_none():
    cfg = SimpleNamespace(joint_names=["yaw"])
    assert pala_main._build_preview_extra(cfg, None) is None


def test_build_preview_extra_falls_back_joint_names_when_mismatch():
    cfg = SimpleNamespace(joint_names=["yaw"])
    cmd = HardwareCommand(timestamp_monotonic_s=1.5, joint_angles_rad=[0.1, -0.2], enable=False)
    extra = pala_main._build_preview_extra(cfg, cmd)
    assert extra is not None
    assert extra["command"]["joint_names"] == ["joint_0", "joint_1"]
    assert extra["command"]["joint_angles_rad"] == [0.1, -0.2]
    assert extra["command"]["enable"] is False


def test_parse_max_runtime_env(monkeypatch):
    monkeypatch.delenv("PALA_MAX_RUNTIME_S", raising=False)
    assert pala_main._parse_max_runtime_s() is None

    monkeypatch.setenv("PALA_MAX_RUNTIME_S", "0.75")
    assert pala_main._parse_max_runtime_s() == 0.75


def test_parse_max_runtime_env_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("PALA_MAX_RUNTIME_S", "not-a-number")
    with pytest.raises(ValueError, match="must be a number"):
        pala_main._parse_max_runtime_s()


def test_apply_mode_override_only_changes_mode():
    cfg = _cfg(mode="dev")
    pala_main._apply_mode_override(cfg, "jetson_perception")
    assert cfg.mode == "jetson_perception"


def test_scope_log_path_uses_run_directory_when_enabled():
    scoped = pala_main._scope_log_path("logs/actions.jsonl", "logs/runs/20260221_200000")
    assert scoped == "logs/runs/20260221_200000/actions.jsonl"


def test_init_run_log_dir_honors_env_flags(monkeypatch, tmp_path):
    cfg = _cfg(mode="jetson_full")
    cfg.logging.enabled = True
    monkeypatch.setenv("PALA_RUN_SCOPED_LOGS", "1")
    monkeypatch.setenv("PALA_RUN_ID", "run_123")
    monkeypatch.setenv("PALA_RUN_LOG_ROOT", str(tmp_path))

    out = pala_main._init_run_log_dir(cfg)
    assert out == str(tmp_path / "run_123")
    assert (tmp_path / "run_123").exists()
