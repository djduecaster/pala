from __future__ import annotations

from types import SimpleNamespace

import pytest

from pala.config.load import (
    CameraConfig,
    CosmosConfig,
    DeepStreamConfig,
    LoggingConfig,
    LoopRates,
    RobotConfig,
)
import pala.main as pala_main
from pala.behavior.policy_v4 import BehaviorPolicyV4Config
from pala.types import HardwareCommand


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
        deepstream=DeepStreamConfig(config_path=None, person_class_id=0, conf_threshold=None),
        cosmos=CosmosConfig(enabled=cosmos_enabled),
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
    assert extra["command"]["timestamp_monotonic_s"] == 1.5


def test_parse_max_runtime_env(monkeypatch):
    monkeypatch.delenv("PALA_MAX_RUNTIME_S", raising=False)
    assert pala_main._parse_max_runtime_s() is None

    monkeypatch.setenv("PALA_MAX_RUNTIME_S", "0.75")
    assert pala_main._parse_max_runtime_s() == 0.75


def test_parse_max_runtime_env_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("PALA_MAX_RUNTIME_S", "not-a-number")
    with pytest.raises(ValueError, match="must be a number"):
        pala_main._parse_max_runtime_s()


def test_apply_mode_override_without_override_promotes_dummy_on_jetson_mode():
    cfg = _cfg(mode="jetson_perception", detector="dummy", cosmos_enabled=True)
    pala_main._apply_mode_override(cfg, None)
    assert cfg.mode == "jetson_perception"
    assert cfg.detector == "deepstream"
    assert cfg.cosmos.enabled is True


def test_apply_mode_override_without_override_keeps_non_dummy_detector():
    cfg = _cfg(mode="jetson_full", detector="deepstream", cosmos_enabled=True)
    pala_main._apply_mode_override(cfg, None)
    assert cfg.mode == "jetson_full"
    assert cfg.detector == "deepstream"
    assert cfg.cosmos.enabled is True


def test_apply_mode_override_dev_without_cosmos_attr():
    cfg = SimpleNamespace(mode="dev", detector="deepstream")
    pala_main._apply_mode_override(cfg, None)
    assert cfg.mode == "dev"
    assert cfg.detector == "dummy"


def test_scope_log_path_uses_run_directory_when_enabled():
    scoped = pala_main._scope_log_path("logs/actions.jsonl", "logs/runs/20260221_200000")
    assert scoped == "logs/runs/20260221_200000/actions.jsonl"


def test_init_run_log_dir_honors_env_flags(monkeypatch, tmp_path):
    cfg = _cfg(mode="jetson_full", detector="deepstream", cosmos_enabled=True)
    cfg.logging.enabled = True
    monkeypatch.setenv("PALA_RUN_SCOPED_LOGS", "1")
    monkeypatch.setenv("PALA_RUN_ID", "run_123")
    monkeypatch.setenv("PALA_RUN_LOG_ROOT", str(tmp_path))

    out = pala_main._init_run_log_dir(cfg)
    assert out == str(tmp_path / "run_123")
    assert (tmp_path / "run_123").exists()


def test_build_behavior_config_returns_v4_config():
    cfg = _cfg(mode="jetson_full", detector="deepstream", cosmos_enabled=True)
    cfg.cosmos.mode_return_home_settle_s = 1.7
    cfg.cosmos.mode_recover_settle_s = 1.3
    cfg.cosmos.mode_boot_timeout_s = 8.4
    cfg.cosmos.action_guard_stale_after_s = 5.5
    cfg.cosmos.action_guard_orient_cooldown_s = 2.1
    cfg.cosmos.action_guard_glance_cooldown_s = 3.2
    cfg.cosmos.action_guard_nod_cooldown_s = 4.4
    cfg.cosmos.action_guard_home_cooldown_s = 5.6
    out = pala_main._build_behavior_config(cfg, run_log_dir=None)
    assert isinstance(out, BehaviorPolicyV4Config)
    assert out.remote_enabled is True
    assert out.model == cfg.cosmos.model
    assert out.startup_wake_enabled is True
    assert out.mode_fsm.return_home_settle_s == pytest.approx(1.7)
    assert out.mode_fsm.recover_settle_s == pytest.approx(1.3)
    assert out.mode_fsm.boot_timeout_s == pytest.approx(8.4)
    assert out.action_guard.stale_after_s == pytest.approx(5.5)
    assert out.action_guard.cooldowns_s["orient_to_zone"] == pytest.approx(2.1)
    assert out.action_guard.cooldowns_s["glance"] == pytest.approx(3.2)
    assert out.action_guard.cooldowns_s["nod"] == pytest.approx(4.4)
    assert out.action_guard.cooldowns_s["home"] == pytest.approx(5.6)
