from __future__ import annotations

from pala.config.load import (
    CameraConfig,
    CosmosConfig,
    DeepStreamConfig,
    LoggingConfig,
    LoopRates,
    RobotConfig,
)
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
        deepstream=DeepStreamConfig(config_path=None, person_class_id=0, conf_threshold=None),
        cosmos=CosmosConfig(enabled=cosmos_enabled),
    )


def test_dev_mode_forces_dummy_detector_and_disables_cosmos():
    cfg = _cfg(mode="jetson_full", detector="deepstream", cosmos_enabled=True)
    pala_main._apply_mode_override(cfg, "dev")
    assert cfg.mode == "dev"
    assert cfg.detector == "dummy"
    assert cfg.cosmos.enabled is False


def test_jetson_mode_promotes_dummy_detector_to_deepstream():
    cfg = _cfg(mode="dev", detector="dummy", cosmos_enabled=False)
    pala_main._apply_mode_override(cfg, "jetson_full")
    assert cfg.mode == "jetson_full"
    assert cfg.detector == "deepstream"
