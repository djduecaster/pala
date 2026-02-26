import time

import pala.main as pala_main
from pala.config.load import LoopRates, LoggingConfig, CameraConfig, RobotConfig, DeepStreamConfig, load_config


class _CaptureLogger:
    def __init__(self):
        self.items = []

    def write(self, obj) -> None:
        self.items.append(obj)

    def close(self) -> None:
        return None


def test_runtime_starts_in_dev_mode(monkeypatch):
    perception_log = _CaptureLogger()
    action_log = _CaptureLogger()

    cfg = RobotConfig(
        mode="dev",
        detector="dummy",
        loop_rates=LoopRates(perception_hz=20, behavior_hz=3, control_hz=80, hardware_hz=120),
        deadman_timeout_ms=250,
        joint_names=["yaw", "pitch1", "pitch2", "roll", "pitch3"],
        joint_limits_rad=[[-1.0, 1.0] for _ in range(5)],
        servo_calibration={},
        logging=LoggingConfig(
            enabled=True,
            perception_jsonl="logs/perception.jsonl",
            actions_jsonl="logs/actions.jsonl",
        ),
        camera=CameraConfig(device="/dev/video0", width=640, height=480, fps=30, pipeline=None),
        deepstream=DeepStreamConfig(config_path=None, person_class_id=0, conf_threshold=None),
    )

    monkeypatch.setattr(pala_main, "load_config", lambda _path: cfg)

    def _logger_stub(path):
        return perception_log if "perception" in (path or "") else action_log

    monkeypatch.setattr(pala_main, "maybe_logger", _logger_stub)
    monkeypatch.setenv("PALA_MAX_RUNTIME_S", "0.5")

    result = pala_main.main()
    assert result == 0
    assert perception_log.items
    assert action_log.items


def test_runtime_starts_with_checked_in_default_config(monkeypatch):
    perception_log = _CaptureLogger()
    action_log = _CaptureLogger()

    def _logger_stub(path):
        return perception_log if "perception" in (path or "") else action_log

    monkeypatch.setattr(pala_main, "maybe_logger", _logger_stub)
    monkeypatch.setenv("PALA_MAX_RUNTIME_S", "0.5")

    cfg = load_config("config/robot.yaml")
    cfg.cosmos.enabled = False
    monkeypatch.setattr(pala_main, "load_config", lambda _path: cfg)

    result = pala_main.main(["--config", "config/robot.yaml"])
    assert result == 0
    assert perception_log.items
    assert action_log.items
