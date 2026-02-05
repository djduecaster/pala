import os
import time

from pala.config.load import LoopRates, LoggingConfig, CameraConfig, RobotConfig, DeepStreamConfig
import pala.main as pala_main
from pala.types import HardwareCommand


class _CaptureLogger:
    def __init__(self):
        self.items = []

    def write(self, obj) -> None:
        self.items.append(obj)

    def close(self) -> None:
        return None


def test_dev_runtime_emits_perception_and_action(monkeypatch):
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

    assert any(getattr(st, "debug", {}).get("zone_hint") for st in perception_log.items)
    assert any(getattr(a, "primitive", None) and a.primitive != "hold" for a in action_log.items)


def test_hardware_respects_enable_false(monkeypatch):
    class _Servo:
        def __init__(self):
            self.enabled_calls = []

        def set_angles(self, _angles):
            return None

        def enable(self, on: bool):
            self.enabled_calls.append(on)

        def shutdown(self):
            return None

    cfg = RobotConfig(
        mode="dev",
        detector="dummy",
        loop_rates=LoopRates(perception_hz=20, behavior_hz=3, control_hz=80, hardware_hz=120),
        deadman_timeout_ms=250,
        joint_names=["yaw", "pitch1", "pitch2", "roll", "pitch3"],
        joint_limits_rad=[[-1.0, 1.0] for _ in range(5)],
        servo_calibration={},
        logging=LoggingConfig(
            enabled=False,
            perception_jsonl=None,
            actions_jsonl=None,
        ),
        camera=CameraConfig(device="/dev/video0", width=640, height=480, fps=30, pipeline=None),
        deepstream=DeepStreamConfig(config_path=None, person_class_id=0, conf_threshold=None),
    )

    monkeypatch.setattr(pala_main, "load_config", lambda _path: cfg)
    servo = _Servo()
    monkeypatch.setattr(pala_main, "_build_servo", lambda _cfg: servo)

    def _step(self, _action, _dt):
        return HardwareCommand(timestamp_monotonic_s=time.monotonic(), joint_angles_rad=[0.0] * 5, enable=False)

    monkeypatch.setattr("pala.control.executor.TrajectoryExecutor.step", _step)
    monkeypatch.setenv("PALA_MAX_RUNTIME_S", "0.3")

    result = pala_main.main()
    assert result == 0
    assert False in servo.enabled_calls
