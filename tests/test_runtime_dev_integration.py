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
    assert any(
        isinstance(entry, dict)
        and "ts_wall_s" in entry
        and "action" in entry
        and getattr(entry["action"], "primitive", None)
        for entry in action_log.items
    )


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


def test_runtime_fails_fast_on_hardware_thread_crash(monkeypatch):
    class _FaultServo:
        def enable(self, _on: bool):
            return None

        def set_angles(self, _angles):
            raise RuntimeError("servo write failed")

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
    monkeypatch.setattr(pala_main, "_build_servo", lambda _cfg: _FaultServo())
    monkeypatch.setenv("PALA_MAX_RUNTIME_S", "1.0")

    result = pala_main.main()
    assert result == 1


def test_main_joins_threads_before_resource_shutdown(monkeypatch):
    order: list[str] = []

    cfg = RobotConfig(
        mode="dev",
        detector="dummy",
        loop_rates=LoopRates(perception_hz=20, behavior_hz=3, control_hz=80, hardware_hz=120),
        deadman_timeout_ms=250,
        joint_names=["yaw", "pitch1", "pitch2", "roll", "pitch3"],
        joint_limits_rad=[[-1.0, 1.0] for _ in range(5)],
        servo_calibration={},
        logging=LoggingConfig(enabled=False, perception_jsonl=None, actions_jsonl=None),
        camera=CameraConfig(device="/dev/video0", width=640, height=480, fps=30, pipeline=None),
        deepstream=DeepStreamConfig(config_path=None, person_class_id=0, conf_threshold=None),
    )
    monkeypatch.setattr(pala_main, "load_config", lambda _path: cfg)
    monkeypatch.setenv("PALA_MAX_RUNTIME_S", "0")

    class _FakePerceptionNode:
        def __init__(self, *args, **kwargs):
            return None

        def step(self):
            raise AssertionError("perception loop should not run in this test")

        def latest_packet(self):
            return None

        def shutdown(self) -> None:
            order.append("perception.shutdown")

    class _FakeBehaviorPolicy:
        def __init__(self, *args, **kwargs):
            return None

        def step(self, _st):
            raise AssertionError("behavior loop should not run in this test")

        def shutdown(self) -> None:
            order.append("behavior.shutdown")

    class _FakeServo:
        def enable(self, _on: bool):
            return None

        def set_angles(self, _angles):
            return None

        def shutdown(self) -> None:
            order.append("servo.shutdown")

    class _FakePreviewTap:
        def write_with_extra(self, *_args, **_kwargs):
            return None

        def close(self) -> None:
            order.append("preview.close")

    class _FakeThread:
        _counter = 0

        def __init__(self, target=None, daemon=None):
            self._target = target
            self._daemon = daemon
            self._alive = False
            self.name = f"fake_thread_{_FakeThread._counter}"
            _FakeThread._counter += 1

        def start(self):
            self._alive = True

        def join(self, timeout=0.0):
            _ = timeout
            order.append(f"{self.name}.join")
            self._alive = False

        def is_alive(self):
            return self._alive

    monkeypatch.setattr(pala_main, "PerceptionNode", _FakePerceptionNode)
    monkeypatch.setattr(pala_main, "BehaviorPolicy", _FakeBehaviorPolicy)
    monkeypatch.setattr(pala_main, "_build_servo", lambda _cfg: _FakeServo())
    monkeypatch.setattr(pala_main, "_build_preview_tap", lambda _cfg: _FakePreviewTap())
    monkeypatch.setattr(pala_main.threading, "Thread", _FakeThread)

    result = pala_main.main()
    assert result == 0

    join_indices = [idx for idx, token in enumerate(order) if token.endswith(".join")]
    shutdown_indices = [idx for idx, token in enumerate(order) if token.endswith(".shutdown") or token.endswith(".close")]
    assert join_indices
    assert shutdown_indices
    assert max(join_indices) < min(shutdown_indices)
