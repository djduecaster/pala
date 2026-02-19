import pytest

from pala.config.load import load_config


def test_config_loads_and_validates(tmp_path):
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: dev",
                "loop_rates:",
                "  perception_hz: 20",
                "  behavior_hz: 3",
                "  control_hz: 80",
                "  hardware_hz: 120",
                "deadman_timeout_ms: 250",
                "joint_names: [yaw, pitch1, pitch2, roll, pitch3]",
                "joint_limits_rad:",
                "  - [-1.0, 1.0]",
                "  - [-1.0, 1.0]",
                "  - [-1.0, 1.0]",
                "  - [-1.0, 1.0]",
                "  - [-1.0, 1.0]",
                "servo_calibration: {}",
                "logging:",
                "  enabled: false",
                "  perception_jsonl: logs/perception.jsonl",
                "  actions_jsonl: logs/actions.jsonl",
            ]
        )
    )

    cfg = load_config(str(config_path))
    assert cfg.mode == "dev"
    assert cfg.loop_rates.perception_hz == 20
    assert cfg.logging.enabled is False

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("deadman_timeout_ms: 250\n")
    with pytest.raises(ValueError, match="loop_rates"):
        load_config(str(invalid_path))


def test_config_bool_string_false_is_parsed_false(tmp_path):
    config_path = tmp_path / "robot_bool.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: dev",
                "detector: dummy",
                "loop_rates:",
                "  perception_hz: 20",
                "  behavior_hz: 3",
                "  control_hz: 80",
                "  hardware_hz: 120",
                "deadman_timeout_ms: 250",
                "joint_names: [yaw, pitch1, pitch2, roll, pitch3]",
                "joint_limits_rad:",
                "  - [-1.0, 1.0]",
                "  - [-1.0, 1.0]",
                "  - [-1.0, 1.0]",
                "  - [-1.0, 1.0]",
                "  - [-1.0, 1.0]",
                "servo_calibration: {}",
                "logging:",
                "  enabled: 'false'",
                "  perception_jsonl: logs/perception.jsonl",
                "  actions_jsonl: logs/actions.jsonl",
                "telemetry_preview:",
                "  enabled: 'false'",
                "cosmos:",
                "  enabled: 'false'",
                "  memory_enabled: 'false'",
                "  inflight_guard_enabled: 'false'",
                "  reasoning_probe_enabled: 'false'",
            ]
        )
    )
    cfg = load_config(str(config_path))
    assert cfg.logging.enabled is False
    assert cfg.telemetry_preview.enabled is False
    assert cfg.cosmos.enabled is False
    assert cfg.cosmos.memory_enabled is False
    assert cfg.cosmos.inflight_guard_enabled is False
    assert cfg.cosmos.reasoning_probe_enabled is False
