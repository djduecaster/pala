from __future__ import annotations

import pytest

from pala.config.load import load_config


def _base_lines() -> list[str]:
    return [
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
    ]


def _write(tmp_path, name: str, lines: list[str]):
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "does-not-exist.yaml"))


def test_load_config_rejects_non_mapping_root(tmp_path):
    path = tmp_path / "root_list.yaml"
    path.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        load_config(str(path))


def test_load_config_rejects_non_mapping_sections(tmp_path):
    lines = _base_lines() + ["logging: []"]
    with pytest.raises(ValueError, match="logging"):
        load_config(_write(tmp_path, "bad_logging.yaml", lines))

    lines = _base_lines() + ["telemetry_preview: []"]
    with pytest.raises(ValueError, match="telemetry_preview"):
        load_config(_write(tmp_path, "bad_preview.yaml", lines))

    lines = _base_lines() + ["camera: []"]
    with pytest.raises(ValueError, match="camera"):
        load_config(_write(tmp_path, "bad_camera.yaml", lines))

    lines = _base_lines() + ["cosmos: []"]
    with pytest.raises(ValueError, match="cosmos"):
        load_config(_write(tmp_path, "bad_cosmos.yaml", lines))


def test_load_config_rejects_invalid_joint_contracts(tmp_path):
    lines = list(_base_lines())
    lines[lines.index("joint_names: [yaw, pitch1, pitch2, roll, pitch3]")] = "joint_names: [yaw, 7]"
    with pytest.raises(ValueError, match="joint_names"):
        load_config(_write(tmp_path, "bad_joint_names.yaml", lines))

    lines = _base_lines()
    lines = [ln for ln in lines if ln != "  - [-1.0, 1.0]"]  # shorten limits
    with pytest.raises(ValueError, match="joint_limits_rad"):
        load_config(_write(tmp_path, "bad_joint_len.yaml", lines))

    lines = _base_lines()
    idx = lines.index("  - [-1.0, 1.0]")
    lines[idx] = "  - [1.0]"
    with pytest.raises(ValueError, match="joint_limits_rad\\[0\\]"):
        load_config(_write(tmp_path, "bad_joint_shape.yaml", lines))


def test_load_config_rejects_bad_scalar_types(tmp_path):
    lines = _base_lines()
    lines[lines.index("deadman_timeout_ms: 250")] = "deadman_timeout_ms: true"
    with pytest.raises(ValueError, match="deadman_timeout_ms"):
        load_config(_write(tmp_path, "bad_deadman.yaml", lines))

    lines = _base_lines() + ["logging:", "  enabled: maybe"]
    with pytest.raises(ValueError, match="logging.enabled"):
        load_config(_write(tmp_path, "bad_bool.yaml", lines))

    lines = _base_lines() + ["telemetry_preview:", "  enabled: 2"]
    with pytest.raises(ValueError, match="telemetry_preview.enabled"):
        load_config(_write(tmp_path, "bad_preview_bool_num.yaml", lines))


def test_load_config_styles_override_and_invalid_styles(tmp_path):
    lines = _base_lines() + [
        "styles:",
        "  calm:",
        "    rate_scale: 1.7",
        "  custom:",
        "    amp_scale: 0.5",
    ]
    cfg = load_config(_write(tmp_path, "styles_ok.yaml", lines))
    assert cfg.style_profiles["calm"]["rate_scale"] == 1.7
    assert cfg.style_profiles["custom"]["amp_scale"] == 0.5

    bad = _base_lines() + ["styles:", "  calm: not_a_mapping"]
    with pytest.raises(ValueError, match="styles.calm"):
        load_config(_write(tmp_path, "styles_bad.yaml", bad))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("perception_hz", 0),
        ("behavior_hz", "nan"),
        ("control_hz", -1),
        ("hardware_hz", "inf"),
    ],
)
def test_load_config_rejects_nonpositive_or_nonfinite_loop_rates(tmp_path, field, value):
    lines = _base_lines()
    index = next(i for i, line in enumerate(lines) if line.strip().startswith(f"{field}:"))
    lines[index] = f"  {field}: {value}"
    with pytest.raises(ValueError, match=f"loop_rates\\.{field}"):
        load_config(_write(tmp_path, f"bad_{field}.yaml", lines))


def test_load_config_rejects_unordered_limits_and_nonpositive_style(tmp_path):
    lines = _base_lines()
    index = lines.index("  - [-1.0, 1.0]")
    lines[index] = "  - [1.0, -1.0]"
    with pytest.raises(ValueError, match="joint_limits_rad\\[0\\]"):
        load_config(_write(tmp_path, "bad_order.yaml", lines))

    lines = _base_lines() + ["styles:", "  calm:", "    amp_scale: .nan"]
    with pytest.raises(ValueError, match="styles.calm.amp_scale"):
        load_config(_write(tmp_path, "bad_style_nan.yaml", lines))
