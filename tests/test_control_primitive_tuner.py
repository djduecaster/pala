from __future__ import annotations

import argparse

import pytest

from pala.control import primitive_tuner as tuner


def _write_text(path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_parse_args_variants():
    show = tuner._parse_args(["show"])
    assert show.cmd == "show"
    assert show.config == "config/robot.yaml"
    assert show.style is None

    set_args = tuner._parse_args(
        [
            "set",
            "--style",
            "calm",
            "--amp-scale",
            "1.2",
            "--rate-scale",
            "1.1",
            "--duration-scale",
            "0.9",
            "--settle-scale",
            "1.4",
            "--write",
        ]
    )
    assert set_args.cmd == "set"
    assert set_args.style == "calm"
    assert set_args.amp_scale == 1.2
    assert set_args.rate_scale == 1.1
    assert set_args.duration_scale == 0.9
    assert set_args.settle_scale == 1.4
    assert set_args.write is True

    reset = tuner._parse_args(["reset", "--all"])
    assert reset.cmd == "reset"
    assert reset.all is True


def test_load_yaml_handles_none_and_rejects_non_mapping(tmp_path):
    config = tmp_path / "robot.yaml"
    _write_text(config, "")
    assert tuner._load_yaml(config) == {}

    _write_text(config, "- bad\n")
    with pytest.raises(ValueError, match="must contain a YAML mapping at top-level"):
        tuner._load_yaml(config)

    _write_text(config, "styles:\n  calm:\n    amp_scale: 1.1\n")
    loaded = tuner._load_yaml(config)
    assert loaded["styles"]["calm"]["amp_scale"] == 1.1


def test_style_overrides_validation_and_normalization():
    assert tuner._style_overrides({}) == {}

    with pytest.raises(ValueError, match="styles must be a mapping"):
        tuner._style_overrides({"styles": []})

    with pytest.raises(ValueError, match=r"styles\.bad must be a mapping"):
        tuner._style_overrides({"styles": {"bad": []}})

    out = tuner._style_overrides(
        {
            "styles": {
                " Calm ": {"amp_scale": "1.25", "rate_scale": 1},
                " ": {"amp_scale": 9},
                "custom": {"duration_scale": 0.8, "unused": 5},
            }
        }
    )
    assert sorted(out) == ["calm", "custom"]
    assert out["calm"] == {"amp_scale": 1.25, "rate_scale": 1.0}
    assert out["custom"] == {"duration_scale": 0.8}


def test_effective_styles_uses_defaults_and_override_merging():
    defaults = tuner.default_style_profiles()
    effective = tuner._effective_styles(
        {
            "curious": {"amp_scale": 1.4},
            "new_style": {"rate_scale": 1.6},
            "  ": {"amp_scale": 3.0},
        }
    )
    assert effective["curious"]["amp_scale"] == 1.4
    assert effective["curious"]["settle_scale"] == defaults["curious"]["settle_scale"]
    assert effective["new_style"]["amp_scale"] == defaults["calm"]["amp_scale"]
    assert effective["new_style"]["rate_scale"] == 1.6
    assert "  " not in effective


def test_print_effective_formats_values_and_handles_missing_filter(capsys):
    effective = tuner._effective_styles({"calm": {"amp_scale": 1.23456789}})
    tuner._print_effective(effective, overrides={"calm": {"amp_scale": 1.23456789}}, style_filter="calm")
    out = capsys.readouterr().out
    assert "calm (override)" in out
    assert "amp=1.23457" in out

    tuner._print_effective(effective, overrides={}, style_filter="missing")
    assert "No matching styles." in capsys.readouterr().out


def test_render_and_top_level_helpers():
    rendered = tuner._render_styles_block(
        {
            "focused": {"duration_scale": 0.9},
            "calm": {"amp_scale": 1.1, "settle_scale": 1.2},
        }
    )
    assert rendered[0] == "styles:\n"
    assert rendered[1] == "  calm:\n"
    assert rendered[2] == "    amp_scale: 1.1\n"
    assert rendered[3] == "    settle_scale: 1.2\n"
    assert rendered[4] == "  focused:\n"
    assert rendered[5] == "    duration_scale: 0.9\n"
    assert tuner._render_styles_block({}) == []

    lines = [
        "# header\n",
        " mode: dev\n",
        "styles:\n",
        "  calm:\n",
        "    amp_scale: 1.0\n",
        "\n",
        "# next section\n",
        "loop_rates:\n",
    ]
    start = tuner._top_level_key_index(lines, "styles")
    assert start == 2
    assert tuner._top_level_key_index(lines, "missing") == -1
    assert tuner._top_level_block_end(lines, start) == 6

    lines_no_comment = ["styles:\n", "  calm:\n", "    amp_scale: 1.0\n", "logging:\n"]
    assert tuner._top_level_block_end(lines_no_comment, 0) == 3


def test_replace_styles_block_cases():
    original = (
        "mode: dev\n"
        "styles:\n"
        "  calm:\n"
        "    amp_scale: 1.0\n"
        "# next section\n"
        "loop_rates:\n"
        "  behavior_hz: 3\n"
    )
    replaced = tuner._replace_styles_block_in_text(original, {"focused": {"rate_scale": 1.1}})
    assert "styles:\n  focused:\n    rate_scale: 1.1\n\n# next section\n" in replaced
    assert "amp_scale: 1.0" not in replaced

    removed = tuner._replace_styles_block_in_text(original, {})
    assert "styles:\n" not in removed
    assert "loop_rates:\n" in removed

    unchanged = "mode: dev\n"
    assert tuner._replace_styles_block_in_text(unchanged, {}) == unchanged

    appended = tuner._replace_styles_block_in_text("mode: dev", {"calm": {"amp_scale": 0.9}})
    assert appended.endswith("styles:\n  calm:\n    amp_scale: 0.9\n")
    assert "mode: dev\nstyles:\n" in appended


def test_validate_positive_and_write_styles(tmp_path):
    tuner._validate_positive("amp_scale", 0.1)
    with pytest.raises(ValueError, match="amp_scale must be > 0"):
        tuner._validate_positive("amp_scale", 0.0)

    config = tmp_path / "robot.yaml"
    _write_text(config, "mode: dev\nstyles:\n  calm:\n    amp_scale: 1.0\n")
    tuner._write_styles(config, {"calm": {"amp_scale": 1.2}, "curious": {"rate_scale": 1.3}})
    updated = config.read_text(encoding="utf-8")
    assert "amp_scale: 1.2" in updated
    assert "curious:\n    rate_scale: 1.3" in updated


def test_run_show_set_and_reset_paths(tmp_path, capsys):
    config = tmp_path / "robot.yaml"
    _write_text(config, "mode: dev\nstyles:\n  calm:\n    amp_scale: 1.0\n")

    show_args = argparse.Namespace(style="calm")
    assert tuner._run_show(config, show_args) == 0
    assert "calm (override)" in capsys.readouterr().out

    with pytest.raises(ValueError, match="set requires at least one"):
        tuner._run_set(
            config,
            argparse.Namespace(
                style="calm",
                amp_scale=None,
                rate_scale=None,
                duration_scale=None,
                settle_scale=None,
                write=False,
            ),
        )

    with pytest.raises(ValueError, match="--style must not be empty"):
        tuner._run_set(
            config,
            argparse.Namespace(
                style="   ",
                amp_scale=1.1,
                rate_scale=None,
                duration_scale=None,
                settle_scale=None,
                write=False,
            ),
        )

    with pytest.raises(ValueError, match="amp_scale must be > 0"):
        tuner._run_set(
            config,
            argparse.Namespace(
                style="calm",
                amp_scale=0.0,
                rate_scale=None,
                duration_scale=None,
                settle_scale=None,
                write=False,
            ),
        )

    set_dry_run = argparse.Namespace(
        style="calm",
        amp_scale=1.3,
        rate_scale=None,
        duration_scale=None,
        settle_scale=None,
        write=False,
    )
    assert tuner._run_set(config, set_dry_run) == 0
    dry_out = capsys.readouterr().out
    assert "Dry-run only. Re-run with --write to persist." in dry_out
    assert "amp=1.3" in dry_out
    assert "amp_scale: 1.3" not in config.read_text(encoding="utf-8")

    set_write = argparse.Namespace(
        style="calm",
        amp_scale=1.3,
        rate_scale=None,
        duration_scale=None,
        settle_scale=None,
        write=True,
    )
    assert tuner._run_set(config, set_write) == 0
    assert "Updated" in capsys.readouterr().out
    assert "amp_scale: 1.3" in config.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="reset requires --style <name> or --all"):
        tuner._run_reset(config, argparse.Namespace(style=None, all=False, write=False))

    no_change = argparse.Namespace(style="missing", all=False, write=False)
    assert tuner._run_reset(config, no_change) == 0
    assert "No override changes. Dry-run only." in capsys.readouterr().out

    remove_style = argparse.Namespace(style="calm", all=False, write=False)
    assert tuner._run_reset(config, remove_style) == 0
    assert "Dry-run only. Re-run with --write to persist." in capsys.readouterr().out

    reset_write = argparse.Namespace(style=None, all=True, write=True)
    assert tuner._run_reset(config, reset_write) == 0
    assert "Updated" in capsys.readouterr().out
    assert "styles:\n" not in config.read_text(encoding="utf-8")


def test_main_dispatch_and_errors(tmp_path, monkeypatch):
    config = tmp_path / "robot.yaml"
    _write_text(config, "mode: dev\nstyles:\n  calm:\n    amp_scale: 1.0\n")

    assert tuner.main(["show", "--config", str(config)]) == 0
    assert tuner.main(["set", "--config", str(config), "--style", "calm", "--amp-scale", "1.25", "--write"]) == 0
    assert tuner.main(["reset", "--config", str(config), "--all", "--write"]) == 0

    with pytest.raises(FileNotFoundError, match="Config not found"):
        tuner.main(["show", "--config", str(tmp_path / "missing.yaml")])

    monkeypatch.setattr(
        tuner,
        "_parse_args",
        lambda argv=None: argparse.Namespace(cmd="mystery", config=str(config), style=None),
    )
    with pytest.raises(ValueError, match="Unknown command: mystery"):
        tuner.main([])
