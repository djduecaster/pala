from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Mapping, Optional, Sequence

import yaml

from ..types.style_profiles import default_style_profiles

PARAM_KEYS: tuple[str, ...] = ("amp_scale", "rate_scale", "duration_scale", "settle_scale")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune primitive style parameters in config/robot.yaml (styles.*)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    show = sub.add_parser("show", help="Show effective style parameters")
    show.add_argument("--config", default="config/robot.yaml", help="Path to robot config YAML")
    show.add_argument("--style", default=None, help="Optional style filter (e.g. calm)")

    set_cmd = sub.add_parser("set", help="Set one or more style parameters")
    set_cmd.add_argument("--config", default="config/robot.yaml", help="Path to robot config YAML")
    set_cmd.add_argument("--style", required=True, help="Style name (e.g. calm, curious, focused)")
    set_cmd.add_argument("--amp-scale", type=float, default=None)
    set_cmd.add_argument("--rate-scale", type=float, default=None)
    set_cmd.add_argument("--duration-scale", type=float, default=None)
    set_cmd.add_argument("--settle-scale", type=float, default=None)
    set_cmd.add_argument(
        "--write",
        action="store_true",
        help="Persist changes into the config file (otherwise dry-run)",
    )

    reset = sub.add_parser("reset", help="Reset style overrides")
    reset.add_argument("--config", default="config/robot.yaml", help="Path to robot config YAML")
    reset.add_argument("--style", default=None, help="Style override to remove; omit with --all")
    reset.add_argument("--all", action="store_true", help="Remove entire styles override section")
    reset.add_argument(
        "--write",
        action="store_true",
        help="Persist changes into the config file (otherwise dry-run)",
    )
    return parser.parse_args(argv)


def _load_yaml(path: pathlib.Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at top-level")
    return data


def _style_overrides(doc: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    raw = doc.get("styles")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("styles must be a mapping")

    out: Dict[str, Dict[str, float]] = {}
    for name, vals in raw.items():
        key = str(name).strip().lower()
        if not key:
            continue
        if not isinstance(vals, Mapping):
            raise ValueError(f"styles.{name} must be a mapping")
        p: Dict[str, float] = {}
        for param in PARAM_KEYS:
            if param in vals:
                p[param] = float(vals[param])  # type: ignore[arg-type]
        out[key] = p
    return out


def _effective_styles(overrides: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    effective = default_style_profiles()
    for name, vals in overrides.items():
        key = str(name).strip().lower()
        if not key:
            continue
        base = dict(effective.get(key, effective["calm"]))
        for param in PARAM_KEYS:
            if param in vals:
                base[param] = float(vals[param])
        effective[key] = base
    return effective


def _format_value(value: float) -> str:
    return f"{float(value):.6g}"


def _print_effective(
    effective: Mapping[str, Mapping[str, float]],
    *,
    overrides: Mapping[str, Mapping[str, float]],
    style_filter: Optional[str] = None,
) -> None:
    names = sorted(effective.keys())
    if style_filter:
        token = style_filter.strip().lower()
        names = [n for n in names if n == token]
    if not names:
        print("No matching styles.")
        return

    for name in names:
        source = "override" if name in overrides else "default"
        vals = effective[name]
        print(
            f"{name} ({source}) "
            f"amp={_format_value(vals['amp_scale'])} "
            f"rate={_format_value(vals['rate_scale'])} "
            f"duration={_format_value(vals['duration_scale'])} "
            f"settle={_format_value(vals['settle_scale'])}"
        )


def _render_styles_block(overrides: Mapping[str, Mapping[str, float]]) -> list[str]:
    lines: list[str] = []
    if not overrides:
        return lines
    lines.append("styles:\n")
    for style_name in sorted(overrides.keys()):
        lines.append(f"  {style_name}:\n")
        vals = overrides[style_name]
        for param in PARAM_KEYS:
            if param in vals:
                lines.append(f"    {param}: {_format_value(vals[param])}\n")
    return lines


def _top_level_key_index(lines: list[str], key: str) -> int:
    needle = f"{key}:"
    for idx, line in enumerate(lines):
        if line.startswith((" ", "\t")):
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(needle):
            return idx
    return -1


def _top_level_block_end(lines: list[str], start_idx: int) -> int:
    for idx in range(start_idx + 1, len(lines)):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            continue
        if line.startswith((" ", "\t")):
            continue
        # Column-0 comment likely belongs to the following section.
        if stripped.startswith("#"):
            return idx
        return idx
    return len(lines)


def _replace_styles_block_in_text(text: str, overrides: Mapping[str, Mapping[str, float]]) -> str:
    lines = text.splitlines(keepends=True)
    new_block = _render_styles_block(overrides)

    start = _top_level_key_index(lines, "styles")
    if start >= 0:
        end = _top_level_block_end(lines, start)
        replacement = list(new_block)
        if replacement and end < len(lines) and lines[end - 1].strip():
            replacement.append("\n")
        out = lines[:start] + replacement + lines[end:]
        return "".join(out)

    if not new_block:
        return text

    out = list(lines)
    if out and out[-1].strip():
        out.append("\n")
    out.extend(new_block)
    return "".join(out)


def _validate_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")


def _write_styles(path: pathlib.Path, overrides: Mapping[str, Mapping[str, float]]) -> None:
    original = path.read_text(encoding="utf-8")
    updated = _replace_styles_block_in_text(original, overrides)
    path.write_text(updated, encoding="utf-8")


def _run_show(config_path: pathlib.Path, args: argparse.Namespace) -> int:
    doc = _load_yaml(config_path)
    overrides = _style_overrides(doc)
    effective = _effective_styles(overrides)
    _print_effective(effective, overrides=overrides, style_filter=args.style)
    return 0


def _run_set(config_path: pathlib.Path, args: argparse.Namespace) -> int:
    doc = _load_yaml(config_path)
    overrides = _style_overrides(doc)

    updates = {
        "amp_scale": args.amp_scale,
        "rate_scale": args.rate_scale,
        "duration_scale": args.duration_scale,
        "settle_scale": args.settle_scale,
    }
    provided = {k: v for k, v in updates.items() if v is not None}
    if not provided:
        raise ValueError("set requires at least one of --amp-scale/--rate-scale/--duration-scale/--settle-scale")

    style = str(args.style).strip().lower()
    if not style:
        raise ValueError("--style must not be empty")

    entry = dict(overrides.get(style, {}))
    for key, value in provided.items():
        v = float(value)
        _validate_positive(key, v)
        entry[key] = v
    overrides[style] = entry

    effective = _effective_styles(overrides)
    _print_effective(effective, overrides=overrides, style_filter=style)
    if args.write:
        _write_styles(config_path, overrides)
        print(f"Updated {config_path}")
    else:
        print("Dry-run only. Re-run with --write to persist.")
    return 0


def _run_reset(config_path: pathlib.Path, args: argparse.Namespace) -> int:
    doc = _load_yaml(config_path)
    overrides = _style_overrides(doc)
    original_count = len(overrides)

    if args.all:
        overrides = {}
    else:
        style = str(args.style or "").strip().lower()
        if not style:
            raise ValueError("reset requires --style <name> or --all")
        overrides.pop(style, None)

    effective = _effective_styles(overrides)
    _print_effective(effective, overrides=overrides)
    if args.write:
        _write_styles(config_path, overrides)
        print(f"Updated {config_path}")
    else:
        if len(overrides) == original_count:
            print("No override changes. Dry-run only.")
        else:
            print("Dry-run only. Re-run with --write to persist.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config_path = pathlib.Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if args.cmd == "show":
        return _run_show(config_path, args)
    if args.cmd == "set":
        return _run_set(config_path, args)
    if args.cmd == "reset":
        return _run_reset(config_path, args)
    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
