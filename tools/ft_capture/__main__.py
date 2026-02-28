from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .catalog import load_catalog, list_scenarios, resolve_scenario
from .capture import build_capture_settings, capture_scenario_takes
from .export import export_openai_jsonl
from .storage import new_session_id



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune capture tooling for PALA/Cosmos datasets")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_cmd = sub.add_parser("list-scenarios", help="List scenarios in catalog")
    list_cmd.add_argument("--catalog", default="config/ft_scenarios.yaml")

    validate_cmd = sub.add_parser("validate-catalog", help="Validate scenario catalog")
    validate_cmd.add_argument("--catalog", default="config/ft_scenarios.yaml")

    cap_cmd = sub.add_parser("capture", help="Capture one scenario with repeated takes")
    cap_cmd.add_argument("--catalog", default="config/ft_scenarios.yaml")
    cap_cmd.add_argument("--scenario", required=True)
    cap_cmd.add_argument("--takes", type=int, default=1)
    cap_cmd.add_argument("--countdown-s", type=float, default=None)
    cap_cmd.add_argument("--duration-s", type=float, default=None)
    cap_cmd.add_argument("--sample-fps", type=float, default=None)
    cap_cmd.add_argument("--out-root", default="logs/ft_capture")
    cap_cmd.add_argument("--session-id", default="")

    cap_cmd.add_argument("--runtime-config", default="config/robot.yaml")
    cap_cmd.add_argument("--camera-source", choices=["auto", "gst", "dummy"], default="auto")
    cap_cmd.add_argument("--camera-device", default="")
    cap_cmd.add_argument("--width", type=int, default=0)
    cap_cmd.add_argument("--height", type=int, default=0)
    cap_cmd.add_argument("--capture-fps", type=int, default=0)
    cap_cmd.add_argument("--camera-pipeline", default="")
    cap_cmd.add_argument("--jpeg-quality", type=int, default=90)

    export_cmd = sub.add_parser("export", help="Export labeled usable takes to OpenAI-compatible JSONL")
    export_cmd.add_argument("--catalog", default="config/ft_scenarios.yaml")
    export_cmd.add_argument("--dataset-root", default="logs/ft_capture")
    export_cmd.add_argument("--out", default="logs/ft_capture_exports/latest")

    return parser



def _cmd_list_scenarios(catalog_path: str) -> int:
    catalog = load_catalog(catalog_path)
    print(f"catalog={catalog.source_path}")
    for scenario in list_scenarios(catalog):
        print(
            f"- {scenario.scenario_id}: {scenario.title} "
            f"(countdown={scenario.countdown_s}s duration={scenario.duration_s}s sample_fps={scenario.sample_fps})"
        )
    return 0



def _cmd_validate_catalog(catalog_path: str) -> int:
    catalog = load_catalog(catalog_path)
    print(
        f"catalog valid: {catalog.source_path} scenarios={len(catalog.scenarios)} "
        f"split_ratio={catalog.split_ratio}"
    )
    return 0



def _cmd_capture(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    scenario = resolve_scenario(catalog, args.scenario)

    session_id = str(args.session_id or "").strip() or new_session_id()
    settings = build_capture_settings(
        out_root=args.out_root,
        session_id=session_id,
        catalog_path=args.catalog,
        scenario=scenario,
        takes=args.takes,
        countdown_s=args.countdown_s,
        duration_s=args.duration_s,
        sample_fps=args.sample_fps,
        camera_source=args.camera_source,
        camera_device=args.camera_device,
        width=args.width if int(args.width or 0) > 0 else None,
        height=args.height if int(args.height or 0) > 0 else None,
        capture_fps=args.capture_fps if int(args.capture_fps or 0) > 0 else None,
        camera_pipeline=str(args.camera_pipeline or "").strip() or None,
        jpeg_quality=args.jpeg_quality,
        runtime_config_path=args.runtime_config,
    )

    results = capture_scenario_takes(settings)
    print(f"capture complete: session={settings.session_id} takes={len(results)}")
    for row in results:
        print(
            f"  {row.take_id}: frames={row.frame_count} sampled={row.sample_frame_count} "
            f"duration={row.duration_s:.2f}s clip={row.clip_path}"
        )
    return 0



def _cmd_export(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    result = export_openai_jsonl(
        dataset_root=args.dataset_root,
        catalog=catalog,
        out_dir=args.out,
    )
    print(
        f"export complete: rows={result.total_rows} out={result.out_dir} "
        f"openai_jsonl={result.openai_jsonl_path}"
    )
    print(f"split_counts={result.split_counts}")
    return 0



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "list-scenarios":
        return _cmd_list_scenarios(args.catalog)
    if args.cmd == "validate-catalog":
        return _cmd_validate_catalog(args.catalog)
    if args.cmd == "capture":
        return _cmd_capture(args)
    if args.cmd == "export":
        return _cmd_export(args)

    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
