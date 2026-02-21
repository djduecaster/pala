from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List

from .packs import list_packs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified telemetry CLI for viewer/agent/migration flows.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("packs", help="List available telemetry signal packs.")
    sub.add_parser("viewer", help="Run Mac telemetry viewer. Pass viewer args after '--'.")
    sub.add_parser("agent", help="Run Jetson telemetry agent. Pass agent args after '--'.")
    sub.add_parser("migrate", help="Run telemetry bundle migration. Pass migrate args after '--'.")
    sub.add_parser("doctor", help="Run telemetry doctor checks on a session. Pass args after '--'.")
    sub.add_parser("compare", help="Compare baseline vs candidate session reports. Pass args after '--'.")
    sub.add_parser("incident", help="Build incident bundle (JSON + markdown) for a session. Pass args after '--'.")
    sub.add_parser("scoreboard", help="Summarize scoreboard trends and leaderboards. Pass args after '--'.")
    sub.add_parser("watchdog", help="Run multi-session regression watchdog. Pass args after '--'.")
    return parser


def _run_module(module: str, argv: List[str]) -> int:
    cmd = [sys.executable, "-m", module, *argv]
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = _build_parser()
    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    if args.command == "packs":
        for pack in list_packs():
            print(f"{pack.name}: {pack.description}")
        return 0
    if args.command == "viewer":
        return _run_module("tools.telemetry.mac_viewer", rest)
    if args.command == "agent":
        return _run_module("tools.telemetry.jetson_agent", rest)
    if args.command == "migrate":
        return _run_module("tools.telemetry.migrate_session", rest)
    if args.command == "doctor":
        return _run_module("tools.telemetry.doctor", rest)
    if args.command == "compare":
        return _run_module("tools.telemetry.compare", rest)
    if args.command == "incident":
        return _run_module("tools.telemetry.incident", rest)
    if args.command == "scoreboard":
        return _run_module("tools.telemetry.scoreboard", rest)
    if args.command == "watchdog":
        return _run_module("tools.telemetry.watchdog", rest)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
