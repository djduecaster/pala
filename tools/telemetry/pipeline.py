from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Sequence

from .dataset_export import export_dataset_rows
from .integrity import build_integrity_report, write_integrity_report
from .run_report import _iter_session_dirs, build_run_report
from .storage_sqlite import build_session_db, ensure_session_db_contract, resolve_session_db_path


def _run_python_module(module: str, args: Sequence[str]) -> int:
    cmd = [sys.executable, "-m", module, *list(args)]
    return subprocess.call(cmd)


def _capture_cmd(args: argparse.Namespace) -> int:
    viewer_args = [
        "--mode",
        "live",
        "--save-session",
        str(args.save_session),
        "--jetson-host",
        str(args.jetson_host),
        "--jetson-dir",
        str(args.jetson_dir),
        "--quality-gate",
        str(args.quality_gate),
    ]
    query = str(args.query or "").strip()
    if query:
        viewer_args.extend(["--query", query])
    if bool(args.no_video):
        viewer_args.append("--no-video")
    return _run_python_module("tools.telemetry.mac_viewer", viewer_args)


def _compile_cmd(args: argparse.Namespace) -> int:
    session_dir = str(args.session_dir or "").strip()
    if not session_dir or (not os.path.isdir(session_dir)):
        raise SystemExit(f"session directory not found: {session_dir}")
    summary = build_session_db(session_dir, replace=not bool(args.no_replace))
    contract = ensure_session_db_contract(str(summary.get("db_path") or session_dir))
    try:
        integrity = build_integrity_report(session_dir)
        write_integrity_report(session_dir, integrity)
    except Exception:
        pass
    print(
        "compiled: "
        f"db={summary.get('db_path')} events={summary.get('event_count')} "
        f"reasoning={summary.get('reasoning_count')} traces={summary.get('trace_count')} "
        f"cases={summary.get('case_count')} labels={summary.get('case_label_count')}"
    )
    print(
        "contract: "
        f"ok={contract.get('ok')} errors={len(contract.get('errors') or [])} "
        f"warnings={len(contract.get('warnings') or [])}"
    )
    return 0


def _review_cmd(args: argparse.Namespace) -> int:
    session_dir = str(args.session_dir or "").strip()
    if not session_dir or (not os.path.isdir(session_dir)):
        raise SystemExit(f"session directory not found: {session_dir}")
    db_path = resolve_session_db_path(session_dir)
    if not os.path.exists(db_path):
        raise SystemExit(f"session db missing (run compile first): {db_path}")
    viewer_args = [
        "--mode",
        "curate",
        "--replay",
        session_dir,
        "--quality-gate",
        str(args.quality_gate),
        "--index-mode",
        "sqlite",
    ]
    query = str(args.query or "").strip()
    if query:
        viewer_args.extend(["--query", query])
    return _run_python_module("tools.telemetry.mac_viewer", viewer_args)


def _export_cmd(args: argparse.Namespace) -> int:
    session_dir = str(args.session_dir or "").strip()
    if not session_dir or (not os.path.isdir(session_dir)):
        raise SystemExit(f"session directory not found: {session_dir}")
    out = export_dataset_rows(
        session_dir,
        profile=str(args.profile),
        include_unlabeled=bool(args.include_unlabeled),
        min_label_confidence=float(args.min_label_confidence),
    )
    print(
        "exported: "
        f"profile={out.get('profile')} rows={out.get('row_count')} "
        f"path={out.get('output_path')} manifest={out.get('manifest_path')}"
    )
    return 0


def _expand(path: str) -> str:
    return os.path.normpath(os.path.expanduser(str(path or "").strip()))


def _report_cmd(args: argparse.Namespace) -> int:
    session_dirs = _iter_session_dirs(
        paths=[_expand(p) for p in args.paths if _expand(p)],
        root=_expand(str(args.root or "logs/telemetry")),
    )
    report = build_run_report(
        session_dirs=session_dirs,
        limit=max(0, int(args.limit)),
        mode_filter=str(args.mode or ""),
    )
    if bool(args.json):
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "report: "
            f"runs={report.get('runs_total')} sessions={report.get('sessions_with_runs')} "
            f"missing_runs={report.get('sessions_without_runs_count')} "
            f"alerts={report.get('alerts_count')} health={report.get('health')}"
        )
        cases = report.get("cases") if isinstance(report.get("cases"), dict) else {}
        print(
            "cases: "
            f"total={cases.get('total_cases')} reviewed={cases.get('reviewed_cases')} "
            f"coverage={cases.get('review_coverage')}"
        )
        stream_health = report.get("stream_health") if isinstance(report.get("stream_health"), dict) else {}
        print(
            "stream: "
            f"agent_q_peak={stream_health.get('transport_queue_peak_max')} "
            f"viewer_q_peak={stream_health.get('local_queue_peak_max')} "
            f"reconnect_p95={stream_health.get('reconnect_total_p95')} "
            f"rx_peak={stream_health.get('rx_rate_peak_max')}"
        )
        latest = report.get("latest_run")
        if isinstance(latest, dict):
            print(
                "latest: "
                f"mode={latest.get('mode')} exit={latest.get('exit_code')} "
                f"quality={latest.get('quality_score')} case_source={latest.get('case_source')} "
                f"case_coverage={latest.get('case_review_coverage')} "
                f"events={latest.get('event_count_total')} rx_peak={latest.get('rx_rate_peak_5s')}"
            )
        alerts = report.get("alerts") if isinstance(report.get("alerts"), list) else []
        if alerts:
            preview = ",".join(str(item) for item in alerts[:5])
            if len(alerts) > 5:
                preview = f"{preview},..."
            print(f"alerts: {preview}")
    if bool(args.strict) and int(report.get("alerts_count", 0) or 0) > 0:
        return 2
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Case-centric telemetry pipeline (capture/compile/review/export/report).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    capture = sub.add_parser("capture", help="Run live capture viewer and write a session bundle.")
    capture.add_argument("--save-session", required=True, help="Session output directory.")
    capture.add_argument("--jetson-host", default="jetson")
    capture.add_argument("--jetson-dir", default="~/pala")
    capture.add_argument("--query", default="")
    capture.add_argument("--quality-gate", choices=["off", "warn", "strict"], default="warn")
    capture.add_argument("--no-video", action="store_true")
    capture.set_defaults(func=_capture_cmd)

    compile_cmd = sub.add_parser("compile", help="Compile session.db and V4 cases from session artifacts.")
    compile_cmd.add_argument("session_dir")
    compile_cmd.add_argument("--no-replace", action="store_true", help="Do not replace an existing db file.")
    compile_cmd.set_defaults(func=_compile_cmd)

    review = sub.add_parser("review", help="Open curate viewer on compiled case data.")
    review.add_argument("session_dir")
    review.add_argument("--query", default="")
    review.add_argument("--quality-gate", choices=["off", "warn", "strict"], default="strict")
    review.set_defaults(func=_review_cmd)

    export = sub.add_parser("export", help="Export dataset rows from case-reviewed session data.")
    export.add_argument("session_dir")
    export.add_argument("--profile", choices=["fast", "strict", "hard_cases"], default="hard_cases")
    export.add_argument("--include-unlabeled", action="store_true")
    export.add_argument("--min-label-confidence", type=float, default=0.6)
    export.set_defaults(func=_export_cmd)

    report = sub.add_parser("report", help="Aggregate telemetry run health.")
    report.add_argument("paths", nargs="*")
    report.add_argument("--root", default="logs/telemetry")
    report.add_argument("--limit", type=int, default=0)
    report.add_argument("--mode", choices=["", "live", "replay", "curate"], default="")
    report.add_argument("--json", action="store_true")
    report.add_argument("--strict", action="store_true", help="Exit non-zero when report alerts are present.")
    report.set_defaults(func=_report_cmd)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    func = getattr(args, "func", None)
    if func is None:
        parser.error("subcommand required")
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
