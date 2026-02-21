from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Mapping, Sequence

from .compare import compare_sessions
from .scoreboard import DEFAULT_SCOREBOARD_PATH, load_scoreboard, summarize_scoreboard


def _is_session_dir(path: str) -> bool:
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "manifest.json")) or os.path.exists(os.path.join(path, "events.jsonl"))
    )


def resolve_candidate_sessions(paths: Sequence[str], *, discover: bool) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in paths:
        root = str(raw or "").strip()
        if not root:
            continue
        if _is_session_dir(root):
            if root not in seen:
                out.append(root)
                seen.add(root)
            continue
        if not discover or not os.path.isdir(root):
            continue
        try:
            entries = list(os.scandir(root))
        except OSError:
            continue
        for entry in sorted(entries, key=lambda e: e.name):
            if not entry.is_dir():
                continue
            path = entry.path
            if _is_session_dir(path) and path not in seen:
                out.append(path)
                seen.add(path)
    return out


def run_watchdog(
    baseline_dir: str,
    candidate_sessions: Sequence[str],
    *,
    quality_drop_tol: float = 2.0,
    doctor_drop_tol: float = 5.0,
    parse_fail_increase_tol: float = 0.03,
    timeout_increase_tol: float = 0.03,
    scoreboard_path: str = DEFAULT_SCOREBOARD_PATH,
) -> Dict[str, Any]:
    comparisons: List[Dict[str, Any]] = []
    for candidate in candidate_sessions:
        result = compare_sessions(
            baseline_dir,
            candidate,
            quality_drop_tol=quality_drop_tol,
            doctor_drop_tol=doctor_drop_tol,
            parse_fail_increase_tol=parse_fail_increase_tol,
            timeout_increase_tol=timeout_increase_tol,
        )
        result["candidate_dir"] = candidate
        comparisons.append(result)

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in comparisons:
        verdict = str(item.get("verdict") or "warn")
        if verdict not in counts:
            verdict = "warn"
        counts[verdict] += 1

    overall = "pass"
    if counts["fail"] > 0:
        overall = "fail"
    elif counts["warn"] > 0:
        overall = "warn"

    board_summary: Dict[str, Any] = {}
    if os.path.exists(str(scoreboard_path or DEFAULT_SCOREBOARD_PATH)):
        board = load_scoreboard(str(scoreboard_path or DEFAULT_SCOREBOARD_PATH))
        board_summary = summarize_scoreboard(board, min_sessions=1, top_n=8)

    return {
        "generated_at_wall_s": time.time(),
        "baseline_dir": baseline_dir,
        "candidate_count": len(candidate_sessions),
        "counts": counts,
        "overall_verdict": overall,
        "comparisons": comparisons,
        "scoreboard": board_summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run telemetry regression watchdog over one baseline and many candidates.")
    parser.add_argument("baseline_dir")
    parser.add_argument("candidates", nargs="+", help="Candidate session dirs or roots (with --discover).")
    parser.add_argument("--discover", action="store_true", help="Treat candidate args as roots and scan child session dirs.")
    parser.add_argument("--quality-drop-tol", type=float, default=2.0)
    parser.add_argument("--doctor-drop-tol", type=float, default=5.0)
    parser.add_argument("--parse-fail-increase-tol", type=float, default=0.03)
    parser.add_argument("--timeout-increase-tol", type=float, default=0.03)
    parser.add_argument("--scoreboard-path", default=DEFAULT_SCOREBOARD_PATH)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--fail-on-warn", action="store_true")
    return parser


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, separators=(",", ":"), ensure_ascii=True)
    return path


def main() -> int:
    args = _build_parser().parse_args()
    baseline = str(args.baseline_dir or "").strip()
    if not _is_session_dir(baseline):
        raise SystemExit(f"baseline session directory not found: {baseline}")
    candidates = resolve_candidate_sessions(args.candidates, discover=bool(args.discover))
    if not candidates:
        raise SystemExit("no candidate sessions resolved")

    report = run_watchdog(
        baseline,
        candidates,
        quality_drop_tol=float(args.quality_drop_tol),
        doctor_drop_tol=float(args.doctor_drop_tol),
        parse_fail_increase_tol=float(args.parse_fail_increase_tol),
        timeout_increase_tol=float(args.timeout_increase_tol),
        scoreboard_path=str(args.scoreboard_path or DEFAULT_SCOREBOARD_PATH),
    )
    if str(args.output or "").strip():
        path = _write_json(str(args.output), report)
        print(f"watchdog report: {path}")

    print(
        f"watchdog baseline={baseline} candidates={report.get('candidate_count')} "
        f"overall={report.get('overall_verdict')} counts={report.get('counts')}"
    )
    for item in report.get("comparisons", [])[:12]:
        if not isinstance(item, dict):
            continue
        print(
            f"  {item.get('candidate_dir')}: verdict={item.get('verdict')} "
            f"delta={item.get('delta')}"
        )
    overall = str(report.get("overall_verdict") or "warn")
    if overall == "fail":
        return 2
    if overall == "warn" and bool(args.fail_on_warn):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
