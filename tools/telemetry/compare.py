from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Mapping, Optional

from .doctor import load_doctor_report
from .insights import load_improvement_report
from .quality import load_quality_report


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_metrics(session_dir: str) -> Dict[str, Any]:
    quality = load_quality_report(session_dir) or {}
    improvement = load_improvement_report(session_dir) or {}
    doctor = load_doctor_report(session_dir) or {}
    summary = improvement.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    readiness = doctor.get("readiness")
    readiness = readiness if isinstance(readiness, dict) else {}

    reasoning_count = int(summary.get("reasoning_count", 0) or 0)
    parse_fail_count = int(summary.get("parse_fail_count", 0) or 0)
    timeout_count = int(summary.get("timeout_count", 0) or 0)
    slow_count = int(summary.get("slow_count", 0) or 0)

    return {
        "session_dir": session_dir,
        "quality_score": _safe_float(quality.get("score"), 0.0),
        "quality_grade": quality.get("grade"),
        "doctor_score": _safe_float(readiness.get("score"), 0.0),
        "doctor_grade": readiness.get("grade"),
        "reasoning_count": reasoning_count,
        "parse_fail_count": parse_fail_count,
        "timeout_count": timeout_count,
        "slow_count": slow_count,
        "parse_fail_rate": (float(parse_fail_count) / float(max(1, reasoning_count))),
        "timeout_rate": (float(timeout_count) / float(max(1, reasoning_count))),
        "slow_rate": (float(slow_count) / float(max(1, reasoning_count))),
    }


def compare_sessions(
    baseline_dir: str,
    candidate_dir: str,
    *,
    quality_drop_tol: float = 2.0,
    parse_fail_increase_tol: float = 0.03,
    timeout_increase_tol: float = 0.03,
    doctor_drop_tol: float = 5.0,
) -> Dict[str, Any]:
    base = _extract_metrics(baseline_dir)
    cand = _extract_metrics(candidate_dir)

    delta = {
        "quality_score": round(cand["quality_score"] - base["quality_score"], 4),
        "doctor_score": round(cand["doctor_score"] - base["doctor_score"], 4),
        "parse_fail_rate": round(cand["parse_fail_rate"] - base["parse_fail_rate"], 6),
        "timeout_rate": round(cand["timeout_rate"] - base["timeout_rate"], 6),
        "slow_rate": round(cand["slow_rate"] - base["slow_rate"], 6),
    }

    regressions = []
    if delta["quality_score"] < -abs(float(quality_drop_tol)):
        regressions.append(f"quality_score_drop={delta['quality_score']}")
    if delta["doctor_score"] < -abs(float(doctor_drop_tol)):
        regressions.append(f"doctor_score_drop={delta['doctor_score']}")
    if delta["parse_fail_rate"] > abs(float(parse_fail_increase_tol)):
        regressions.append(f"parse_fail_increase={delta['parse_fail_rate']}")
    if delta["timeout_rate"] > abs(float(timeout_increase_tol)):
        regressions.append(f"timeout_increase={delta['timeout_rate']}")

    verdict = "pass"
    if regressions:
        verdict = "fail"
    elif cand.get("quality_grade") in {"warn", "fail"} or cand.get("doctor_grade") in {"warn", "fail"}:
        verdict = "warn"

    return {
        "generated_at_wall_s": time.time(),
        "baseline": base,
        "candidate": cand,
        "delta": delta,
        "thresholds": {
            "quality_drop_tol": float(quality_drop_tol),
            "doctor_drop_tol": float(doctor_drop_tol),
            "parse_fail_increase_tol": float(parse_fail_increase_tol),
            "timeout_increase_tol": float(timeout_increase_tol),
        },
        "regressions": regressions,
        "verdict": verdict,
    }


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, separators=(",", ":"), ensure_ascii=True)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two telemetry sessions for regressions.")
    parser.add_argument("baseline_dir")
    parser.add_argument("candidate_dir")
    parser.add_argument("--quality-drop-tol", type=float, default=2.0)
    parser.add_argument("--doctor-drop-tol", type=float, default=5.0)
    parser.add_argument("--parse-fail-increase-tol", type=float, default=0.03)
    parser.add_argument("--timeout-increase-tol", type=float, default=0.03)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--fail-on-warn", action="store_true", help="Return non-zero for warn verdict.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    baseline = str(args.baseline_dir or "").strip()
    candidate = str(args.candidate_dir or "").strip()
    if not baseline or not os.path.isdir(baseline):
        raise SystemExit(f"baseline directory not found: {baseline}")
    if not candidate or not os.path.isdir(candidate):
        raise SystemExit(f"candidate directory not found: {candidate}")

    out = compare_sessions(
        baseline,
        candidate,
        quality_drop_tol=float(args.quality_drop_tol),
        doctor_drop_tol=float(args.doctor_drop_tol),
        parse_fail_increase_tol=float(args.parse_fail_increase_tol),
        timeout_increase_tol=float(args.timeout_increase_tol),
    )
    if str(args.output or "").strip():
        path = _write_json(str(args.output), out)
        print(f"compare report: {path}")

    print(
        "compare verdict="
        f"{out.get('verdict')} quality_delta={out.get('delta', {}).get('quality_score')} "
        f"doctor_delta={out.get('delta', {}).get('doctor_score')} "
        f"parse_fail_delta={out.get('delta', {}).get('parse_fail_rate')} "
        f"timeout_delta={out.get('delta', {}).get('timeout_rate')}"
    )
    regressions = out.get("regressions")
    if isinstance(regressions, list) and regressions:
        for item in regressions:
            print(f"regression: {item}")

    verdict = str(out.get("verdict") or "warn")
    if verdict == "fail":
        return 2
    if verdict == "warn" and bool(args.fail_on_warn):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
