from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


VIEWER_SUMMARY_PATH = "viewer_summary.json"
VIEWER_RUNS_PATH = "viewer_runs.jsonl"
SESSION_MARKER_FILES = (VIEWER_RUNS_PATH, VIEWER_SUMMARY_PATH, "manifest.json", "events.jsonl")


def _expand_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    return os.path.normpath(os.path.expanduser(raw))


def _iter_session_dirs(*, paths: Sequence[str], root: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in paths:
        expanded = _expand_path(value)
        if not expanded:
            continue
        if os.path.isdir(expanded) and expanded not in seen:
            out.append(expanded)
            seen.add(expanded)
    if out:
        return out
    root_path = _expand_path(root)
    if not root_path or (not os.path.isdir(root_path)):
        return out
    for name in sorted(os.listdir(root_path)):
        path = os.path.join(root_path, name)
        if os.path.isdir(path) and _looks_like_session_dir(path) and path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _looks_like_session_dir(path: str) -> bool:
    root = _expand_path(path)
    if not root or (not os.path.isdir(root)):
        return False
    for rel in SESSION_MARKER_FILES:
        if os.path.exists(os.path.join(root, rel)):
            return True
    return False


def _parse_json_obj(raw: str) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _run_ts(run: Mapping[str, Any]) -> float:
    for key in ("ended_at_wall_s", "created_at_wall_s", "started_at_wall_s"):
        value = run.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _load_session_runs(session_dir: str) -> List[Dict[str, Any]]:
    root = _expand_path(session_dir)
    if not root:
        return []
    runs_path = os.path.join(root, VIEWER_RUNS_PATH)
    summary_path = os.path.join(root, VIEWER_SUMMARY_PATH)
    rows: List[Dict[str, Any]] = []
    if os.path.exists(runs_path):
        try:
            with open(runs_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    obj = _parse_json_obj(line)
                    if obj is None:
                        continue
                    obj.setdefault("_artifact", VIEWER_RUNS_PATH)
                    rows.append(obj)
        except Exception:
            rows = []
    if rows:
        rows.sort(key=_run_ts)
        return rows
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            if isinstance(obj, dict):
                obj.setdefault("_artifact", VIEWER_SUMMARY_PATH)
                return [obj]
        except Exception:
            return []
    return []


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    pos = (max(0.0, min(100.0, float(pct))) / 100.0) * (len(ordered) - 1)
    idx = int(round(pos))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_ratio(numer: int, denom: int) -> Optional[float]:
    if denom <= 0:
        return None
    return float(numer) / float(denom)


def _build_alerts(rows: Sequence[Tuple[str, Dict[str, Any]]]) -> List[str]:
    if not rows:
        return []
    latest = rows[-1][1]
    alerts: List[str] = []

    latest_exit = latest.get("exit_code")
    if isinstance(latest_exit, int) and latest_exit != 0:
        alerts.append(f"latest_exit_code_nonzero:{latest_exit}")

    latest_gate = latest.get("quality_gate_passed")
    if latest_gate is False:
        alerts.append("latest_quality_gate_failed")

    latest_curation = latest.get("curation_result")
    if isinstance(latest_curation, dict) and latest_curation.get("ok") is False:
        alerts.append("latest_curation_failed")

    if len(rows) < 2:
        return alerts
    previous_runs = [run for _, run in rows[:-1]]

    prev_quality = [_as_float(run.get("quality_score")) for run in previous_runs]
    prev_quality = [v for v in prev_quality if v is not None]
    latest_quality = _as_float(latest.get("quality_score"))
    if prev_quality and latest_quality is not None:
        baseline = _percentile(prev_quality, 50.0)
        if baseline is not None and latest_quality <= (baseline - 10.0):
            alerts.append(f"quality_score_regression:{latest_quality:.2f}<={baseline - 10.0:.2f}")

    prev_agent_drops = [_as_float(run.get("dropped_events_agent")) for run in previous_runs]
    prev_agent_drops = [v for v in prev_agent_drops if v is not None]
    latest_agent_drops = _as_float(latest.get("dropped_events_agent"))
    if prev_agent_drops and latest_agent_drops is not None:
        baseline = _percentile(prev_agent_drops, 95.0)
        threshold = max(20.0, (baseline or 0.0) * 2.0)
        if latest_agent_drops >= threshold:
            alerts.append(f"agent_drop_spike:{latest_agent_drops:.0f}>={threshold:.0f}")

    prev_local_drops = [_as_float(run.get("dropped_events_local")) for run in previous_runs]
    prev_local_drops = [v for v in prev_local_drops if v is not None]
    latest_local_drops = _as_float(latest.get("dropped_events_local"))
    if prev_local_drops and latest_local_drops is not None:
        baseline = _percentile(prev_local_drops, 95.0)
        threshold = max(20.0, (baseline or 0.0) * 2.0)
        if latest_local_drops >= threshold:
            alerts.append(f"local_drop_spike:{latest_local_drops:.0f}>={threshold:.0f}")

    return alerts


def build_run_report(
    *,
    session_dirs: Sequence[str],
    limit: int = 0,
    mode_filter: str = "",
) -> Dict[str, Any]:
    rows: List[Tuple[str, Dict[str, Any]]] = []
    mode_filter_norm = str(mode_filter or "").strip().lower()

    for session_dir in session_dirs:
        runs = _load_session_runs(session_dir)
        if not runs:
            continue
        for run in runs:
            mode = str(run.get("mode") or "").strip().lower()
            if mode_filter_norm and mode_filter_norm != mode:
                continue
            rows.append((session_dir, run))

    rows.sort(key=lambda item: _run_ts(item[1]))
    if int(limit) > 0 and len(rows) > int(limit):
        rows = rows[-int(limit) :]

    sessions_with_runs: set[str] = set()
    mode_counts: Dict[str, int] = {}
    exit_counts: Dict[str, int] = {}
    duration_values: List[float] = []
    quality_score_values: List[float] = []
    dropped_agent_values: List[float] = []
    dropped_local_values: List[float] = []
    exit_known = 0
    exit_nonzero = 0
    quality_known = 0
    quality_passed = 0
    curation_known = 0
    curation_ok = 0
    for session_dir, run in rows:
        sessions_with_runs.add(session_dir)
        mode = str(run.get("mode") or "").strip().lower()
        mode_counts[mode or "unknown"] = mode_counts.get(mode or "unknown", 0) + 1
        exit_code = run.get("exit_code")
        if isinstance(exit_code, int):
            exit_known += 1
            if exit_code != 0:
                exit_nonzero += 1
            exit_counts[str(exit_code)] = exit_counts.get(str(exit_code), 0) + 1
        duration_s = run.get("session_duration_s")
        if isinstance(duration_s, (int, float)):
            duration_values.append(max(0.0, float(duration_s)))
        quality_score = run.get("quality_score")
        if isinstance(quality_score, (int, float)):
            quality_score_values.append(float(quality_score))
        dropped_agent = run.get("dropped_events_agent")
        if isinstance(dropped_agent, (int, float)):
            dropped_agent_values.append(max(0.0, float(dropped_agent)))
        dropped_local = run.get("dropped_events_local")
        if isinstance(dropped_local, (int, float)):
            dropped_local_values.append(max(0.0, float(dropped_local)))
        qgp = run.get("quality_gate_passed")
        if isinstance(qgp, bool):
            quality_known += 1
            if qgp:
                quality_passed += 1
        curation = run.get("curation_result")
        if isinstance(curation, dict):
            ok = curation.get("ok")
            if isinstance(ok, bool):
                curation_known += 1
                if ok:
                    curation_ok += 1

    latest = None
    if rows:
        latest_session, latest_run = rows[-1]
        latest = {
            "session_dir": latest_session,
            "run_id": latest_run.get("run_id"),
            "mode": latest_run.get("mode"),
            "exit_code": latest_run.get("exit_code"),
            "ts_wall_s": _run_ts(latest_run),
            "quality_score": latest_run.get("quality_score"),
            "quality_gate_passed": latest_run.get("quality_gate_passed"),
            "dropped_events_agent": latest_run.get("dropped_events_agent"),
            "dropped_events_local": latest_run.get("dropped_events_local"),
        }

    avg_duration = None
    if duration_values:
        avg_duration = sum(duration_values) / len(duration_values)
    avg_quality = None
    if quality_score_values:
        avg_quality = sum(quality_score_values) / len(quality_score_values)
    alerts = _build_alerts(rows)
    quality_failed = max(0, quality_known - quality_passed)
    curation_failed = max(0, curation_known - curation_ok)
    sessions_without_runs = sorted(path for path in session_dirs if path not in sessions_with_runs)

    return {
        "sessions_scanned": len(session_dirs),
        "sessions_with_runs": len(sessions_with_runs),
        "sessions_without_runs_count": len(sessions_without_runs),
        "sessions_without_runs": sessions_without_runs[:20],
        "runs_total": len(rows),
        "mode_counts": mode_counts,
        "exit_code_counts": exit_counts,
        "exit_code_sample_count": exit_known,
        "exit_nonzero_rate": _safe_ratio(exit_nonzero, exit_known),
        "quality_gate_pass_rate": (quality_passed / quality_known) if quality_known > 0 else None,
        "quality_gate_sample_count": quality_known,
        "quality_gate_fail_rate": _safe_ratio(quality_failed, quality_known),
        "curation_success_rate": (curation_ok / curation_known) if curation_known > 0 else None,
        "curation_sample_count": curation_known,
        "curation_fail_rate": _safe_ratio(curation_failed, curation_known),
        "duration_s": {
            "avg": avg_duration,
            "p50": _percentile(duration_values, 50.0),
            "p95": _percentile(duration_values, 95.0),
        },
        "quality_score": {
            "avg": avg_quality,
            "p50": _percentile(quality_score_values, 50.0),
            "p95": _percentile(quality_score_values, 95.0),
        },
        "drops": {
            "agent_total": sum(dropped_agent_values) if dropped_agent_values else 0.0,
            "agent_p95": _percentile(dropped_agent_values, 95.0),
            "local_total": sum(dropped_local_values) if dropped_local_values else 0.0,
            "local_p95": _percentile(dropped_local_values, 95.0),
        },
        "alerts": alerts,
        "alerts_count": len(alerts),
        "health": "warn" if alerts else "ok",
        "latest_run": latest,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize telemetry viewer run artifacts across session directories.")
    parser.add_argument("paths", nargs="*", help="Session directories (defaults to scanning --root).")
    parser.add_argument("--root", default="logs/telemetry", help="Root directory to scan when no paths are provided.")
    parser.add_argument("--limit", type=int, default=0, help="Include only the latest N runs in output stats.")
    parser.add_argument("--mode", default="", choices=["", "live", "replay", "curate"], help="Optional mode filter.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when alerts are detected.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    session_dirs = _iter_session_dirs(paths=[str(p) for p in args.paths], root=str(args.root))
    report = build_run_report(session_dirs=session_dirs, limit=int(args.limit), mode_filter=str(args.mode))
    if args.json:
        print(json.dumps(report, separators=(",", ":"), ensure_ascii=True))
        return 0

    print("Telemetry Run Report")
    print(
        "summary: "
        f"sessions_scanned={report['sessions_scanned']} "
        f"sessions_with_runs={report['sessions_with_runs']} "
        f"sessions_without_runs={report.get('sessions_without_runs_count', 0)} "
        f"runs_total={report['runs_total']}"
    )
    print(f"modes: {report['mode_counts'] or '{}'}")
    print(f"exit_codes: {report['exit_code_counts'] or '{}'}")
    exit_nonzero_rate = report.get("exit_nonzero_rate")
    exit_n = int(report.get("exit_code_sample_count", 0) or 0)
    if isinstance(exit_nonzero_rate, float):
        print(f"exit_nonzero_rate: {exit_nonzero_rate:.3f} (n={exit_n})")
    else:
        print("exit_nonzero_rate: n/a")
    q_rate = report.get("quality_gate_pass_rate")
    q_n = int(report.get("quality_gate_sample_count", 0) or 0)
    if isinstance(q_rate, float):
        print(f"quality_gate_pass_rate: {q_rate:.3f} (n={q_n})")
    else:
        print("quality_gate_pass_rate: n/a")
    q_fail_rate = report.get("quality_gate_fail_rate")
    if isinstance(q_fail_rate, float):
        print(f"quality_gate_fail_rate: {q_fail_rate:.3f} (n={q_n})")
    c_rate = report.get("curation_success_rate")
    c_n = int(report.get("curation_sample_count", 0) or 0)
    if isinstance(c_rate, float):
        print(f"curation_success_rate: {c_rate:.3f} (n={c_n})")
    else:
        print("curation_success_rate: n/a")
    c_fail_rate = report.get("curation_fail_rate")
    if isinstance(c_fail_rate, float):
        print(f"curation_fail_rate: {c_fail_rate:.3f} (n={c_n})")
    duration = report.get("duration_s") if isinstance(report.get("duration_s"), dict) else {}
    print(
        "duration_s: "
        f"avg={duration.get('avg') if duration else None} "
        f"p50={duration.get('p50') if duration else None} "
        f"p95={duration.get('p95') if duration else None}"
    )
    quality_score = report.get("quality_score") if isinstance(report.get("quality_score"), dict) else {}
    print(
        "quality_score: "
        f"avg={quality_score.get('avg') if quality_score else None} "
        f"p50={quality_score.get('p50') if quality_score else None} "
        f"p95={quality_score.get('p95') if quality_score else None}"
    )
    drops = report.get("drops") if isinstance(report.get("drops"), dict) else {}
    print(
        "drops: "
        f"agent_total={drops.get('agent_total') if drops else None} "
        f"agent_p95={drops.get('agent_p95') if drops else None} "
        f"local_total={drops.get('local_total') if drops else None} "
        f"local_p95={drops.get('local_p95') if drops else None}"
    )
    latest = report.get("latest_run")
    if isinstance(latest, dict):
        print(
            "latest: "
            f"session={latest.get('session_dir')} mode={latest.get('mode')} "
            f"exit={latest.get('exit_code')} run_id={latest.get('run_id')}"
        )
    else:
        print("latest: n/a")
    alerts = report.get("alerts") if isinstance(report.get("alerts"), list) else []
    if alerts:
        print("alerts:")
        for item in alerts:
            print(f"- {item}")
    else:
        print("alerts: none")
    print(f"health: {report.get('health')}")
    if bool(args.strict) and bool(alerts):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
