from __future__ import annotations

import json
import os
import time
import argparse
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_SCOREBOARD_PATH = "logs/telemetry/scoreboard.json"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def load_scoreboard(path: str = DEFAULT_SCOREBOARD_PATH) -> Dict[str, Any]:
    target = str(path or DEFAULT_SCOREBOARD_PATH)
    if not os.path.exists(target):
        return {"version": 1, "updated_at_wall_s": time.time(), "sessions": []}
    try:
        with open(target, "r", encoding="utf-8") as fh:
            decoded = json.load(fh)
    except Exception:
        return {"version": 1, "updated_at_wall_s": time.time(), "sessions": []}
    if not isinstance(decoded, dict):
        return {"version": 1, "updated_at_wall_s": time.time(), "sessions": []}
    sessions = decoded.get("sessions")
    if not isinstance(sessions, list):
        decoded["sessions"] = []
    return decoded


def save_scoreboard(board: Mapping[str, Any], path: str = DEFAULT_SCOREBOARD_PATH) -> str:
    target = str(path or DEFAULT_SCOREBOARD_PATH)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(dict(board), fh, separators=(",", ":"), ensure_ascii=True)
    return target


def _rolling_avg(entries: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values: List[float] = []
    for row in entries:
        val = _safe_float(row.get(key))
        if val is None:
            continue
        values.append(val)
    if not values:
        return None
    return sum(values) / float(len(values))


def compute_scoreboard_trend(
    sessions: Sequence[Mapping[str, Any]],
    *,
    lookback: int = 5,
) -> Dict[str, Any]:
    if not sessions:
        return {"has_baseline": False}
    recent = list(sessions)[-max(1, int(lookback)) :]
    latest = dict(sessions[-1])
    baseline = recent[:-1]
    quality_latest = _safe_float(latest.get("quality_score"))
    parse_latest = _safe_float(latest.get("parse_fail_rate"))
    timeout_latest = _safe_float(latest.get("timeout_rate"))
    slow_latest = _safe_float(latest.get("slow_rate"))
    label_latest = _safe_float(latest.get("weak_label_count"))

    quality_base = _rolling_avg(baseline, "quality_score")
    parse_base = _rolling_avg(baseline, "parse_fail_rate")
    timeout_base = _rolling_avg(baseline, "timeout_rate")
    slow_base = _rolling_avg(baseline, "slow_rate")
    label_base = _rolling_avg(baseline, "weak_label_count")

    return {
        "has_baseline": bool(baseline),
        "latest_session": latest.get("session_name"),
        "quality_delta": None if quality_latest is None or quality_base is None else round(quality_latest - quality_base, 3),
        "parse_fail_delta": None if parse_latest is None or parse_base is None else round(parse_latest - parse_base, 4),
        "timeout_delta": None if timeout_latest is None or timeout_base is None else round(timeout_latest - timeout_base, 4),
        "slow_delta": None if slow_latest is None or slow_base is None else round(slow_latest - slow_base, 4),
        "label_yield_delta": None if label_latest is None or label_base is None else round(label_latest - label_base, 3),
    }


def add_scoreboard_session(
    *,
    path: str = DEFAULT_SCOREBOARD_PATH,
    session_dir: str,
    manifest: Optional[Mapping[str, Any]],
    quality_report: Optional[Mapping[str, Any]],
    improvement_report: Optional[Mapping[str, Any]],
    scenario_tags: Optional[Sequence[str]] = None,
    goal_tags: Optional[Sequence[str]] = None,
    runbook: str = "",
) -> Dict[str, Any]:
    board = load_scoreboard(path)
    sessions = board.get("sessions")
    if not isinstance(sessions, list):
        sessions = []

    summary = (improvement_report or {}).get("summary")
    summary = summary if isinstance(summary, dict) else {}

    reasoning_count = _safe_int(summary.get("reasoning_count"), 0)
    parse_fail_count = _safe_int(summary.get("parse_fail_count"), 0)
    timeout_count = _safe_int(summary.get("timeout_count"), 0)
    slow_count = _safe_int(summary.get("slow_count"), 0)
    weak_label_count = _safe_int(summary.get("weak_label_count"), 0)
    trace_count = _safe_int(summary.get("trace_count"), 0)
    event_count = _safe_int(summary.get("event_count"), _safe_int((manifest or {}).get("event_count"), 0))

    entry = {
        "created_at_wall_s": time.time(),
        "session_dir": str(session_dir),
        "session_name": os.path.basename(str(session_dir).rstrip("/")) or str(session_dir),
        "event_count": event_count,
        "reasoning_count": reasoning_count,
        "trace_count": trace_count,
        "weak_label_count": weak_label_count,
        "parse_fail_count": parse_fail_count,
        "timeout_count": timeout_count,
        "slow_count": slow_count,
        "parse_fail_rate": round(float(parse_fail_count) / float(max(1, reasoning_count)), 5),
        "timeout_rate": round(float(timeout_count) / float(max(1, reasoning_count)), 5),
        "slow_rate": round(float(slow_count) / float(max(1, reasoning_count)), 5),
        "quality_grade": (quality_report or {}).get("grade"),
        "quality_score": (quality_report or {}).get("score"),
        "scenario_tags": [str(x) for x in (scenario_tags or []) if str(x).strip()],
        "goal_tags": [str(x) for x in (goal_tags or []) if str(x).strip()],
        "runbook": str(runbook or ""),
    }

    sessions.append(entry)
    max_sessions = 500
    if len(sessions) > max_sessions:
        sessions = sessions[-max_sessions:]

    board["sessions"] = sessions
    board["updated_at_wall_s"] = time.time()
    board["version"] = 1
    board["trend"] = compute_scoreboard_trend(sessions, lookback=5)
    saved_path = save_scoreboard(board, path)
    return {"path": saved_path, "entry": entry, "trend": board.get("trend")}


def compute_tag_leaderboard(
    sessions: Sequence[Mapping[str, Any]],
    *,
    tag_key: str,
    min_sessions: int = 1,
    top_n: int = 12,
) -> List[Dict[str, Any]]:
    agg: Dict[str, Dict[str, float]] = {}
    for row in sessions:
        values = row.get(tag_key)
        if not isinstance(values, list):
            continue
        tags = [str(x).strip() for x in values if str(x).strip()]
        if not tags:
            continue
        for tag in tags:
            item = agg.setdefault(
                tag,
                {
                    "session_count": 0.0,
                    "quality_sum": 0.0,
                    "quality_count": 0.0,
                    "parse_fail_rate_sum": 0.0,
                    "timeout_rate_sum": 0.0,
                    "slow_rate_sum": 0.0,
                    "weak_label_sum": 0.0,
                },
            )
            item["session_count"] += 1.0
            q = _safe_float(row.get("quality_score"))
            if q is not None:
                item["quality_sum"] += q
                item["quality_count"] += 1.0
            item["parse_fail_rate_sum"] += float(_safe_float(row.get("parse_fail_rate"), 0.0) or 0.0)
            item["timeout_rate_sum"] += float(_safe_float(row.get("timeout_rate"), 0.0) or 0.0)
            item["slow_rate_sum"] += float(_safe_float(row.get("slow_rate"), 0.0) or 0.0)
            item["weak_label_sum"] += float(_safe_float(row.get("weak_label_count"), 0.0) or 0.0)

    out: List[Dict[str, Any]] = []
    for tag, item in agg.items():
        count = int(item["session_count"])
        if count < max(1, int(min_sessions)):
            continue
        q_count = max(1.0, item["quality_count"])
        out.append(
            {
                "tag": tag,
                "session_count": count,
                "quality_avg": round(item["quality_sum"] / q_count, 3),
                "parse_fail_rate_avg": round(item["parse_fail_rate_sum"] / float(count), 5),
                "timeout_rate_avg": round(item["timeout_rate_sum"] / float(count), 5),
                "slow_rate_avg": round(item["slow_rate_sum"] / float(count), 5),
                "weak_labels_avg": round(item["weak_label_sum"] / float(count), 3),
            }
        )
    out.sort(key=lambda row: (-float(row.get("quality_avg", 0.0)), float(row.get("parse_fail_rate_avg", 0.0)), str(row.get("tag"))))
    return out[: max(1, int(top_n))]


def summarize_scoreboard(
    board: Mapping[str, Any],
    *,
    min_sessions: int = 1,
    top_n: int = 12,
) -> Dict[str, Any]:
    sessions = board.get("sessions")
    sessions = sessions if isinstance(sessions, list) else []
    return {
        "generated_at_wall_s": time.time(),
        "session_count": len(sessions),
        "trend": board.get("trend"),
        "scenario_leaderboard": compute_tag_leaderboard(
            sessions,
            tag_key="scenario_tags",
            min_sessions=min_sessions,
            top_n=top_n,
        ),
        "goal_leaderboard": compute_tag_leaderboard(
            sessions,
            tag_key="goal_tags",
            min_sessions=min_sessions,
            top_n=top_n,
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize telemetry scoreboard trends and tag leaderboards.")
    parser.add_argument("--scoreboard-path", default=DEFAULT_SCOREBOARD_PATH)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--min-sessions", type=int, default=1)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, separators=(",", ":"), ensure_ascii=True)
    return path


def main() -> int:
    args = _build_parser().parse_args()
    board = load_scoreboard(str(args.scoreboard_path or DEFAULT_SCOREBOARD_PATH))
    summary = summarize_scoreboard(
        board,
        min_sessions=max(1, int(args.min_sessions)),
        top_n=max(1, int(args.top_n)),
    )
    if str(args.output or "").strip():
        path = _write_json(str(args.output), summary)
        print(f"scoreboard summary: {path}")

    print(f"scoreboard sessions={summary.get('session_count')} trend={summary.get('trend')}")
    scenarios = summary.get("scenario_leaderboard")
    if isinstance(scenarios, list) and scenarios:
        print("top scenarios:")
        for row in scenarios[:5]:
            print(
                f"  {row.get('tag')}: sessions={row.get('session_count')} "
                f"quality_avg={row.get('quality_avg')} parse_fail_avg={row.get('parse_fail_rate_avg')}"
            )
    goals = summary.get("goal_leaderboard")
    if isinstance(goals, list) and goals:
        print("top goals:")
        for row in goals[:5]:
            print(
                f"  {row.get('tag')}: sessions={row.get('session_count')} "
                f"quality_avg={row.get('quality_avg')} parse_fail_avg={row.get('parse_fail_rate_avg')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
