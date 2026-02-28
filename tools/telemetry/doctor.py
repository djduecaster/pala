from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import sqlite3
import subprocess
import sys
from typing import List, Sequence

from .integrity import verify_integrity_report
from .run_report import build_run_report
from .schema_v3 import INTEGRITY_REPORT_PATH, QUALITY_REPORT_PATH, REASONING_TRACE_INDEX_PATH, SESSION_DB_PATH, WEAK_LABELS_PATH

VIEWER_SUMMARY_PATH = "viewer_summary.json"
VIEWER_RUNS_PATH = "viewer_runs.jsonl"


@dataclass
class Check:
    name: str
    status: str  # pass | warn | fail
    detail: str


def _run(cmd: Sequence[str], *, timeout_s: float = 3.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(0.5, float(timeout_s)),
        check=False,
    )


def _check_python() -> Check:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 10):
        return Check(name="python", status="fail", detail=f"python={ver} (need >=3.10)")
    return Check(name="python", status="pass", detail=f"python={ver}")


def _check_uv() -> Check:
    try:
        proc = _run(["uv", "--version"], timeout_s=3.0)
    except Exception as exc:
        return Check(name="uv", status="fail", detail=f"uv unavailable: {exc!r}")
    text = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return Check(name="uv", status="fail", detail=f"uv error: {text or 'unknown'}")
    return Check(name="uv", status="pass", detail=text or "uv ok")


def _check_imports() -> List[Check]:
    out: List[Check] = []
    required = ["json", "sqlite3"]
    optional = ["PIL", "numpy", "pytest"]
    for mod in required:
        try:
            __import__(mod)
        except Exception as exc:
            out.append(Check(name=f"import:{mod}", status="fail", detail=repr(exc)))
        else:
            out.append(Check(name=f"import:{mod}", status="pass", detail="ok"))
    for mod in optional:
        try:
            __import__(mod)
        except Exception as exc:
            out.append(Check(name=f"import:{mod}", status="warn", detail=repr(exc)))
        else:
            out.append(Check(name=f"import:{mod}", status="pass", detail="ok"))
    return out


def _check_workspace(repo_root: str) -> List[Check]:
    out: List[Check] = []
    must_exist = ["tools/telemetry", "pala/main.py", "config/robot.yaml"]
    for rel in must_exist:
        path = os.path.join(repo_root, rel)
        if os.path.exists(path):
            out.append(Check(name=f"path:{rel}", status="pass", detail="ok"))
        else:
            out.append(Check(name=f"path:{rel}", status="fail", detail="missing"))
    logs_dir = os.path.join(repo_root, "logs")
    if os.path.isdir(logs_dir):
        out.append(Check(name="logs_dir", status="pass", detail=logs_dir))
    else:
        out.append(Check(name="logs_dir", status="warn", detail=f"missing ({logs_dir})"))
    return out


def _check_ssh(host: str) -> Check:
    target = str(host or "").strip()
    if not target:
        return Check(name="ssh", status="warn", detail="skipped (no host)")
    try:
        proc = _run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", target, "echo", "ok"],
            timeout_s=6.0,
        )
    except Exception as exc:
        return Check(name="ssh", status="warn", detail=f"{target}: {exc!r}")
    text = (proc.stdout or "").strip()
    if proc.returncode == 0 and "ok" in text:
        return Check(name="ssh", status="pass", detail=f"{target} reachable")
    return Check(name="ssh", status="warn", detail=f"{target} unreachable ({text or 'no output'})")


def _check_session_dir(session_dir: str) -> List[Check]:
    root = str(session_dir).strip()
    if not root:
        return []
    out: List[Check] = []
    if not os.path.isdir(root):
        out.append(Check(name="session_dir", status="fail", detail=f"not found: {root}"))
        return out
    out.append(Check(name="session_dir", status="pass", detail=root))
    for rel in ("events.jsonl", "manifest.json"):
        path = os.path.join(root, rel)
        if os.path.exists(path):
            out.append(Check(name=f"session:{rel}", status="pass", detail="ok"))
        else:
            out.append(Check(name=f"session:{rel}", status="warn", detail="missing"))

    manifest_obj = {}
    manifest_path = os.path.join(root, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                decoded = json.load(fh)
            if isinstance(decoded, dict):
                manifest_obj = decoded
        except Exception:
            out.append(Check(name="session:manifest.parse", status="warn", detail="manifest.json parse failed"))
    schema_version = 0
    if isinstance(manifest_obj, dict):
        try:
            schema_version = int(manifest_obj.get("schema_version", 0) or 0)
        except (TypeError, ValueError):
            schema_version = 0
            out.append(Check(name="session:schema_version", status="warn", detail="invalid"))
    if schema_version > 0:
        out.append(Check(name="session:schema_version", status="pass", detail=str(schema_version)))

    v3_artifacts = [
        SESSION_DB_PATH,
        QUALITY_REPORT_PATH,
        WEAK_LABELS_PATH,
        REASONING_TRACE_INDEX_PATH,
        INTEGRITY_REPORT_PATH,
    ]
    missing_v3: List[str] = []
    for rel in v3_artifacts:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            out.append(Check(name=f"session:{rel}", status="pass", detail="ok"))
        else:
            out.append(Check(name=f"session:{rel}", status="warn", detail="missing"))
            missing_v3.append(rel)
    if schema_version >= 3 and missing_v3:
        out.append(Check(name="session:v3_artifacts", status="fail", detail=f"missing={','.join(missing_v3)}"))

    viewer_summary_obj = None
    summary_path = os.path.join(root, VIEWER_SUMMARY_PATH)
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                decoded = json.load(fh)
            if not isinstance(decoded, dict):
                raise ValueError("not an object")
            viewer_summary_obj = decoded
            out.append(Check(name="session:viewer_summary.parse", status="pass", detail="ok"))
        except Exception as exc:
            out.append(Check(name="session:viewer_summary.parse", status="fail", detail=repr(exc)))

    viewer_runs_last_obj = None
    runs_path = os.path.join(root, VIEWER_RUNS_PATH)
    if os.path.exists(runs_path):
        line_count = 0
        parse_errors = 0
        try:
            with open(runs_path, "r", encoding="utf-8") as fh:
                for raw in fh:
                    text = str(raw or "").strip()
                    if not text:
                        continue
                    line_count += 1
                    try:
                        obj = json.loads(text)
                    except Exception:
                        parse_errors += 1
                        continue
                    if isinstance(obj, dict):
                        viewer_runs_last_obj = obj
                    else:
                        parse_errors += 1
            if parse_errors > 0:
                out.append(
                    Check(
                        name="session:viewer_runs.parse",
                        status="fail",
                        detail=f"errors={parse_errors} lines={line_count}",
                    )
                )
            elif line_count == 0:
                out.append(Check(name="session:viewer_runs.parse", status="warn", detail="empty file"))
            else:
                out.append(Check(name="session:viewer_runs.parse", status="pass", detail=f"lines={line_count}"))
        except Exception as exc:
            out.append(Check(name="session:viewer_runs.parse", status="fail", detail=repr(exc)))

    if isinstance(viewer_summary_obj, dict) and isinstance(viewer_runs_last_obj, dict):
        summary_run_id = viewer_summary_obj.get("run_id")
        runs_run_id = viewer_runs_last_obj.get("run_id")
        if summary_run_id and runs_run_id and summary_run_id != runs_run_id:
            out.append(
                Check(
                    name="session:viewer_runs.latest_match",
                    status="warn",
                    detail=f"summary={summary_run_id} runs={runs_run_id}",
                )
            )
        else:
            out.append(Check(name="session:viewer_runs.latest_match", status="pass", detail="ok"))

    latest_viewer_obj = viewer_runs_last_obj if isinstance(viewer_runs_last_obj, dict) else viewer_summary_obj
    if isinstance(latest_viewer_obj, dict):
        case_source = str(latest_viewer_obj.get("case_source") or "").strip()
        case_reason = str(latest_viewer_obj.get("case_unavailable_reason") or "").strip()
        if case_source == "sqlite.cases.v4":
            out.append(Check(name="session:case_explorer.source", status="pass", detail=case_source))
            out.append(Check(name="session:case_explorer.ready", status="pass", detail="ready"))
        elif case_source:
            out.append(Check(name="session:case_explorer.source", status="warn", detail=case_source))
            out.append(
                Check(
                    name="session:case_explorer.ready",
                    status="warn",
                    detail=(case_reason or "unavailable"),
                )
            )
        else:
            out.append(Check(name="session:case_explorer.source", status="warn", detail="missing"))
            out.append(Check(name="session:case_explorer.ready", status="warn", detail="no case metadata"))

        queue_peak = latest_viewer_obj.get("transport_queue_peak_utilization")
        queue_peak_f = None
        if isinstance(queue_peak, (int, float)):
            queue_peak_f = float(queue_peak)
        if queue_peak_f is None:
            out.append(Check(name="session:stream.queue_peak", status="warn", detail="missing"))
        elif queue_peak_f >= 0.90:
            out.append(Check(name="session:stream.queue_peak", status="warn", detail=f"{queue_peak_f:.3f}"))
        else:
            out.append(Check(name="session:stream.queue_peak", status="pass", detail=f"{queue_peak_f:.3f}"))

        reconnect_total = latest_viewer_obj.get("reconnect_total")
        reconnect_total_f = None
        if isinstance(reconnect_total, (int, float)):
            reconnect_total_f = float(reconnect_total)
        if reconnect_total_f is None:
            out.append(Check(name="session:stream.reconnects", status="warn", detail="missing"))
        elif reconnect_total_f >= 3.0:
            out.append(Check(name="session:stream.reconnects", status="warn", detail=str(int(reconnect_total_f))))
        else:
            out.append(Check(name="session:stream.reconnects", status="pass", detail=str(int(reconnect_total_f))))
    elif os.path.exists(summary_path) or os.path.exists(runs_path):
        out.append(Check(name="session:case_explorer.source", status="warn", detail="unavailable"))
        out.append(Check(name="session:case_explorer.ready", status="warn", detail="no viewer run rows"))

    db_path = os.path.join(root, SESSION_DB_PATH)
    if os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                tables = {
                    str(row[0])
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    if row and isinstance(row[0], str)
                }
                if "cases" in tables:
                    total_cases = int(conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0])
                    reviewed_cases = 0
                    if "case_reviews" in tables:
                        reviewed_cases = int(conn.execute("SELECT COUNT(*) FROM case_reviews").fetchone()[0])
                    coverage = (float(reviewed_cases) / float(total_cases)) if total_cases > 0 else None
                    if total_cases <= 0:
                        out.append(Check(name="session:case_reviews.coverage", status="warn", detail="0/0 (no cases)"))
                    elif reviewed_cases <= 0:
                        out.append(
                            Check(
                                name="session:case_reviews.coverage",
                                status="warn",
                                detail=f"{reviewed_cases}/{total_cases} ({coverage:.3f})",
                            )
                        )
                    else:
                        out.append(
                            Check(
                                name="session:case_reviews.coverage",
                                status="pass",
                                detail=f"{reviewed_cases}/{total_cases} ({coverage:.3f})",
                            )
                        )
        except Exception as exc:
            out.append(Check(name="session:case_reviews.coverage", status="warn", detail=repr(exc)))
    if os.path.exists(summary_path) or os.path.exists(runs_path):
        try:
            run_health = build_run_report(session_dirs=[root], limit=25)
            runs_total = int(run_health.get("runs_total", 0) or 0)
            alerts = run_health.get("alerts") if isinstance(run_health.get("alerts"), list) else []
            health = str(run_health.get("health") or "ok")
            status = "warn" if alerts else "pass"
            out.append(
                Check(
                    name="session:viewer_runs.health",
                    status=status,
                    detail=f"health={health} runs={runs_total} alerts={len(alerts)}",
                )
            )
            if alerts:
                preview = ",".join(str(item) for item in alerts[:3])
                if len(alerts) > 3:
                    preview = f"{preview},..."
                out.append(Check(name="session:viewer_runs.alerts", status="warn", detail=preview))
                live_activity = [str(item) for item in alerts if str(item).startswith("latest_live_low_activity_events:")]
                if live_activity:
                    out.append(Check(name="session:viewer_runs.live_activity", status="warn", detail=live_activity[0]))
        except Exception as exc:
            out.append(Check(name="session:viewer_runs.health", status="warn", detail=repr(exc)))

    integrity_path = os.path.join(root, INTEGRITY_REPORT_PATH)
    if os.path.exists(integrity_path):
        try:
            verify = verify_integrity_report(root)
        except Exception as exc:
            out.append(Check(name="session:integrity.verify", status="fail", detail=repr(exc)))
        else:
            ok = bool(verify.get("ok"))
            if ok:
                out.append(
                    Check(
                        name="session:integrity.verify",
                        status="pass",
                        detail=f"checked={int(verify.get('checked_file_count', 0))}",
                    )
                )
            else:
                out.append(
                    Check(
                        name="session:integrity.verify",
                        status="fail",
                        detail=(
                            f"missing={len(verify.get('missing') or [])} "
                            f"mismatch={len(verify.get('mismatch') or [])}"
                        ),
                    )
                )
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telemetry preflight checks for local and Jetson workflows.")
    parser.add_argument("--jetson-host", default="jetson")
    parser.add_argument("--session-dir", default="", help="Optional existing capture dir to validate.")
    parser.add_argument("--skip-ssh", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warn/fail checks.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = os.getcwd()
    checks: List[Check] = []
    checks.append(_check_python())
    checks.append(_check_uv())
    checks.extend(_check_imports())
    checks.extend(_check_workspace(repo_root))
    checks.extend(_check_session_dir(str(args.session_dir or "")))
    if not args.skip_ssh:
        checks.append(_check_ssh(str(args.jetson_host or "")))

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1

    if args.json:
        payload = {"counts": counts, "checks": [asdict(c) for c in checks]}
        print(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))
    else:
        print("Telemetry Doctor")
        for check in checks:
            print(f"[{check.status.upper()}] {check.name}: {check.detail}")
        print(f"summary: pass={counts['pass']} warn={counts['warn']} fail={counts['fail']}")

    if args.strict and (counts["warn"] > 0 or counts["fail"] > 0):
        return 2
    if counts["fail"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
