from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import subprocess
import sys
from typing import List, Sequence

from .integrity import verify_integrity_report
from .schema_v3 import INTEGRITY_REPORT_PATH, QUALITY_REPORT_PATH, REASONING_TRACE_INDEX_PATH, SESSION_DB_PATH, WEAK_LABELS_PATH


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
