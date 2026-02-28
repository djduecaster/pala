from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from .models import EnvProbeRun


def ensure_logs_root(logs_root: Path) -> Path:
    logs_root.mkdir(parents=True, exist_ok=True)
    return logs_root


def new_run_dir(logs_root: Path) -> tuple[str, Path]:
    ensure_logs_root(logs_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_{uuid4().hex[:6]}"
    run_dir = logs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def save_run(run_dir: Path, run: EnvProbeRun) -> Dict[str, Any]:
    write_json(run_dir / "run_full.json", run.to_dict())
    write_json(run_dir / "run_config.json", run.params)
    write_json(run_dir / "inputs_manifest.json", run.images)
    write_json(
        run_dir / "packet_view.json",
        {
            "compact": run.packet_compact,
            "expanded": run.packet_expanded,
            "message_structure": run.message_structure,
        },
    )
    write_json(run_dir / "request_payload_redacted.json", run.request_payload_redacted)
    write_json(
        run_dir / "response_raw.json",
        {
            "response_meta": run.response_meta,
            "raw_content": run.raw_content,
            "reasoning_content": run.reasoning_content,
        },
    )
    write_json(
        run_dir / "parsed_output.json",
        {
            "parse_ok": run.parse_ok,
            "parse_stage": run.parse_stage,
            "parse_error": run.parse_error,
            "parsed_output": run.parsed_output,
        },
    )
    if isinstance(run.effective_inputs, dict):
        write_json(run_dir / "effective_inputs.json", run.effective_inputs)

    if isinstance(run.planner_phase, dict):
        write_json(
            run_dir / "planner_packet_view.json",
            {
                "compact": run.planner_phase.get("packet_compact", []),
                "expanded": run.planner_phase.get("packet_expanded", []),
                "message_structure": run.planner_phase.get("message_structure", []),
            },
        )
        write_json(
            run_dir / "planner_request_payload_redacted.json",
            run.planner_phase.get("request_payload_redacted", {}),
        )
        write_json(
            run_dir / "planner_response_raw.json",
            {
                "response_meta": run.planner_phase.get("response_meta", {}),
                "raw_content": run.planner_phase.get("raw_content"),
                "reasoning_content": run.planner_phase.get("reasoning_content"),
            },
        )
        write_json(
            run_dir / "planner_parsed_output.json",
            {
                "executed": bool(run.planner_phase.get("executed", False)),
                "parse_ok": bool(run.planner_phase.get("parse_ok", False)),
                "parse_stage": run.planner_phase.get("parse_stage"),
                "parse_error": run.planner_phase.get("parse_error"),
                "parsed_output": run.planner_phase.get("parsed_output"),
                "skip_reason": run.planner_phase.get("skip_reason"),
            },
        )

    summary = {
        "run_id": run.run_id,
        "created_at_utc": run.created_at_utc,
        "mode": run.mode,
        "chain_status": run.chain_status,
        "parse_ok": run.parse_ok,
        "parse_stage": run.parse_stage,
        "parse_error": run.parse_error,
        "http_status": run.response_meta.get("http_status"),
        "http_ok": run.response_meta.get("http_ok"),
        "latency_ms": run.response_meta.get("latency_ms"),
        "provider": run.params.get("provider"),
        "model": run.params.get("model"),
        "image_count": len(run.images),
        "planner_executed": bool((run.planner_phase or {}).get("executed", False)),
        "planner_http_status": (run.planner_phase or {}).get("response_meta", {}).get("http_status"),
        "planner_parse_ok": (run.planner_phase or {}).get("parse_ok"),
        "summary_short": None
        if not isinstance(run.parsed_output, dict)
        else run.parsed_output.get("summary_short"),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def _index_path(logs_root: Path) -> Path:
    return logs_root / "recent_index.json"


def update_recent_index(logs_root: Path, summary: Dict[str, Any], *, max_items: int = 30) -> None:
    ensure_logs_root(logs_root)
    path = _index_path(logs_root)
    current: List[Dict[str, Any]] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                current = [item for item in loaded if isinstance(item, dict)]
        except Exception:
            current = []

    run_id = str(summary.get("run_id", "")).strip()
    trimmed = [item for item in current if str(item.get("run_id", "")).strip() != run_id]
    out = [summary] + trimmed
    out = out[: max(1, int(max_items))]
    write_json(path, out)


def list_recent_runs(logs_root: Path, *, limit: int = 12) -> List[Dict[str, Any]]:
    path = _index_path(logs_root)
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(loaded, list):
        return []
    out = [item for item in loaded if isinstance(item, dict)]
    return out[: max(1, int(limit))]


def load_run(logs_root: Path, run_id: str) -> Dict[str, Any] | None:
    token = str(run_id).strip()
    if not token:
        return None
    run_dir = logs_root / token
    if not run_dir.is_dir():
        return None

    def _load(name: str) -> Any:
        path = run_dir / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    run_full = _load("run_full.json") or {}
    run_config = _load("run_config.json") or {}
    inputs_manifest = _load("inputs_manifest.json") or []
    packet_view = _load("packet_view.json") or {}
    response_raw = _load("response_raw.json") or {}
    parsed_output = _load("parsed_output.json") or {}
    effective_inputs = _load("effective_inputs.json") or {}
    planner_packet_view = _load("planner_packet_view.json") or {}
    planner_response_raw = _load("planner_response_raw.json") or {}
    planner_parsed_output = _load("planner_parsed_output.json") or {}
    planner_request_payload = _load("planner_request_payload_redacted.json") or {}
    summary = _load("summary.json") or {}

    return {
        "run_id": token,
        "run_dir": str(run_dir),
        "run_full": run_full,
        "run_config": run_config,
        "inputs_manifest": inputs_manifest,
        "packet_view": packet_view,
        "response_raw": response_raw,
        "parsed": parsed_output,
        "effective_inputs": effective_inputs,
        "planner_packet_view": planner_packet_view,
        "planner_response_raw": planner_response_raw,
        "planner_parsed": planner_parsed_output,
        "planner_request_payload": planner_request_payload,
        "summary": summary,
    }
