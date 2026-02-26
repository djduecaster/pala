#!/usr/bin/env python3
"""Static-image Cosmos API diagnostics tool for env/planner request+parse loops."""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pala.behavior.env_summarizer import EnvSummarizer
from pala.behavior.intent_proposer import IntentProposer
from pala.behavior.model_clients import extract_message_content, normalize_chat_url, post_chat_json
from pala.behavior.prompts import SYSTEM_PROMPT, build_env_user_text, build_messages, build_planner_user_text
from pala.behavior.schemas import env_response_format, intent_response_format
from pala.config import load_config


@dataclass
class ProbeRecord:
    ts_wall_s: float
    kind: str
    iteration: int
    image_name: str
    image_path: str
    http_ok: bool
    http_status: int
    latency_ms: float
    parse_ok: bool
    parse_stage: str
    error: Optional[str]
    response_preview: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    finish_reason: Optional[str]
    proposal_count: Optional[int] = None
    top_proposal: Optional[dict[str, Any]] = None
    env_delta_score: Optional[float] = None
    env_zone_hint: Optional[str] = None
    env_person_present: Optional[bool] = None


@dataclass
class ImageInput:
    name: str
    src_path: Path
    copied_path: Path
    data_url: str
    original_shape: list[int]
    resized_shape: list[int]
    jpeg_bytes: int


@dataclass
class DisplayConfig:
    level: str
    show_raw_content: bool
    show_http_json: bool
    max_preview_chars: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", default=None, help="Directory containing probe images.")
    p.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Explicit image paths (overrides --images-dir when provided).",
    )
    p.add_argument(
        "--patterns",
        nargs="+",
        default=["*.jpg", "*.jpeg", "*.png", "*.webp"],
        help="Glob patterns used with --images-dir.",
    )
    p.add_argument("--mode", choices=["env", "planner", "both"], default="both")
    p.add_argument("--loops", type=int, default=1, help="How many passes over the image set.")
    p.add_argument("--sleep-s", type=float, default=0.0, help="Sleep between requests.")
    p.add_argument("--request-timeout-s", type=float, default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--planner-prompt", default=None)
    p.add_argument("--planner-max-proposals", type=int, default=None)
    p.add_argument("--planner-use-env-context", action="store_true", default=None)
    p.add_argument("--planner-no-env-context", action="store_true", default=False)
    p.add_argument("--env-max-tokens", type=int, default=None)
    p.add_argument("--planner-max-tokens", type=int, default=None)
    p.add_argument("--max-width", type=int, default=None)
    p.add_argument("--jpeg-quality", type=int, default=None)
    p.add_argument("--out-dir", default=None, help="Optional output directory.")
    p.add_argument(
        "--console-level",
        choices=["full", "brief", "quiet"],
        default="full",
        help="Console detail level for request/response tracing.",
    )
    p.add_argument(
        "--show-raw-content",
        action="store_true",
        help="Include full content string from model in console output.",
    )
    p.add_argument(
        "--show-http-json",
        action="store_true",
        help="Include full HTTP response JSON in console output.",
    )
    p.add_argument(
        "--max-preview-chars",
        type=int,
        default=320,
        help="Max characters for preview fields in console and JSONL rows.",
    )
    return p.parse_args()


def _collect_image_paths(args: argparse.Namespace) -> List[Path]:
    if args.images:
        paths = [Path(p).expanduser() for p in args.images]
        return [p for p in paths if p.is_file()]

    if not args.images_dir:
        return []

    base = Path(args.images_dir).expanduser()
    if not base.is_dir():
        return []

    out: list[Path] = []
    for pat in args.patterns:
        out.extend(base.glob(pat))
    return sorted({p.resolve() for p in out if p.is_file()})


def _encode_image(path: Path, *, max_width: int, jpeg_quality: int) -> tuple[str, list[int], list[int], int]:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        original_shape = [rgb.height, rgb.width, 3]
        if max_width > 0 and rgb.width > max_width:
            new_h = int(round((max_width / float(rgb.width)) * rgb.height))
            rgb = rgb.resize((max_width, max(1, new_h)))
        resized_shape = [rgb.height, rgb.width, 3]
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        payload = buf.getvalue()
    b64 = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", original_shape, resized_shape, len(payload)


def _response_meta(resp: dict[str, Any]) -> tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    choices = resp.get("choices")
    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    finish_reason = None
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            finish_reason = c0.get("finish_reason")
    return (
        finish_reason,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )


def _preview(text: Optional[str], *, max_chars: int = 320) -> Optional[str]:
    if text is None:
        return None
    token = " ".join(str(text).split()).strip()
    if len(token) <= max_chars:
        return token
    return token[: max_chars - 3] + "..."


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True))
        f.write("\n")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _json_block(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=True)


def _print_header(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def _print_subheader(title: str) -> None:
    print("\n" + "-" * 96)
    print(title)
    print("-" * 96)


def _print_section(title: str, obj: Any) -> None:
    print(f"\n[{title}]")
    if isinstance(obj, str):
        print(obj)
    else:
        print(_json_block(obj))


def _build_env_context(current_action: dict[str, Any], env_summary: str) -> dict[str, Any]:
    return {
        "current_action": current_action,
        "latest_env_summary": env_summary,
        "recent_env_events": [],
        "control_state": None,
        "frame_timeline": [{"ordinal": 1, "age_s": 0.0}],
    }


def _build_planner_context(current_action: dict[str, Any], latest_env: dict[str, Any]) -> dict[str, Any]:
    features = latest_env.get("features") if isinstance(latest_env.get("features"), dict) else {}
    zone_hint = str(features.get("zone_hint", "unknown"))
    return {
        "current_action": {
            "primitive": current_action.get("primitive", "hold"),
            "command": current_action.get("command", {}),
            "style": current_action.get("style", "calm"),
            "confidence": float(current_action.get("confidence", 0.1)),
            "age_s": 0.0,
        },
        "signals": {
            "person_conf": None,
            "zone_hint": zone_hint,
            "env_delta": float(latest_env.get("delta_score", 0.0)),
            "activity_level": float(features.get("activity_level", 0.0)),
            "novelty": float(features.get("novelty", 0.0)),
            "person_present": bool(features.get("person_present", False)),
        },
        "latest_env": {
            "scene": str(latest_env.get("scene", "")),
            "summary": str(latest_env.get("summary_short", latest_env.get("summary", ""))),
        },
        "control_state": None,
        "planner_health": {"state": "HEALTHY"},
        "anti_collapse": {"no_commit_s": 0.0},
        "evidence_index": {"available": ["frame:latest", "env:latest", f"perception:zone:{zone_hint}"]},
    }


def _redact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    redacted = json.loads(json.dumps(payload))
    return _redact_data_urls(redacted)


_DATA_IMAGE_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


def _redact_data_urls(value: Any) -> Any:
    if isinstance(value, str):
        if _DATA_IMAGE_URL_RE.match(value.strip()):
            return f"<image_data_url chars={len(value)}>"
        return value
    if isinstance(value, list):
        return [_redact_data_urls(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_data_urls(item) for key, item in value.items()}
    return value


def _message_structure(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        row: dict[str, Any] = {"index": idx, "role": role}
        if isinstance(content, str):
            row["content_type"] = "text"
            row["chars"] = len(content)
        elif isinstance(content, list):
            image_blocks = 0
            text_blocks = 0
            text_chars = 0
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", ""))
                if item_type == "image_url":
                    image_blocks += 1
                elif item_type == "text":
                    text_blocks += 1
                    text = item.get("text")
                    if isinstance(text, str):
                        text_chars += len(text)
            row["content_type"] = "multimodal_list"
            row["image_blocks"] = image_blocks
            row["text_blocks"] = text_blocks
            row["text_chars"] = text_chars
        else:
            row["content_type"] = type(content).__name__
        out.append(row)
    return out


def _data_flow_steps(kind: str) -> list[str]:
    if kind == "env":
        return [
            "image -> JPEG/base64 data URL",
            "env context -> build_env_user_text()",
            "messages -> build_messages(system+image+user_text)",
            "payload -> add response_format=json_schema(env)",
            "POST /v1/chat/completions",
            "extract_message_content()",
            "EnvSummarizer.complete_request() -> canonical EnvSummary",
        ]
    return [
        "image -> JPEG/base64 data URL",
        "planner context -> build_planner_user_text()",
        "messages -> build_messages(system+image+user_text)",
        "payload -> add response_format=json_schema(planner)",
        "POST /v1/chat/completions",
        "extract_message_content()",
        "IntentProposer.complete_request() -> canonical proposals",
    ]


def _kind_outcome(records: list[ProbeRecord], kind: str) -> dict[str, Any]:
    entries = [r for r in records if r.kind == kind]
    if not entries:
        return {}

    lat_ok = [r.latency_ms for r in entries if r.http_ok]
    stage_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}

    for row in entries:
        stage_counts[row.parse_stage] = stage_counts.get(row.parse_stage, 0) + 1
        status_key = "ok" if row.http_ok else f"http_{row.http_status}"
        status_counts[status_key] = status_counts.get(status_key, 0) + 1
        if row.error:
            error_counts[row.error] = error_counts.get(row.error, 0) + 1

    out = {
        "count": len(entries),
        "http_status": status_counts,
        "parse_ok_count": sum(1 for r in entries if r.parse_ok),
        "parse_ok_rate": round(sum(1 for r in entries if r.parse_ok) / len(entries), 4),
        "parse_stage_counts": stage_counts,
        "latency_ms": {},
        "errors_top": sorted(
            ({"error": err, "count": count} for err, count in error_counts.items()),
            key=lambda x: (-x["count"], x["error"]),
        )[:10],
    }
    if lat_ok:
        lat_sorted = sorted(lat_ok)
        out["latency_ms"] = {
            "p50": round(lat_sorted[len(lat_sorted) // 2], 1),
            "p90": round(lat_sorted[int(0.9 * (len(lat_sorted) - 1))], 1),
            "max": round(max(lat_sorted), 1),
            "mean": round(statistics.mean(lat_sorted), 1),
        }
    if kind == "planner":
        top_primitives: dict[str, int] = {}
        for row in entries:
            primitive = (row.top_proposal or {}).get("primitive")
            if primitive:
                top_primitives[str(primitive)] = top_primitives.get(str(primitive), 0) + 1
        out["top_primitive_counts"] = top_primitives
    return out


def _summarize_records(records: Iterable[ProbeRecord]) -> dict[str, Any]:
    rows = list(records)
    return {
        "total_requests": len(rows),
        "kinds": {
            "env": _kind_outcome(rows, "env"),
            "planner": _kind_outcome(rows, "planner"),
        },
    }


def _record_from_env(
    *,
    loop_idx: int,
    image: ImageInput,
    result: Any,
    parsed: Any,
    parser: EnvSummarizer,
    content: Optional[str],
    finish_reason: Optional[str],
    ptok: Optional[int],
    ctok: Optional[int],
    ttok: Optional[int],
    latest_env: dict[str, Any],
    max_preview_chars: int,
) -> ProbeRecord:
    parse_ok = bool(parsed) and bool(result.ok)
    error = None
    if not result.ok:
        error = result.error
    elif not parsed:
        error = f"env_parse_failed:{parser.last_parse_error}"
    return ProbeRecord(
        ts_wall_s=time.time(),
        kind="env",
        iteration=loop_idx,
        image_name=image.name,
        image_path=str(image.copied_path),
        http_ok=result.ok,
        http_status=result.status_code,
        latency_ms=round(result.latency_ms, 1),
        parse_ok=parse_ok,
        parse_stage=parser.last_parse_stage,
        error=error,
        response_preview=_preview(content, max_chars=max_preview_chars),
        prompt_tokens=ptok,
        completion_tokens=ctok,
        total_tokens=ttok,
        finish_reason=finish_reason,
        env_delta_score=None if not parse_ok else float(latest_env.get("delta_score", 0.0)),
        env_zone_hint=None if not parse_ok else str((latest_env.get("features") or {}).get("zone_hint", "unknown")),
        env_person_present=None if not parse_ok else bool((latest_env.get("features") or {}).get("person_present", False)),
    )


def _record_from_planner(
    *,
    loop_idx: int,
    image: ImageInput,
    result: Any,
    parsed: Any,
    parser: IntentProposer,
    content: Optional[str],
    finish_reason: Optional[str],
    ptok: Optional[int],
    ctok: Optional[int],
    ttok: Optional[int],
    max_preview_chars: int,
) -> ProbeRecord:
    parse_ok = bool(parsed) and bool(result.ok)
    error = None
    proposal_count = 0
    top_proposal = None
    if not result.ok:
        error = result.error
    elif not parsed:
        error = f"planner_parse_failed:{parser.last_parse_error}"
    else:
        proposal_count = len(parsed.response.proposals)
        if parsed.response.proposals:
            top = parsed.response.proposals[0]
            top_proposal = {
                "intent": top.intent,
                "primitive": top.primitive,
                "score": top.score,
                "confidence": top.confidence,
            }
    return ProbeRecord(
        ts_wall_s=time.time(),
        kind="planner",
        iteration=loop_idx,
        image_name=image.name,
        image_path=str(image.copied_path),
        http_ok=result.ok,
        http_status=result.status_code,
        latency_ms=round(result.latency_ms, 1),
        parse_ok=parse_ok,
        parse_stage=parser.last_parse_stage,
        error=error,
        response_preview=_preview(content, max_chars=max_preview_chars),
        prompt_tokens=ptok,
        completion_tokens=ctok,
        total_tokens=ttok,
        finish_reason=finish_reason,
        proposal_count=proposal_count,
        top_proposal=top_proposal,
    )


def _request_artifact(
    *,
    request_id: int,
    kind: str,
    loop_idx: int,
    image: ImageInput,
    context: dict[str, Any],
    user_text: str,
    messages: list[dict[str, Any]],
    payload: dict[str, Any],
    result: Any,
    content: Optional[str],
    reasoning: Optional[str],
    parsed_obj: Optional[dict[str, Any]],
    parse_ok: bool,
    parse_stage: str,
    parse_error: Optional[str],
    show_preview_chars: int,
) -> dict[str, Any]:
    redacted_content = _redact_data_urls(content)
    redacted_reasoning = _redact_data_urls(reasoning)
    response_meta = {
        "http_ok": result.ok,
        "http_status": result.status_code,
        "latency_ms": round(result.latency_ms, 1),
    }
    if isinstance(result.response_json, dict):
        finish_reason, ptok, ctok, ttok = _response_meta(result.response_json)
        response_meta.update(
            {
                "finish_reason": finish_reason,
                "prompt_tokens": ptok,
                "completion_tokens": ctok,
                "total_tokens": ttok,
            }
        )
    if result.error:
        response_meta["error"] = result.error

    return {
        "request_id": request_id,
        "kind": kind,
        "iteration": loop_idx,
        "image": {
            "name": image.name,
            "path": str(image.copied_path),
            "original_shape": image.original_shape,
            "resized_shape": image.resized_shape,
            "jpeg_bytes": image.jpeg_bytes,
        },
        "data_flow": _data_flow_steps(kind),
        "input_context": context,
        "prompt": {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_text,
        },
        "messages_structure": _message_structure(messages),
        "request_payload_redacted": _redact_payload(payload),
        "request_payload_bytes": len(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")),
        "response": response_meta,
        "message_content_preview": _preview(redacted_content, max_chars=show_preview_chars),
        "reasoning_content_preview": _preview(redacted_reasoning, max_chars=show_preview_chars),
        "parse": {
            "ok": parse_ok,
            "stage": parse_stage,
            "error": parse_error,
            "canonical_output": parsed_obj,
        },
    }


def _print_request(artifact: dict[str, Any], *, cfg: DisplayConfig) -> None:
    if cfg.level == "quiet":
        return

    response = artifact["response"]
    parse = artifact["parse"]
    header = (
        f"{artifact['kind'].upper()} req#{artifact['request_id']} "
        f"loop={artifact['iteration']} image={artifact['image']['name']} "
        f"http={response.get('http_status')} parse={'ok' if parse.get('ok') else 'fail'} "
        f"stage={parse.get('stage')} latency_ms={response.get('latency_ms')}"
    )
    _print_subheader(header)

    if cfg.level == "brief":
        if response.get("error"):
            print(f"error={response['error']}")
        if parse.get("error"):
            print(f"parse_error={parse['error']}")
        preview = artifact.get("message_content_preview")
        if preview:
            print(f"preview={preview}")
        return

    _print_section("Data Flow", artifact["data_flow"])
    _print_section("Image Input", artifact["image"])
    _print_section("Input Context", artifact["input_context"])
    _print_section("System Prompt", artifact["prompt"]["system_prompt"])
    _print_section("User Prompt", artifact["prompt"]["user_prompt"])
    _print_section("Messages Structure", artifact["messages_structure"])
    _print_section("Request Payload (Redacted)", artifact["request_payload_redacted"])
    _print_section("Response Metadata", artifact["response"])
    if artifact.get("message_content_preview"):
        _print_section("Message Content (Preview)", artifact["message_content_preview"])
    if artifact.get("reasoning_content_preview"):
        _print_section("Reasoning Content (Preview)", artifact["reasoning_content_preview"])
    if cfg.show_raw_content and isinstance(artifact.get("raw_message_content"), str):
        _print_section("Message Content (Raw)", artifact["raw_message_content"])
    _print_section("Parser Output", artifact["parse"])
    if cfg.show_http_json and isinstance(artifact.get("http_response_json"), dict):
        _print_section("HTTP Response JSON", artifact["http_response_json"])


def main() -> int:
    args = _parse_args()
    display = DisplayConfig(
        level=str(args.console_level),
        show_raw_content=bool(args.show_raw_content),
        show_http_json=bool(args.show_http_json),
        max_preview_chars=max(40, int(args.max_preview_chars)),
    )
    cfg = load_config("config/robot.yaml")
    cosmos = cfg.cosmos

    base_url = args.base_url or os.getenv("PALA_COSMOS_BASE_URL") or cosmos.base_url
    provider = (
        os.getenv("PALA_MODEL_PROVIDER")
        or os.getenv("PALA_COSMOS_PROVIDER")
        or getattr(cosmos, "provider", "auto")
        or "auto"
    )
    api_key = args.api_key or os.getenv("PALA_COSMOS_API_KEY")
    model = args.model or os.getenv("PALA_COSMOS_MODEL") or cosmos.model
    planner_prompt = args.planner_prompt or os.getenv("PALA_COSMOS_PROMPT") or cosmos.planner_prompt
    chat_url = normalize_chat_url(str(base_url or ""), provider=str(provider))

    if not chat_url:
        raise SystemExit("Missing Cosmos base URL (set --base-url or PALA_COSMOS_BASE_URL).")

    image_paths = _collect_image_paths(args)
    if not image_paths:
        raise SystemExit("No images found. Provide --images or --images-dir.")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else Path("logs/cosmos_api_probe") / run_name
    images_out = out_dir / "images"
    requests_out = out_dir / "requests"
    images_out.mkdir(parents=True, exist_ok=True)
    requests_out.mkdir(parents=True, exist_ok=True)

    planner_max_proposals = (
        int(args.planner_max_proposals)
        if args.planner_max_proposals is not None
        else int(getattr(cosmos, "planner_max_proposals", 3))
    )
    if args.planner_no_env_context:
        planner_use_env_context = False
    elif args.planner_use_env_context is None:
        planner_use_env_context = bool(getattr(cosmos, "planner_use_env_context", True))
    else:
        planner_use_env_context = True

    timeout_s = (
        float(args.request_timeout_s)
        if args.request_timeout_s is not None
        else max(1.0, float(getattr(cosmos, "request_timeout_ms", 20000)) / 1000.0)
    )
    env_max_tokens = int(args.env_max_tokens) if args.env_max_tokens is not None else int(getattr(cosmos, "env_max_tokens", 600))
    planner_max_tokens = (
        int(args.planner_max_tokens)
        if args.planner_max_tokens is not None
        else int(getattr(cosmos, "planner_max_tokens", 360))
    )
    max_width = int(args.max_width) if args.max_width is not None else int(getattr(cosmos, "summary_max_width", 320))
    jpeg_quality = (
        int(args.jpeg_quality)
        if args.jpeg_quality is not None
        else int(getattr(cosmos, "summary_jpeg_quality", 55))
    )

    image_inputs: list[ImageInput] = []
    for idx, src in enumerate(image_paths, start=1):
        copied = images_out / f"{idx:03d}_{src.name}"
        shutil.copy2(src, copied)
        data_url, original_shape, resized_shape, jpeg_bytes = _encode_image(
            copied,
            max_width=max_width,
            jpeg_quality=jpeg_quality,
        )
        image_inputs.append(
            ImageInput(
                name=src.name,
                src_path=src,
                copied_path=copied,
                data_url=data_url,
                original_shape=original_shape,
                resized_shape=resized_shape,
                jpeg_bytes=jpeg_bytes,
            )
        )

    manifest = [
        {
            "name": img.name,
            "path": str(img.copied_path),
            "original_shape": img.original_shape,
            "resized_shape": img.resized_shape,
            "jpeg_bytes": img.jpeg_bytes,
        }
        for img in image_inputs
    ]
    run_config = {
        "mode": args.mode,
        "loops": int(args.loops),
        "images": len(image_inputs),
        "base_url": chat_url,
        "provider": str(provider),
        "model": model,
        "planner_prompt": planner_prompt,
        "planner_max_proposals": planner_max_proposals,
        "planner_use_env_context": planner_use_env_context,
        "env_max_tokens": env_max_tokens,
        "planner_max_tokens": planner_max_tokens,
        "max_width": max_width,
        "jpeg_quality": jpeg_quality,
        "timeout_s": timeout_s,
    }
    _write_json(out_dir / "run_config.json", run_config)
    _write_json(out_dir / "image_manifest.json", manifest)

    if display.level != "quiet":
        _print_header("Cosmos API Test Run")
        _print_section("Run Config", run_config)
        _print_section("Image Manifest", manifest)
        print(f"\nArtifacts directory: {out_dir}")

    env_parser = EnvSummarizer()
    planner_parser = IntentProposer()
    current_action: dict[str, Any] = {"primitive": "hold", "command": {}, "style": "calm", "confidence": 0.1}
    latest_env: dict[str, Any] = {}

    records: list[ProbeRecord] = []
    env_log = out_dir / "env_requests.jsonl"
    planner_log = out_dir / "planner_requests.jsonl"

    request_idx = 0
    for loop_idx in range(1, max(1, int(args.loops)) + 1):
        for image in image_inputs:
            if args.mode in {"env", "both"}:
                request_idx += 1
                env_context = _build_env_context(current_action=current_action, env_summary=str(latest_env.get("summary_short", "")))
                user_text = build_env_user_text(
                    context=env_context,
                    policy_identity=str(getattr(cosmos, "policy_identity", "PALA")),
                )
                messages = build_messages(user_text=user_text, image_data_urls=[image.data_url])
                body = {
                    "model": str(model),
                    "messages": messages,
                    "temperature": 0.0,
                    "top_p": 0.3,
                    "presence_penalty": 0.0,
                    "max_tokens": env_max_tokens,
                    "stream": False,
                    "response_format": env_response_format(),
                }
                result = post_chat_json(
                    url=chat_url,
                    payload=body,
                    timeout_s=timeout_s,
                    api_key=api_key,
                    provider=str(provider),
                )
                content, reasoning = extract_message_content(result.response_json or {}) if result.response_json else (None, None)
                finish_reason, ptok, ctok, ttok = (
                    _response_meta(result.response_json or {}) if result.response_json else (None, None, None, None)
                )
                env_parser.submit_or_replace({"id": request_idx})
                parsed = env_parser.complete_request(content or "")
                parsed_obj = asdict(parsed.summary) if parsed else None
                if parsed_obj:
                    latest_env = parsed_obj

                rec = _record_from_env(
                    loop_idx=loop_idx,
                    image=image,
                    result=result,
                    parsed=parsed,
                    parser=env_parser,
                    content=content,
                    finish_reason=finish_reason,
                    ptok=ptok,
                    ctok=ctok,
                    ttok=ttok,
                    latest_env=latest_env,
                    max_preview_chars=display.max_preview_chars,
                )
                records.append(rec)
                _write_jsonl(env_log, asdict(rec))

                artifact = _request_artifact(
                    request_id=request_idx,
                    kind="env",
                    loop_idx=loop_idx,
                    image=image,
                    context=env_context,
                    user_text=user_text,
                    messages=messages,
                    payload=body,
                    result=result,
                    content=content,
                    reasoning=reasoning,
                    parsed_obj=parsed_obj,
                    parse_ok=rec.parse_ok,
                    parse_stage=rec.parse_stage,
                    parse_error=rec.error,
                    show_preview_chars=display.max_preview_chars,
                )
                artifact["raw_message_content"] = _redact_data_urls(content)
                artifact["http_response_json"] = _redact_data_urls(result.response_json)
                _write_json(requests_out / f"{request_idx:04d}_env.json", artifact)
                _print_request(artifact, cfg=display)

            if args.mode in {"planner", "both"}:
                request_idx += 1
                planner_context = _build_planner_context(
                    current_action=current_action,
                    latest_env=latest_env if planner_use_env_context else {},
                )
                user_text = build_planner_user_text(
                    context=planner_context,
                    policy_identity=str(getattr(cosmos, "policy_identity", "PALA")),
                    policy_capabilities=str(getattr(cosmos, "policy_capabilities", "")),
                    policy_safety=str(getattr(cosmos, "policy_safety", "")),
                    policy_style=str(getattr(cosmos, "policy_style", "calm")),
                    planner_prompt=str(planner_prompt or ""),
                    max_proposals=max(1, planner_max_proposals),
                )
                messages = build_messages(user_text=user_text, image_data_urls=[image.data_url])
                body = {
                    "model": str(model),
                    "messages": messages,
                    "temperature": 0.0,
                    "top_p": 0.3,
                    "presence_penalty": 0.0,
                    "max_tokens": planner_max_tokens,
                    "stream": False,
                    "response_format": intent_response_format(),
                }
                result = post_chat_json(
                    url=chat_url,
                    payload=body,
                    timeout_s=timeout_s,
                    api_key=api_key,
                    provider=str(provider),
                )
                content, reasoning = extract_message_content(result.response_json or {}) if result.response_json else (None, None)
                finish_reason, ptok, ctok, ttok = (
                    _response_meta(result.response_json or {}) if result.response_json else (None, None, None, None)
                )
                planner_parser.submit_or_replace({"id": request_idx})
                parsed = planner_parser.complete_request(content or "")
                parsed_obj = asdict(parsed.response) if parsed else None
                if parsed and parsed.response.proposals:
                    top = parsed.response.proposals[0]
                    current_action = {
                        "primitive": top.primitive,
                        "command": dict(top.command),
                        "style": top.style,
                        "confidence": top.confidence,
                    }

                rec = _record_from_planner(
                    loop_idx=loop_idx,
                    image=image,
                    result=result,
                    parsed=parsed,
                    parser=planner_parser,
                    content=content,
                    finish_reason=finish_reason,
                    ptok=ptok,
                    ctok=ctok,
                    ttok=ttok,
                    max_preview_chars=display.max_preview_chars,
                )
                records.append(rec)
                _write_jsonl(planner_log, asdict(rec))

                artifact = _request_artifact(
                    request_id=request_idx,
                    kind="planner",
                    loop_idx=loop_idx,
                    image=image,
                    context=planner_context,
                    user_text=user_text,
                    messages=messages,
                    payload=body,
                    result=result,
                    content=content,
                    reasoning=reasoning,
                    parsed_obj=parsed_obj,
                    parse_ok=rec.parse_ok,
                    parse_stage=rec.parse_stage,
                    parse_error=rec.error,
                    show_preview_chars=display.max_preview_chars,
                )
                artifact["raw_message_content"] = _redact_data_urls(content)
                artifact["http_response_json"] = _redact_data_urls(result.response_json)
                _write_json(requests_out / f"{request_idx:04d}_planner.json", artifact)
                _print_request(artifact, cfg=display)

            if args.sleep_s > 0.0:
                time.sleep(float(args.sleep_s))

    summary = _summarize_records(records)
    summary["config"] = run_config
    summary["image_manifest"] = manifest
    summary_path = out_dir / "summary.json"
    _write_json(summary_path, summary)

    _print_header("Final Results")
    _print_section("Summary", summary)
    print(f"\nWrote probe run: {out_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"Per-request artifacts: {requests_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
