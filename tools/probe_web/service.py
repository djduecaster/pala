from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import UploadFile
from PIL import Image

from pala.behavior import ContextBuilder, IntentProposer
from pala.behavior.env_summarizer import EnvSummarizer
from pala.behavior.model_clients import extract_message_content, normalize_chat_url, post_chat_json
from pala.behavior.prompts import build_env_user_text, build_messages, build_planner_user_text
from pala.behavior.schemas import ENV_SUMMARY_SCHEMA, INTENT_PROPOSALS_SCHEMA, env_response_format, intent_response_format
from pala.types import ActionPlan, HoldCommand, PrimitiveKind

from . import storage
from .defaults import API_KEY_ENV_VAR, resolve_api_key
from .models import EnvPlannerProbeParams, EnvProbeParams, EnvProbeRun, PreparedImage


_DATA_IMAGE_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


class ProbeInputError(ValueError):
    pass


def _safe_filename(name: str, *, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]", "_", str(name or "")).strip("._")
    if not token:
        return fallback
    return token[:120]


def _json_preview(value: Any, *, max_chars: int = 240) -> str:
    token = " ".join(str(value or "").split()).strip()
    if len(token) <= max_chars:
        return token
    return token[: max_chars - 3] + "..."


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


def _response_meta(resp: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    choices = resp.get("choices") if isinstance(resp.get("choices"), list) else []
    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    finish_reason = None
    if choices:
        first = choices[0]
        if isinstance(first, dict):
            finish_reason = first.get("finish_reason")
    return (
        finish_reason,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )


def _message_structure(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        row: Dict[str, Any] = {"index": idx, "role": role}
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
                if item.get("type") == "image_url":
                    image_blocks += 1
                elif item.get("type") == "text":
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


def parse_image_order(order_text: str, *, file_count: int) -> List[int]:
    if file_count <= 0:
        raise ProbeInputError("No images were provided.")
    token = str(order_text or "").strip()
    if not token:
        return list(range(file_count))
    try:
        parts = [int(item.strip()) for item in token.split(",") if item.strip() != ""]
    except ValueError as exc:
        raise ProbeInputError("Invalid image ordering payload.") from exc
    if len(parts) != file_count:
        raise ProbeInputError("Image ordering did not match uploaded image count.")
    if sorted(parts) != list(range(file_count)):
        raise ProbeInputError("Image ordering must reference each uploaded image exactly once.")
    return parts


async def prepare_images(
    upload_files: List[UploadFile],
    *,
    image_order: str,
    jpeg_quality: int = 85,
    max_upload_bytes: int = 25 * 1024 * 1024,
) -> tuple[List[PreparedImage], List[bytes]]:
    if len(upload_files) != 4:
        raise ProbeInputError("Exactly 4 images are required per probe run.")

    order = parse_image_order(image_order, file_count=len(upload_files))
    selected = [upload_files[i] for i in order]

    prepared: List[PreparedImage] = []
    jpeg_payloads: List[bytes] = []

    for idx, item in enumerate(selected, start=1):
        raw = await item.read()
        if not raw:
            raise ProbeInputError(f"Image slot {idx} is empty or unreadable.")
        if len(raw) > max_upload_bytes:
            raise ProbeInputError(f"Image '{item.filename or idx}' exceeds upload size limit.")

        try:
            with Image.open(io.BytesIO(raw)) as img:
                rgb = img.convert("RGB")
                width, height = rgb.size
                out = io.BytesIO()
                rgb.save(out, format="JPEG", quality=max(30, min(95, int(jpeg_quality))), optimize=True)
        except Exception as exc:
            raise ProbeInputError(f"Failed to decode image '{item.filename or idx}': {exc}") from exc

        payload = out.getvalue()
        jpeg_payloads.append(payload)
        data_url = "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")

        prepared.append(
            PreparedImage(
                filename=_safe_filename(item.filename or "", fallback=f"image_{idx}.jpg"),
                content_type=str(item.content_type or "application/octet-stream"),
                original_width=int(width),
                original_height=int(height),
                encoded_width=int(width),
                encoded_height=int(height),
                jpeg_bytes=len(payload),
                data_url=data_url,
            )
        )

    return prepared, jpeg_payloads


def _bootstrap_action() -> ActionPlan:
    return ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.1,
        style="calm",
        explanation="probe_bootstrap",
    )


def _build_frame_timeline(count: int, *, inter_frame_ms: float) -> List[Dict[str, float]]:
    span_ms = max(1.0, float(inter_frame_ms))
    out: List[Dict[str, float]] = []
    for idx in range(count):
        age_s = ((count - 1) - idx) * (span_ms / 1000.0)
        out.append({"ordinal": idx + 1, "age_s": round(age_s, 3)})
    return out


def _build_env_context(frame_timeline: List[Dict[str, float]]) -> Dict[str, Any]:
    builder = ContextBuilder()
    world_snapshot = {
        "latest_env_snapshot": {},
        "event_tail": [],
        "control_state_latest": "unknown",
    }
    return builder.build_env_context(
        world_snapshot=world_snapshot,
        current_action=_bootstrap_action(),
        frame_timeline=frame_timeline,
        mode="idle_presence",
    )


def _build_planner_context(*, latest_env: Dict[str, Any], use_env_context: bool) -> Dict[str, Any]:
    builder = ContextBuilder()
    snapshot = {
        "latest_env_snapshot": latest_env if use_env_context else {},
        "event_tail": [],
        "control_state_latest": "unknown",
    }
    return builder.build_planner_context(
        st=None,
        world_snapshot=snapshot,
        current_action=_bootstrap_action(),
        planner_health={"state": "HEALTHY", "last_latency_ms": 0.0, "no_signal_streak": 0},
        mode="idle_presence",
        now_mono_s=1.0,
        last_commit_mono_s=0.0,
        no_commit_s=1.0,
    )


def _env_packet_compact(
    *,
    params: EnvProbeParams,
    chat_url: str,
    images: List[Dict[str, Any]],
    frame_timeline: List[Dict[str, float]],
    user_text: str,
) -> List[Dict[str, str]]:
    return [
        {"label": "Provider", "value": params.provider},
        {"label": "Model", "value": params.model},
        {"label": "Chat URL", "value": chat_url},
        {"label": "Image Count", "value": str(len(images))},
        {"label": "Timeout (s)", "value": f"{params.timeout_s:.2f}"},
        {"label": "Max Tokens", "value": str(params.env_max_tokens)},
        {"label": "Temperature", "value": f"{params.temperature:.3f}"},
        {"label": "Top P", "value": f"{params.top_p:.3f}"},
        {"label": "Presence Penalty", "value": f"{params.presence_penalty:.3f}"},
        {"label": "Frame Timeline", "value": ", ".join(f"#{i['ordinal']} age={i['age_s']}s" for i in frame_timeline)},
        {"label": "User Prompt Chars", "value": str(len(user_text))},
    ]


def _env_packet_expanded(
    *,
    params: EnvProbeParams,
    frame_timeline: List[Dict[str, float]],
    env_context: Dict[str, Any],
    response_format: Dict[str, Any],
    user_text: str,
    images: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    schema_required = ENV_SUMMARY_SCHEMA.get("required") if isinstance(ENV_SUMMARY_SCHEMA, dict) else []
    feature_required = (
        ENV_SUMMARY_SCHEMA.get("properties", {}).get("features", {}).get("required", [])
        if isinstance(ENV_SUMMARY_SCHEMA, dict)
        else []
    )
    return [
        {
            "title": "Request Target",
            "rows": [
                {"label": "Provider", "value": params.provider},
                {"label": "Model", "value": params.model},
                {"label": "Base URL", "value": params.base_url},
                {"label": "Timeout (s)", "value": params.timeout_s},
            ],
        },
        {
            "title": "Request Knobs",
            "rows": [
                {"label": "max_tokens", "value": params.env_max_tokens},
                {"label": "temperature", "value": params.temperature},
                {"label": "top_p", "value": params.top_p},
                {"label": "presence_penalty", "value": params.presence_penalty},
                {"label": "inter_frame_ms", "value": params.inter_frame_ms},
            ],
        },
        {
            "title": "Prompt Fields",
            "rows": [
                {"label": "System prompt", "value": params.system_prompt},
                {"label": "Env contract", "value": params.env_contract},
                {"label": "Policy identity", "value": params.policy_identity},
                {"label": "User prompt", "value": user_text},
            ],
        },
        {
            "title": "Context Fields",
            "rows": [
                {"label": "mode", "value": env_context.get("mode")},
                {
                    "label": "current_action",
                    "value": _json_preview(env_context.get("current_action"), max_chars=260),
                },
                {
                    "label": "latest_env_summary",
                    "value": env_context.get("latest_env_summary", ""),
                },
                {
                    "label": "frame_timeline",
                    "value": _json_preview(frame_timeline, max_chars=260),
                },
            ],
        },
        {
            "title": "Image Packet",
            "rows": [
                {
                    "label": f"Image #{idx + 1}",
                    "value": (
                        f"{img['filename']} | {img['original_width']}x{img['original_height']} | "
                        f"jpeg_bytes={img['jpeg_bytes']}"
                    ),
                }
                for idx, img in enumerate(images)
            ],
        },
        {
            "title": "Response Format Fields",
            "rows": [
                {"label": "type", "value": response_format.get("type")},
                {
                    "label": "json_schema.name",
                    "value": response_format.get("json_schema", {}).get("name"),
                },
                {
                    "label": "json_schema.strict",
                    "value": response_format.get("json_schema", {}).get("strict"),
                },
                {
                    "label": "schema.required (top-level)",
                    "value": ", ".join(str(item) for item in schema_required),
                },
                {
                    "label": "features.required",
                    "value": ", ".join(str(item) for item in feature_required),
                },
            ],
        },
    ]


def _planner_packet_compact(
    *,
    params: EnvPlannerProbeParams,
    chat_url: str,
    planner_image_count: int,
    planner_image_indices: str,
    user_text: str,
) -> List[Dict[str, str]]:
    return [
        {"label": "Provider", "value": params.provider},
        {"label": "Model", "value": params.model},
        {"label": "Chat URL", "value": chat_url},
        {"label": "Planner Image Count", "value": str(planner_image_count)},
        {"label": "Timeout (s)", "value": f"{params.timeout_s:.2f}"},
        {"label": "Max Tokens", "value": str(params.planner_max_tokens)},
        {"label": "Temperature", "value": f"{params.planner_temperature:.3f}"},
        {"label": "Top P", "value": f"{params.planner_top_p:.3f}"},
        {"label": "Presence Penalty", "value": f"{params.planner_presence_penalty:.3f}"},
        {"label": "Planner Image Indices", "value": planner_image_indices},
        {"label": "Planner Prompt Chars", "value": str(len(params.planner_prompt))},
        {"label": "User Prompt Chars", "value": str(len(user_text))},
    ]


def _planner_packet_expanded(
    *,
    params: EnvPlannerProbeParams,
    planner_system_prompt: str,
    planner_context: Dict[str, Any],
    response_format: Dict[str, Any],
    user_text: str,
    planner_image_manifest: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    schema_required = INTENT_PROPOSALS_SCHEMA.get("required") if isinstance(INTENT_PROPOSALS_SCHEMA, dict) else []
    proposal_required = (
        INTENT_PROPOSALS_SCHEMA.get("$defs", {}).get("Proposal", {}).get("required", [])
        if isinstance(INTENT_PROPOSALS_SCHEMA, dict)
        else []
    )
    return [
        {
            "title": "Request Target",
            "rows": [
                {"label": "Provider", "value": params.provider},
                {"label": "Model", "value": params.model},
                {"label": "Base URL", "value": params.base_url},
                {"label": "Timeout (s)", "value": params.timeout_s},
            ],
        },
        {
            "title": "Request Knobs",
            "rows": [
                {"label": "planner_max_tokens", "value": params.planner_max_tokens},
                {"label": "temperature", "value": params.planner_temperature},
                {"label": "top_p", "value": params.planner_top_p},
                {"label": "presence_penalty", "value": params.planner_presence_penalty},
                {"label": "planner_max_proposals", "value": params.planner_max_proposals},
                {"label": "planner_use_env_context", "value": params.planner_use_env_context},
            ],
        },
        {
            "title": "Prompt Fields",
            "rows": [
                {"label": "System prompt", "value": planner_system_prompt},
                {"label": "Policy identity", "value": params.policy_identity},
                {"label": "Policy capabilities", "value": params.policy_capabilities},
                {"label": "Policy safety", "value": params.policy_safety},
                {"label": "Policy style", "value": params.policy_style},
                {"label": "Planner prompt", "value": params.planner_prompt},
                {"label": "User prompt", "value": user_text},
                {"label": "Planner user text override", "value": params.planner_user_text_override or ""},
                {"label": "Planner context override JSON", "value": params.planner_context_override_json or ""},
                {"label": "Planner payload override JSON", "value": params.planner_payload_override_json or ""},
            ],
        },
        {
            "title": "Context Fields",
            "rows": [
                {"label": "mode", "value": planner_context.get("mode")},
                {
                    "label": "current_action",
                    "value": _json_preview(planner_context.get("current_action"), max_chars=260),
                },
                {
                    "label": "signals",
                    "value": _json_preview(planner_context.get("signals"), max_chars=260),
                },
                {
                    "label": "latest_env",
                    "value": _json_preview(planner_context.get("latest_env"), max_chars=260),
                },
                {
                    "label": "planner_health",
                    "value": _json_preview(planner_context.get("planner_health"), max_chars=260),
                },
            ],
        },
        {
            "title": "Planner Image Input",
            "rows": [
                {
                    "label": f"Image #{idx + 1}",
                    "value": (
                        f"packet_index={img['packet_index']} {img['filename']} | "
                        f"{img['original_width']}x{img['original_height']} | jpeg_bytes={img['jpeg_bytes']}"
                    ),
                }
                for idx, img in enumerate(planner_image_manifest)
            ]
            + [{"label": "policy", "value": "latest-frame-only"}],
        },
        {
            "title": "Response Format Fields",
            "rows": [
                {"label": "type", "value": response_format.get("type")},
                {
                    "label": "json_schema.name",
                    "value": response_format.get("json_schema", {}).get("name"),
                },
                {
                    "label": "json_schema.strict",
                    "value": response_format.get("json_schema", {}).get("strict"),
                },
                {
                    "label": "schema.required (top-level)",
                    "value": ", ".join(str(item) for item in schema_required),
                },
                {
                    "label": "proposal.required",
                    "value": ", ".join(str(item) for item in proposal_required),
                },
            ],
        },
    ]


def _build_env_request_payload(*, params: EnvProbeParams, user_text: str, image_data_urls: List[str]) -> Dict[str, Any]:
    user_content: List[Dict[str, Any]] = []
    for url in image_data_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    user_content.append({"type": "text", "text": user_text})
    return {
        "model": str(params.model),
        "messages": [
            {"role": "system", "content": params.system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": float(params.temperature),
        "top_p": float(params.top_p),
        "presence_penalty": float(params.presence_penalty),
        "max_tokens": int(params.env_max_tokens),
        "stream": False,
        "response_format": env_response_format(),
    }


def _build_planner_request_payload(
    *,
    params: EnvPlannerProbeParams,
    planner_system_prompt: str,
    planner_user_text: str,
    planner_image_data_urls: List[str],
) -> Dict[str, Any]:
    messages = build_messages(user_text=planner_user_text, image_data_urls=planner_image_data_urls)
    if messages and isinstance(messages[0], dict):
        messages[0]["content"] = planner_system_prompt
    return {
        "model": str(params.model),
        "messages": messages,
        "temperature": float(params.planner_temperature),
        "top_p": float(params.planner_top_p),
        "presence_penalty": float(params.planner_presence_penalty),
        "max_tokens": int(params.planner_max_tokens),
        "stream": False,
        "response_format": intent_response_format(),
    }


def _result_finish_reason(result: Any) -> str:
    response_json = getattr(result, "response_json", None)
    if not isinstance(response_json, dict):
        return ""
    finish_reason, _ptok, _ctok, _ttok = _response_meta(response_json)
    return str(finish_reason or "").strip().lower()


def _env_error_from_result(result: Any, parser: EnvSummarizer, parsed: Any) -> str | None:
    if not bool(getattr(result, "ok", False)):
        return str(getattr(result, "error", "transport_error"))
    if parsed is None:
        detail = parser.last_parse_error or "unknown"
        if _result_finish_reason(result) == "length":
            return f"env_parse_failed:truncated_response:finish_reason=length:likely_token_budget_exhausted:{detail}"
        return f"env_parse_failed:{detail}"
    return None


def _planner_error_from_result(result: Any, parser: IntentProposer, parsed: Any) -> str | None:
    if not bool(getattr(result, "ok", False)):
        return str(getattr(result, "error", "transport_error"))
    if parsed is None:
        detail = parser.last_parse_error or "unknown"
        if _result_finish_reason(result) == "length":
            return f"planner_parse_failed:truncated_response:finish_reason=length:likely_token_budget_exhausted:{detail}"
        return f"planner_parse_failed:{detail}"
    return None


def _ensure_image_url(logs_root: Path, run_id: str, rel_path: str) -> str:
    _ = logs_root
    safe_rel = str(rel_path).replace("\\", "/")
    return f"/probe-logs/{run_id}/{safe_rel}"


def _coerce_float(value: Any, *, lo: float, hi: float) -> float:
    try:
        token = float(value)
    except (TypeError, ValueError):
        token = lo
    if token < lo:
        return lo
    if token > hi:
        return hi
    return token


def _coerce_int(value: Any, *, lo: int, hi: int) -> int:
    try:
        token = int(value)
    except (TypeError, ValueError):
        token = lo
    if token < lo:
        return lo
    if token > hi:
        return hi
    return token


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _parse_json_object(raw: str, *, field_name: str) -> Dict[str, Any]:
    token = str(raw or "").strip()
    if not token:
        raise ProbeInputError(f"{field_name} is empty.")
    try:
        obj = json.loads(token)
    except json.JSONDecodeError as exc:
        raise ProbeInputError(f"Invalid JSON for {field_name}: {exc.msg}.") from exc
    if not isinstance(obj, dict):
        raise ProbeInputError(f"{field_name} must decode to a JSON object.")
    return obj


def _parse_planner_image_indices(indices_text: str, *, image_count: int) -> List[int]:
    if image_count <= 0:
        return []
    token = str(indices_text or "").strip()
    if not token:
        return [image_count]
    try:
        parts = [int(item.strip()) for item in token.split(",") if item.strip()]
    except ValueError as exc:
        raise ProbeInputError("Planner image indices must be comma-separated integers.") from exc
    if not parts:
        return [image_count]
    out: List[int] = []
    seen: set[int] = set()
    for part in parts:
        if part < 1 or part > image_count:
            raise ProbeInputError(f"Planner image index out of range: {part} (valid 1..{image_count}).")
        if part in seen:
            continue
        seen.add(part)
        out.append(part)
    return out


def normalize_params(raw: Dict[str, Any], *, defaults: Any) -> EnvProbeParams:
    provider = str(raw.get("provider", defaults.provider)).strip().lower() or defaults.provider
    model = str(raw.get("model", defaults.model)).strip() or defaults.model
    base_url = str(raw.get("base_url", defaults.base_url)).strip() or defaults.base_url
    system_prompt = str(raw.get("system_prompt", defaults.system_prompt)).strip() or defaults.system_prompt
    env_contract = str(raw.get("env_contract", defaults.env_contract)).strip() or defaults.env_contract
    policy_identity = str(raw.get("policy_identity", defaults.policy_identity)).strip() or defaults.policy_identity
    packet_view_mode = str(raw.get("packet_view_mode", defaults.packet_view_mode)).strip().lower()
    if packet_view_mode not in {"compact", "expanded"}:
        packet_view_mode = defaults.packet_view_mode

    return EnvProbeParams(
        provider=provider,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        env_contract=env_contract,
        policy_identity=policy_identity,
        timeout_s=_coerce_float(raw.get("timeout_s", defaults.timeout_s), lo=0.5, hi=120.0),
        env_max_tokens=_coerce_int(raw.get("env_max_tokens", defaults.env_max_tokens), lo=64, hi=4000),
        temperature=_coerce_float(raw.get("temperature", defaults.temperature), lo=0.0, hi=2.0),
        top_p=_coerce_float(raw.get("top_p", defaults.top_p), lo=0.0, hi=1.0),
        presence_penalty=_coerce_float(raw.get("presence_penalty", defaults.presence_penalty), lo=-2.0, hi=2.0),
        planner_prompt_override=str(raw.get("planner_prompt_override", defaults.planner_prompt_override)).strip(),
        inter_frame_ms=_coerce_float(raw.get("inter_frame_ms", defaults.inter_frame_ms), lo=1.0, hi=3000.0),
        packet_view_mode=packet_view_mode,
    )


def normalize_chain_params(raw: Dict[str, Any], *, defaults: Any) -> EnvPlannerProbeParams:
    env_params = normalize_params(raw, defaults=defaults)
    return EnvPlannerProbeParams(
        **asdict(env_params),
        planner_prompt=str(raw.get("planner_prompt", defaults.planner_prompt)).strip() or defaults.planner_prompt,
        policy_capabilities=(
            str(raw.get("policy_capabilities", defaults.policy_capabilities)).strip() or defaults.policy_capabilities
        ),
        policy_safety=str(raw.get("policy_safety", defaults.policy_safety)).strip() or defaults.policy_safety,
        policy_style=str(raw.get("policy_style", defaults.policy_style)).strip() or defaults.policy_style,
        planner_max_proposals=_coerce_int(
            raw.get("planner_max_proposals", defaults.planner_max_proposals),
            lo=1,
            hi=5,
        ),
        planner_use_env_context=_coerce_bool(
            raw.get("planner_use_env_context", defaults.planner_use_env_context),
            default=bool(defaults.planner_use_env_context),
        ),
        planner_max_tokens=_coerce_int(raw.get("planner_max_tokens", defaults.planner_max_tokens), lo=64, hi=4000),
        planner_temperature=_coerce_float(
            raw.get("planner_temperature", defaults.planner_temperature),
            lo=0.0,
            hi=2.0,
        ),
        planner_top_p=_coerce_float(raw.get("planner_top_p", defaults.planner_top_p), lo=0.0, hi=1.0),
        planner_presence_penalty=_coerce_float(
            raw.get("planner_presence_penalty", defaults.planner_presence_penalty),
            lo=-2.0,
            hi=2.0,
        ),
        planner_system_prompt=(
            str(raw.get("planner_system_prompt", defaults.planner_system_prompt)).strip() or defaults.planner_system_prompt
        ),
        planner_image_indices=(
            str(raw.get("planner_image_indices", defaults.planner_image_indices)).strip() or defaults.planner_image_indices
        ),
        planner_context_override_json=str(
            raw.get("planner_context_override_json", defaults.planner_context_override_json)
        ).strip(),
        planner_user_text_override=str(
            raw.get("planner_user_text_override", defaults.planner_user_text_override)
        ).strip(),
        planner_payload_override_json=str(
            raw.get("planner_payload_override_json", defaults.planner_payload_override_json)
        ).strip(),
    )


def _build_images_manifest(
    *,
    prepared: List[PreparedImage],
    jpeg_payloads: List[bytes],
    run_dir: Path,
    run_id: str,
    logs_root: Path,
) -> tuple[List[Dict[str, Any]], List[str]]:
    images_manifest: List[Dict[str, Any]] = []
    image_data_urls: List[str] = []
    for idx, (img, payload) in enumerate(zip(prepared, jpeg_payloads), start=1):
        rel_path = f"images/{idx:03d}_packet.jpg"
        storage.write_bytes(run_dir / rel_path, payload)
        image_data_urls.append(img.data_url)
        images_manifest.append(
            {
                "index": idx,
                "filename": img.filename,
                "content_type": img.content_type,
                "original_width": img.original_width,
                "original_height": img.original_height,
                "encoded_width": img.encoded_width,
                "encoded_height": img.encoded_height,
                "jpeg_bytes": img.jpeg_bytes,
                "packet_image_rel_path": rel_path,
                "packet_image_url": _ensure_image_url(logs_root, run_id, rel_path),
            }
        )
    return images_manifest, image_data_urls


def _copy_images_from_existing_run(
    *,
    source_run_dir: Path,
    source_images: List[Dict[str, Any]],
    target_run_dir: Path,
    run_id: str,
    logs_root: Path,
) -> tuple[List[Dict[str, Any]], List[str]]:
    images_manifest: List[Dict[str, Any]] = []
    image_data_urls: List[str] = []
    for idx, item in enumerate(source_images, start=1):
        src_rel = str(item.get("packet_image_rel_path") or f"images/{idx:03d}_packet.jpg")
        src_path = source_run_dir / src_rel
        if not src_path.exists():
            raise ProbeInputError(f"Missing source packet image for planner replay: {src_rel}")
        payload = src_path.read_bytes()
        rel_path = f"images/{idx:03d}_packet.jpg"
        storage.write_bytes(target_run_dir / rel_path, payload)
        data_url = "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")
        image_data_urls.append(data_url)
        images_manifest.append(
            {
                "index": idx,
                "filename": item.get("filename") or f"image_{idx}.jpg",
                "content_type": item.get("content_type") or "image/jpeg",
                "original_width": int(item.get("original_width") or 0),
                "original_height": int(item.get("original_height") or 0),
                "encoded_width": int(item.get("encoded_width") or item.get("original_width") or 0),
                "encoded_height": int(item.get("encoded_height") or item.get("original_height") or 0),
                "jpeg_bytes": int(item.get("jpeg_bytes") or len(payload)),
                "packet_image_rel_path": rel_path,
                "packet_image_url": _ensure_image_url(logs_root, run_id, rel_path),
            }
        )
    return images_manifest, image_data_urls


def _planner_preflight(
    *,
    params: EnvPlannerProbeParams,
    latest_env: Dict[str, Any],
    images_manifest: List[Dict[str, Any]],
    image_data_urls: List[str],
) -> Dict[str, Any]:
    auto_planner_context = _build_planner_context(
        latest_env=latest_env,
        use_env_context=bool(params.planner_use_env_context),
    )
    planner_context = auto_planner_context
    planner_context_override_used = False
    if params.planner_context_override_json:
        candidate_context = _parse_json_object(
            params.planner_context_override_json,
            field_name="planner_context_override_json",
        )
        if candidate_context != auto_planner_context:
            planner_context = candidate_context
            planner_context_override_used = True

    selected_indices = _parse_planner_image_indices(
        params.planner_image_indices,
        image_count=len(images_manifest),
    )
    planner_image_data_urls = [image_data_urls[idx - 1] for idx in selected_indices if 1 <= idx <= len(image_data_urls)]
    planner_image_manifest = []
    for idx in selected_indices:
        if idx < 1 or idx > len(images_manifest):
            continue
        item = images_manifest[idx - 1]
        planner_image_manifest.append(
            {
                "packet_index": item.get("index"),
                "filename": item.get("filename"),
                "original_width": item.get("original_width"),
                "original_height": item.get("original_height"),
                "jpeg_bytes": item.get("jpeg_bytes"),
            }
        )

    chat_url = normalize_chat_url(params.base_url, provider=params.provider)
    if not chat_url:
        raise ProbeInputError("Unable to resolve chat URL from base URL/provider settings.")

    default_planner_user_text = build_planner_user_text(
        context=planner_context,
        policy_identity=params.policy_identity,
        policy_capabilities=params.policy_capabilities,
        policy_safety=params.policy_safety,
        policy_style=params.policy_style,
        planner_prompt=params.planner_prompt,
        max_proposals=max(1, int(params.planner_max_proposals)),
    )
    planner_user_text = default_planner_user_text
    planner_user_text_override_used = False
    if params.planner_user_text_override:
        candidate_user_text = str(params.planner_user_text_override)
        if candidate_user_text.strip() != default_planner_user_text.strip():
            planner_user_text = candidate_user_text
            planner_user_text_override_used = True

    planner_system_prompt = str(params.planner_system_prompt or params.system_prompt).strip() or params.system_prompt
    default_payload = _build_planner_request_payload(
        params=params,
        planner_system_prompt=planner_system_prompt,
        planner_user_text=planner_user_text,
        planner_image_data_urls=planner_image_data_urls,
    )
    payload = default_payload
    payload_override_used = False
    if params.planner_payload_override_json:
        candidate_payload = _parse_json_object(
            params.planner_payload_override_json,
            field_name="planner_payload_override_json",
        )
        if candidate_payload != default_payload:
            payload = candidate_payload
            payload_override_used = True
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []

    response_format = intent_response_format()
    packet_compact = _planner_packet_compact(
        params=params,
        chat_url=chat_url,
        planner_image_count=len(planner_image_manifest),
        planner_image_indices=params.planner_image_indices,
        user_text=planner_user_text,
    )
    packet_expanded = _planner_packet_expanded(
        params=params,
        planner_system_prompt=planner_system_prompt,
        planner_context=planner_context,
        response_format=response_format,
        user_text=planner_user_text,
        planner_image_manifest=planner_image_manifest,
    )
    message_structure = _message_structure(messages)
    request_payload_redacted = _redact_data_urls(payload)
    effective_inputs = {
        "target": {
            "provider": params.provider,
            "model": params.model,
            "base_url": params.base_url,
            "chat_url": chat_url,
            "timeout_s": params.timeout_s,
        },
        "knobs": {
            "max_tokens": params.planner_max_tokens,
            "temperature": params.planner_temperature,
            "top_p": params.planner_top_p,
            "presence_penalty": params.planner_presence_penalty,
            "planner_max_proposals": params.planner_max_proposals,
            "planner_use_env_context": params.planner_use_env_context,
            "planner_image_indices": params.planner_image_indices,
        },
        "prompt_fields": {
            "system_prompt": planner_system_prompt,
            "policy_identity": params.policy_identity,
            "policy_capabilities": params.policy_capabilities,
            "policy_safety": params.policy_safety,
            "policy_style": params.policy_style,
            "planner_prompt": params.planner_prompt,
            "user_text": planner_user_text,
            "planner_user_text_override": params.planner_user_text_override,
            "planner_context_override_json": params.planner_context_override_json,
            "planner_payload_override_json": params.planner_payload_override_json,
        },
        "context": planner_context,
        "images": planner_image_manifest,
        "response_format": response_format,
        "message_structure": message_structure,
        "request_payload_redacted": request_payload_redacted,
        "overrides": {
            "planner_context_override_used": planner_context_override_used,
            "planner_user_text_override_used": planner_user_text_override_used,
            "planner_payload_override_used": payload_override_used,
        },
    }

    return {
        "chat_url": chat_url,
        "payload": payload,
        "messages": messages,
        "packet_compact": packet_compact,
        "packet_expanded": packet_expanded,
        "message_structure": message_structure,
        "request_payload_redacted": request_payload_redacted,
        "effective_inputs": effective_inputs,
    }


def _execute_env_phase(
    *,
    run_id: str,
    api_key: str,
    params: EnvProbeParams,
    images_manifest: List[Dict[str, Any]],
    image_data_urls: List[str],
    frame_timeline: List[Dict[str, float]],
) -> Dict[str, Any]:
    env_context = _build_env_context(frame_timeline)
    user_text = build_env_user_text(
        context=env_context,
        policy_identity=params.policy_identity,
        contract_override=params.env_contract,
    )
    if params.planner_prompt_override:
        user_text += f"\noperator_guidance={params.planner_prompt_override}"

    chat_url = normalize_chat_url(params.base_url, provider=params.provider)
    if not chat_url:
        raise ProbeInputError("Unable to resolve chat URL from base URL/provider settings.")

    payload = _build_env_request_payload(params=params, user_text=user_text, image_data_urls=image_data_urls)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []

    result = post_chat_json(
        url=chat_url,
        payload=payload,
        timeout_s=params.timeout_s,
        api_key=api_key,
        provider=params.provider,
    )

    content, reasoning = extract_message_content(result.response_json or {}) if result.response_json else (None, None)
    finish_reason, ptok, ctok, ttok = (
        _response_meta(result.response_json or {}) if result.response_json else (None, None, None, None)
    )

    parser = EnvSummarizer()
    parser.submit_or_replace({"request_id": run_id})
    parsed = parser.complete_request(content or "")
    parsed_output = asdict(parsed.summary) if parsed is not None else None

    parse_error = _env_error_from_result(result, parser, parsed)
    response_meta = {
        "http_ok": bool(result.ok),
        "http_status": int(result.status_code),
        "latency_ms": round(float(result.latency_ms), 1),
        "error": None if result.ok else result.error,
        "finish_reason": finish_reason,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "total_tokens": ttok,
    }

    response_format = env_response_format()
    packet_compact = _env_packet_compact(
        params=params,
        chat_url=chat_url,
        images=images_manifest,
        frame_timeline=frame_timeline,
        user_text=user_text,
    )
    packet_expanded = _env_packet_expanded(
        params=params,
        frame_timeline=frame_timeline,
        env_context=env_context,
        response_format=response_format,
        user_text=user_text,
        images=images_manifest,
    )
    message_structure = _message_structure(messages)
    request_payload_redacted = _redact_data_urls(payload)

    effective_inputs = {
        "target": {
            "provider": params.provider,
            "model": params.model,
            "base_url": params.base_url,
            "chat_url": chat_url,
            "timeout_s": params.timeout_s,
        },
        "knobs": {
            "max_tokens": params.env_max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "inter_frame_ms": params.inter_frame_ms,
        },
        "prompt_fields": {
            "system_prompt": params.system_prompt,
            "env_contract": params.env_contract,
            "policy_identity": params.policy_identity,
            "planner_prompt_override": params.planner_prompt_override,
            "user_text": user_text,
        },
        "context": env_context,
        "frame_timeline": frame_timeline,
        "images": images_manifest,
        "response_format": response_format,
        "message_structure": message_structure,
        "request_payload_redacted": request_payload_redacted,
    }

    return {
        "chat_url": chat_url,
        "packet_compact": packet_compact,
        "packet_expanded": packet_expanded,
        "message_structure": message_structure,
        "request_payload_redacted": request_payload_redacted,
        "response_meta": response_meta,
        "raw_content": content,
        "reasoning_content": reasoning,
        "parse_ok": parsed is not None and bool(result.ok),
        "parse_stage": parser.last_parse_stage,
        "parse_error": parse_error,
        "parsed_output": parsed_output,
        "effective_inputs": effective_inputs,
    }


def _execute_planner_phase(
    *,
    run_id: str,
    api_key: str,
    params: EnvPlannerProbeParams,
    latest_env: Dict[str, Any],
    images_manifest: List[Dict[str, Any]],
    image_data_urls: List[str],
) -> Dict[str, Any]:
    pre = _planner_preflight(
        params=params,
        latest_env=latest_env,
        images_manifest=images_manifest,
        image_data_urls=image_data_urls,
    )

    result = post_chat_json(
        url=pre["chat_url"],
        payload=pre["payload"],
        timeout_s=params.timeout_s,
        api_key=api_key,
        provider=params.provider,
    )

    content, reasoning = extract_message_content(result.response_json or {}) if result.response_json else (None, None)
    finish_reason, ptok, ctok, ttok = (
        _response_meta(result.response_json or {}) if result.response_json else (None, None, None, None)
    )

    parser = IntentProposer()
    parser.submit_or_replace({"request_id": run_id})
    parsed = parser.complete_request(content or "")
    parsed_output = asdict(parsed.response) if parsed is not None else None

    parse_error = _planner_error_from_result(result, parser, parsed)
    response_meta = {
        "http_ok": bool(result.ok),
        "http_status": int(result.status_code),
        "latency_ms": round(float(result.latency_ms), 1),
        "error": None if result.ok else result.error,
        "finish_reason": finish_reason,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "total_tokens": ttok,
    }

    return {
        "packet_compact": pre["packet_compact"],
        "packet_expanded": pre["packet_expanded"],
        "message_structure": pre["message_structure"],
        "request_payload_redacted": pre["request_payload_redacted"],
        "response_meta": response_meta,
        "raw_content": content,
        "reasoning_content": reasoning,
        "parse_ok": parsed is not None and bool(result.ok),
        "parse_stage": parser.last_parse_stage,
        "parse_error": parse_error,
        "parsed_output": parsed_output,
        "effective_inputs": pre["effective_inputs"],
    }


def _preview_planner_phase(
    *,
    params: EnvPlannerProbeParams,
    latest_env: Dict[str, Any],
    images_manifest: List[Dict[str, Any]],
    image_data_urls: List[str],
) -> Dict[str, Any]:
    pre = _planner_preflight(
        params=params,
        latest_env=latest_env,
        images_manifest=images_manifest,
        image_data_urls=image_data_urls,
    )
    return {
        "executed": False,
        "skip_reason": "awaiting_manual_run",
        "response_meta": {},
        "parse_ok": False,
        "parse_stage": "preview",
        "parse_error": None,
        "parsed_output": None,
        "raw_content": None,
        "reasoning_content": None,
        "packet_compact": pre["packet_compact"],
        "packet_expanded": pre["packet_expanded"],
        "message_structure": pre["message_structure"],
        "request_payload_redacted": pre["request_payload_redacted"],
        "effective_inputs": pre["effective_inputs"],
    }


async def run_env_probe(
    *,
    params: EnvProbeParams,
    upload_files: List[UploadFile],
    image_order: str,
    logs_root: Path,
) -> EnvProbeRun:
    api_key = resolve_api_key()
    if api_key is None:
        raise ProbeInputError(f"Missing API key in environment variable {API_KEY_ENV_VAR}.")

    run_id, run_dir = storage.new_run_dir(logs_root)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prepared, jpeg_payloads = await prepare_images(upload_files, image_order=image_order)
    images_manifest, image_data_urls = _build_images_manifest(
        prepared=prepared,
        jpeg_payloads=jpeg_payloads,
        run_dir=run_dir,
        run_id=run_id,
        logs_root=logs_root,
    )

    frame_timeline = _build_frame_timeline(len(images_manifest), inter_frame_ms=params.inter_frame_ms)
    env = _execute_env_phase(
        run_id=run_id,
        api_key=api_key,
        params=params,
        images_manifest=images_manifest,
        image_data_urls=image_data_urls,
        frame_timeline=frame_timeline,
    )

    run = EnvProbeRun(
        run_id=run_id,
        created_at_utc=created_at,
        mode="env",
        chain_status="env_only",
        params={
            "mode": "env",
            "provider": params.provider,
            "model": params.model,
            "base_url": params.base_url,
            "chat_url": env["chat_url"],
            "system_prompt": params.system_prompt,
            "env_contract": params.env_contract,
            "policy_identity": params.policy_identity,
            "timeout_s": params.timeout_s,
            "env_max_tokens": params.env_max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "planner_prompt_override": params.planner_prompt_override,
            "inter_frame_ms": params.inter_frame_ms,
            "packet_view_mode": params.packet_view_mode,
        },
        images=images_manifest,
        packet_compact=env["packet_compact"],
        packet_expanded=env["packet_expanded"],
        message_structure=env["message_structure"],
        request_payload_redacted=env["request_payload_redacted"],
        response_meta=env["response_meta"],
        raw_content=env["raw_content"],
        reasoning_content=env["reasoning_content"],
        parse_ok=env["parse_ok"],
        parse_stage=env["parse_stage"],
        parse_error=env["parse_error"],
        parsed_output=env["parsed_output"],
        effective_inputs={"env": env["effective_inputs"]},
    )

    summary = storage.save_run(run_dir, run)
    storage.update_recent_index(logs_root, summary)
    return run


async def run_env_planner_probe(
    *,
    params: EnvPlannerProbeParams,
    upload_files: List[UploadFile],
    image_order: str,
    logs_root: Path,
) -> EnvProbeRun:
    api_key = resolve_api_key()
    if api_key is None:
        raise ProbeInputError(f"Missing API key in environment variable {API_KEY_ENV_VAR}.")

    run_id, run_dir = storage.new_run_dir(logs_root)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prepared, jpeg_payloads = await prepare_images(upload_files, image_order=image_order)
    images_manifest, image_data_urls = _build_images_manifest(
        prepared=prepared,
        jpeg_payloads=jpeg_payloads,
        run_dir=run_dir,
        run_id=run_id,
        logs_root=logs_root,
    )

    frame_timeline = _build_frame_timeline(len(images_manifest), inter_frame_ms=params.inter_frame_ms)
    env = _execute_env_phase(
        run_id=run_id,
        api_key=api_key,
        params=params,
        images_manifest=images_manifest,
        image_data_urls=image_data_urls,
        frame_timeline=frame_timeline,
    )

    planner_phase = None
    planner_effective_inputs = None
    planner_skipped_reason = None
    chain_status = "ok"

    env_http_ok = bool((env.get("response_meta") or {}).get("http_ok"))
    env_parse_ok = bool(env.get("parse_ok"))

    if not env_http_ok:
        chain_status = "env_transport_fail"
        planner_skipped_reason = "env_transport_fail"
    elif not env_parse_ok:
        chain_status = "partial_env_parse_fail"
        planner_skipped_reason = "env_parse_fail"
    else:
        planner = _execute_planner_phase(
            run_id=run_id,
            api_key=api_key,
            params=params,
            latest_env=env.get("parsed_output") or {},
            images_manifest=images_manifest,
            image_data_urls=image_data_urls,
        )
        planner_phase = {
            "executed": True,
            "packet_compact": planner["packet_compact"],
            "packet_expanded": planner["packet_expanded"],
            "message_structure": planner["message_structure"],
            "request_payload_redacted": planner["request_payload_redacted"],
            "response_meta": planner["response_meta"],
            "raw_content": planner["raw_content"],
            "reasoning_content": planner["reasoning_content"],
            "parse_ok": planner["parse_ok"],
            "parse_stage": planner["parse_stage"],
            "parse_error": planner["parse_error"],
            "parsed_output": planner["parsed_output"],
        }
        planner_effective_inputs = planner["effective_inputs"]

        planner_http_ok = bool((planner.get("response_meta") or {}).get("http_ok"))
        planner_parse_ok = bool(planner.get("parse_ok"))
        if not planner_http_ok:
            chain_status = "planner_transport_fail"
        elif not planner_parse_ok:
            chain_status = "planner_parse_fail"
        else:
            chain_status = "ok"

    if planner_phase is None:
        planner_phase = {
            "executed": False,
            "skip_reason": planner_skipped_reason,
            "response_meta": {},
            "parse_ok": False,
            "parse_stage": "skipped",
            "parse_error": planner_skipped_reason,
            "parsed_output": None,
            "raw_content": None,
            "reasoning_content": None,
            "packet_compact": [],
            "packet_expanded": [],
            "message_structure": [],
            "request_payload_redacted": {},
        }

    effective_inputs = {"env": env["effective_inputs"]}
    if planner_effective_inputs is not None:
        effective_inputs["planner"] = planner_effective_inputs
    else:
        effective_inputs["planner"] = {
            "skipped": True,
            "reason": planner_skipped_reason,
        }

    run = EnvProbeRun(
        run_id=run_id,
        created_at_utc=created_at,
        mode="env_planner",
        chain_status=chain_status,
        planner_skipped_reason=planner_skipped_reason,
        params={
            "mode": "env_planner",
            "provider": params.provider,
            "model": params.model,
            "base_url": params.base_url,
            "chat_url": env["chat_url"],
            "system_prompt": params.system_prompt,
            "env_contract": params.env_contract,
            "policy_identity": params.policy_identity,
            "timeout_s": params.timeout_s,
            "env_max_tokens": params.env_max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "planner_prompt_override": params.planner_prompt_override,
            "inter_frame_ms": params.inter_frame_ms,
            "packet_view_mode": params.packet_view_mode,
            "planner_prompt": params.planner_prompt,
            "policy_capabilities": params.policy_capabilities,
            "policy_safety": params.policy_safety,
            "policy_style": params.policy_style,
            "planner_max_proposals": params.planner_max_proposals,
            "planner_use_env_context": params.planner_use_env_context,
            "planner_max_tokens": params.planner_max_tokens,
            "planner_temperature": params.planner_temperature,
            "planner_top_p": params.planner_top_p,
            "planner_presence_penalty": params.planner_presence_penalty,
            "planner_system_prompt": params.planner_system_prompt,
            "planner_image_indices": params.planner_image_indices,
            "planner_context_override_json": params.planner_context_override_json,
            "planner_user_text_override": params.planner_user_text_override,
            "planner_payload_override_json": params.planner_payload_override_json,
        },
        images=images_manifest,
        packet_compact=env["packet_compact"],
        packet_expanded=env["packet_expanded"],
        message_structure=env["message_structure"],
        request_payload_redacted=env["request_payload_redacted"],
        response_meta=env["response_meta"],
        raw_content=env["raw_content"],
        reasoning_content=env["reasoning_content"],
        parse_ok=env["parse_ok"],
        parse_stage=env["parse_stage"],
        parse_error=env["parse_error"],
        parsed_output=env["parsed_output"],
        effective_inputs=effective_inputs,
        planner_phase=planner_phase,
    )

    summary = storage.save_run(run_dir, run)
    storage.update_recent_index(logs_root, summary)
    return run


async def run_env_then_preview_planner(
    *,
    params: EnvPlannerProbeParams,
    upload_files: List[UploadFile],
    image_order: str,
    logs_root: Path,
) -> EnvProbeRun:
    api_key = resolve_api_key()
    if api_key is None:
        raise ProbeInputError(f"Missing API key in environment variable {API_KEY_ENV_VAR}.")

    run_id, run_dir = storage.new_run_dir(logs_root)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prepared, jpeg_payloads = await prepare_images(upload_files, image_order=image_order)
    images_manifest, image_data_urls = _build_images_manifest(
        prepared=prepared,
        jpeg_payloads=jpeg_payloads,
        run_dir=run_dir,
        run_id=run_id,
        logs_root=logs_root,
    )

    frame_timeline = _build_frame_timeline(len(images_manifest), inter_frame_ms=params.inter_frame_ms)
    env = _execute_env_phase(
        run_id=run_id,
        api_key=api_key,
        params=params,
        images_manifest=images_manifest,
        image_data_urls=image_data_urls,
        frame_timeline=frame_timeline,
    )

    planner_phase = None
    planner_effective_inputs = None
    planner_skipped_reason = None
    chain_status = "planner_pending"

    env_http_ok = bool((env.get("response_meta") or {}).get("http_ok"))
    env_parse_ok = bool(env.get("parse_ok"))
    if not env_http_ok:
        chain_status = "env_transport_fail"
        planner_skipped_reason = "env_transport_fail"
    elif not env_parse_ok:
        chain_status = "partial_env_parse_fail"
        planner_skipped_reason = "env_parse_fail"
    else:
        planner_phase = _preview_planner_phase(
            params=params,
            latest_env=env.get("parsed_output") or {},
            images_manifest=images_manifest,
            image_data_urls=image_data_urls,
        )
        planner_effective_inputs = planner_phase.get("effective_inputs")

    if planner_phase is None:
        planner_phase = {
            "executed": False,
            "skip_reason": planner_skipped_reason,
            "response_meta": {},
            "parse_ok": False,
            "parse_stage": "skipped",
            "parse_error": planner_skipped_reason,
            "parsed_output": None,
            "raw_content": None,
            "reasoning_content": None,
            "packet_compact": [],
            "packet_expanded": [],
            "message_structure": [],
            "request_payload_redacted": {},
        }

    effective_inputs = {"env": env["effective_inputs"]}
    if planner_effective_inputs is not None:
        effective_inputs["planner"] = planner_effective_inputs
    else:
        effective_inputs["planner"] = {"skipped": True, "reason": planner_skipped_reason}

    run = EnvProbeRun(
        run_id=run_id,
        created_at_utc=created_at,
        mode="env_planner_preview",
        chain_status=chain_status,
        planner_skipped_reason=planner_skipped_reason,
        params={
            "mode": "env_planner_preview",
            "provider": params.provider,
            "model": params.model,
            "base_url": params.base_url,
            "chat_url": env["chat_url"],
            "system_prompt": params.system_prompt,
            "env_contract": params.env_contract,
            "policy_identity": params.policy_identity,
            "timeout_s": params.timeout_s,
            "env_max_tokens": params.env_max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "planner_prompt_override": params.planner_prompt_override,
            "inter_frame_ms": params.inter_frame_ms,
            "packet_view_mode": params.packet_view_mode,
            "planner_prompt": params.planner_prompt,
            "policy_capabilities": params.policy_capabilities,
            "policy_safety": params.policy_safety,
            "policy_style": params.policy_style,
            "planner_max_proposals": params.planner_max_proposals,
            "planner_use_env_context": params.planner_use_env_context,
            "planner_max_tokens": params.planner_max_tokens,
            "planner_temperature": params.planner_temperature,
            "planner_top_p": params.planner_top_p,
            "planner_presence_penalty": params.planner_presence_penalty,
            "planner_system_prompt": params.planner_system_prompt,
            "planner_image_indices": params.planner_image_indices,
            "planner_context_override_json": params.planner_context_override_json,
            "planner_user_text_override": params.planner_user_text_override,
            "planner_payload_override_json": params.planner_payload_override_json,
        },
        images=images_manifest,
        packet_compact=env["packet_compact"],
        packet_expanded=env["packet_expanded"],
        message_structure=env["message_structure"],
        request_payload_redacted=env["request_payload_redacted"],
        response_meta=env["response_meta"],
        raw_content=env["raw_content"],
        reasoning_content=env["reasoning_content"],
        parse_ok=env["parse_ok"],
        parse_stage=env["parse_stage"],
        parse_error=env["parse_error"],
        parsed_output=env["parsed_output"],
        effective_inputs=effective_inputs,
        planner_phase=planner_phase,
    )

    summary = storage.save_run(run_dir, run)
    storage.update_recent_index(logs_root, summary)
    return run


async def run_planner_from_prepared_env(
    *,
    params: EnvPlannerProbeParams,
    source_env_run_id: str,
    logs_root: Path,
) -> EnvProbeRun:
    api_key = resolve_api_key()
    if api_key is None:
        raise ProbeInputError(f"Missing API key in environment variable {API_KEY_ENV_VAR}.")

    loaded = storage.load_run(logs_root, source_env_run_id)
    if loaded is None:
        raise ProbeInputError("Prepared env run was not found.")
    source_full = loaded.get("run_full") or {}
    if not isinstance(source_full, dict) or not source_full.get("run_id"):
        raise ProbeInputError("Prepared env run did not include full run metadata.")

    source_http_ok = bool((source_full.get("response_meta") or {}).get("http_ok"))
    source_parse_ok = bool(source_full.get("parse_ok"))
    if not source_http_ok:
        raise ProbeInputError("Prepared env run did not complete a successful env request.")
    if not source_parse_ok:
        raise ProbeInputError("Prepared env run did not produce a parseable env output.")

    source_images = source_full.get("images")
    if not isinstance(source_images, list) or not source_images:
        raise ProbeInputError("Prepared env run did not include packet images.")
    source_run_dir = Path(str(loaded.get("run_dir", "")))
    if not source_run_dir.exists():
        raise ProbeInputError("Prepared env run directory is missing.")

    run_id, run_dir = storage.new_run_dir(logs_root)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    images_manifest, image_data_urls = _copy_images_from_existing_run(
        source_run_dir=source_run_dir,
        source_images=source_images,
        target_run_dir=run_dir,
        run_id=run_id,
        logs_root=logs_root,
    )

    planner = _execute_planner_phase(
        run_id=run_id,
        api_key=api_key,
        params=params,
        latest_env=source_full.get("parsed_output") or {},
        images_manifest=images_manifest,
        image_data_urls=image_data_urls,
    )

    planner_phase = {
        "executed": True,
        "packet_compact": planner["packet_compact"],
        "packet_expanded": planner["packet_expanded"],
        "message_structure": planner["message_structure"],
        "request_payload_redacted": planner["request_payload_redacted"],
        "response_meta": planner["response_meta"],
        "raw_content": planner["raw_content"],
        "reasoning_content": planner["reasoning_content"],
        "parse_ok": planner["parse_ok"],
        "parse_stage": planner["parse_stage"],
        "parse_error": planner["parse_error"],
        "parsed_output": planner["parsed_output"],
    }

    planner_http_ok = bool((planner.get("response_meta") or {}).get("http_ok"))
    planner_parse_ok = bool(planner.get("parse_ok"))
    if not planner_http_ok:
        chain_status = "planner_transport_fail"
    elif not planner_parse_ok:
        chain_status = "planner_parse_fail"
    else:
        chain_status = "ok"

    source_effective_env = (source_full.get("effective_inputs") or {}).get("env")
    if not isinstance(source_effective_env, dict):
        source_effective_env = {"source_env_run_id": source_env_run_id}
    effective_inputs = {
        "env": source_effective_env,
        "planner": planner["effective_inputs"],
    }

    run = EnvProbeRun(
        run_id=run_id,
        created_at_utc=created_at,
        mode="env_planner_from_prepared",
        chain_status=chain_status,
        planner_skipped_reason=None,
        params={
            "mode": "env_planner_from_prepared",
            "source_env_run_id": source_env_run_id,
            "provider": params.provider,
            "model": params.model,
            "base_url": params.base_url,
            "system_prompt": params.system_prompt,
            "env_contract": params.env_contract,
            "policy_identity": params.policy_identity,
            "timeout_s": params.timeout_s,
            "env_max_tokens": params.env_max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "planner_prompt_override": params.planner_prompt_override,
            "inter_frame_ms": params.inter_frame_ms,
            "packet_view_mode": params.packet_view_mode,
            "planner_prompt": params.planner_prompt,
            "policy_capabilities": params.policy_capabilities,
            "policy_safety": params.policy_safety,
            "policy_style": params.policy_style,
            "planner_max_proposals": params.planner_max_proposals,
            "planner_use_env_context": params.planner_use_env_context,
            "planner_max_tokens": params.planner_max_tokens,
            "planner_temperature": params.planner_temperature,
            "planner_top_p": params.planner_top_p,
            "planner_presence_penalty": params.planner_presence_penalty,
            "planner_system_prompt": params.planner_system_prompt,
            "planner_image_indices": params.planner_image_indices,
            "planner_context_override_json": params.planner_context_override_json,
            "planner_user_text_override": params.planner_user_text_override,
            "planner_payload_override_json": params.planner_payload_override_json,
        },
        images=images_manifest,
        packet_compact=source_full.get("packet_compact") or [],
        packet_expanded=source_full.get("packet_expanded") or [],
        message_structure=source_full.get("message_structure") or [],
        request_payload_redacted=source_full.get("request_payload_redacted") or {},
        response_meta=source_full.get("response_meta") or {},
        raw_content=source_full.get("raw_content"),
        reasoning_content=source_full.get("reasoning_content"),
        parse_ok=bool(source_full.get("parse_ok")),
        parse_stage=str(source_full.get("parse_stage") or "unknown"),
        parse_error=source_full.get("parse_error"),
        parsed_output=source_full.get("parsed_output"),
        effective_inputs=effective_inputs,
        planner_phase=planner_phase,
    )

    summary = storage.save_run(run_dir, run)
    storage.update_recent_index(logs_root, summary)
    return run


def load_run_for_ui(logs_root: Path, run_id: str) -> Dict[str, Any] | None:
    loaded = storage.load_run(logs_root, run_id)
    if loaded is None:
        return None

    full = loaded.get("run_full")
    if isinstance(full, dict) and full.get("run_id"):
        return full

    packet_view = loaded.get("packet_view") or {}
    parsed = loaded.get("parsed") or {}
    response_raw = loaded.get("response_raw") or {}
    return {
        "run_id": loaded.get("run_id"),
        "created_at_utc": (loaded.get("summary") or {}).get("created_at_utc"),
        "mode": (loaded.get("summary") or {}).get("mode", "env"),
        "chain_status": (loaded.get("summary") or {}).get("chain_status", "env_only"),
        "run_config": loaded.get("run_config", {}),
        "params": loaded.get("run_config", {}),
        "inputs_manifest": loaded.get("inputs_manifest", []),
        "images": loaded.get("inputs_manifest", []),
        "packet_compact": packet_view.get("compact", []),
        "packet_expanded": packet_view.get("expanded", []),
        "message_structure": packet_view.get("message_structure", []),
        "response_meta": response_raw.get("response_meta", {}),
        "raw_content": response_raw.get("raw_content"),
        "reasoning_content": response_raw.get("reasoning_content"),
        "parse_ok": bool(parsed.get("parse_ok", False)),
        "parse_stage": parsed.get("parse_stage", "unknown"),
        "parse_error": parsed.get("parse_error"),
        "parsed_output": parsed.get("parsed_output"),
        "summary": loaded.get("summary", {}),
        "effective_inputs": {"env": (loaded.get("run_config") or {})},
        "planner_phase": loaded.get("planner_parsed")
        or {
            "executed": False,
            "parse_ok": False,
            "parse_stage": "unknown",
            "parse_error": None,
            "parsed_output": None,
            "response_meta": {},
            "raw_content": None,
            "reasoning_content": None,
            "packet_compact": [],
            "packet_expanded": [],
            "message_structure": [],
            "request_payload_redacted": {},
        },
    }
