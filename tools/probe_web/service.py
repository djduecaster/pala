from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import base64
import io
import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import UploadFile
from PIL import Image

from pala.behavior.action_guard import ActionGuard, ActionGuardConfig, GuardContext
from pala.behavior.decision_schema_v4 import (
    BehaviorDecisionParseResult,
    BehaviorDecisionParser,
    behavior_decision_response_format,
)
from pala.behavior.mode_fsm_v4 import MacroMode, ModeFsmV4, ModeFsmV4Config, ModeSignalsV4
from pala.behavior.model_clients import extract_message_content, normalize_chat_url, post_chat_json
from pala.behavior.prompts import SYSTEM_PROMPT, build_behavior_v4_user_text, build_messages
from pala.behavior.skills_v4 import (
    allowed_primitives_for,
    allowed_skills_for_mode,
    default_action_payload_for_mode,
    default_skill_for_mode,
)
from pala.types import ActionPlan, action_plan_from_dict, to_json_dict

from . import storage
from .defaults import API_KEY_ENV_VAR, resolve_api_key
from .models import BehaviorProbeParams, BehaviorProbeRun, PreparedImage, ProbeDefaults


_DATA_IMAGE_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


class ProbeInputError(ValueError):
    pass


def _safe_filename(name: str, *, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]", "_", str(name or "")).strip("._")
    if not token:
        return fallback
    return token[:120]


def _json_preview(value: Any, *, max_chars: int = 280) -> str:
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
    max_width: int,
    jpeg_quality: int,
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
                original_width, original_height = rgb.size
                if original_width > max_width:
                    target_h = max(1, int(round((float(original_height) * float(max_width)) / float(original_width))))
                    rgb = rgb.resize((int(max_width), target_h), resample=Image.Resampling.LANCZOS)
                encoded_width, encoded_height = rgb.size
                out = io.BytesIO()
                rgb.save(
                    out,
                    format="JPEG",
                    quality=max(30, min(95, int(jpeg_quality))),
                    optimize=True,
                )
        except Exception as exc:  # noqa: BLE001
            raise ProbeInputError(f"Failed to decode image '{item.filename or idx}': {exc}") from exc

        payload = out.getvalue()
        jpeg_payloads.append(payload)
        data_url = "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")

        prepared.append(
            PreparedImage(
                filename=_safe_filename(item.filename or "", fallback=f"image_{idx}.jpg"),
                content_type=str(item.content_type or "application/octet-stream"),
                original_width=int(original_width),
                original_height=int(original_height),
                encoded_width=int(encoded_width),
                encoded_height=int(encoded_height),
                jpeg_bytes=len(payload),
                data_url=data_url,
            )
        )

    return prepared, jpeg_payloads


def _ensure_image_url(logs_root: Path, run_id: str, rel_path: str) -> str:
    _ = logs_root
    safe_rel = str(rel_path).replace("\\", "/")
    return f"/probe-logs/{run_id}/{safe_rel}"


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


def normalize_params(raw: Dict[str, Any], *, defaults: ProbeDefaults) -> BehaviorProbeParams:
    provider = str(raw.get("provider", defaults.provider)).strip().lower() or defaults.provider
    model = str(raw.get("model", defaults.model)).strip() or defaults.model
    base_url = str(raw.get("base_url", defaults.base_url)).strip() or defaults.base_url
    system_prompt = str(raw.get("system_prompt", defaults.system_prompt)).strip() or defaults.system_prompt

    packet_view_mode = str(raw.get("packet_view_mode", defaults.packet_view_mode)).strip().lower()
    if packet_view_mode not in {"compact", "expanded"}:
        packet_view_mode = defaults.packet_view_mode

    return BehaviorProbeParams(
        provider=provider,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        timeout_s=_coerce_float(raw.get("timeout_s", defaults.timeout_s), lo=0.5, hi=120.0),
        max_tokens=_coerce_int(raw.get("max_tokens", defaults.max_tokens), lo=64, hi=4000),
        temperature=_coerce_float(raw.get("temperature", defaults.temperature), lo=0.0, hi=2.0),
        top_p=_coerce_float(raw.get("top_p", defaults.top_p), lo=0.0, hi=1.0),
        presence_penalty=_coerce_float(raw.get("presence_penalty", defaults.presence_penalty), lo=-2.0, hi=2.0),
        frame_max_width=_coerce_int(raw.get("frame_max_width", defaults.frame_max_width), lo=64, hi=2048),
        frame_jpeg_quality=_coerce_int(raw.get("frame_jpeg_quality", defaults.frame_jpeg_quality), lo=30, hi=95),
        policy_identity=str(raw.get("policy_identity", defaults.policy_identity)).strip() or defaults.policy_identity,
        policy_capabilities=(
            str(raw.get("policy_capabilities", defaults.policy_capabilities)).strip() or defaults.policy_capabilities
        ),
        policy_safety=str(raw.get("policy_safety", defaults.policy_safety)).strip() or defaults.policy_safety,
        policy_style=str(raw.get("policy_style", defaults.policy_style)).strip() or defaults.policy_style,
        planner_prompt=str(raw.get("planner_prompt", defaults.planner_prompt)).strip() or defaults.planner_prompt,
        context_override_json=str(raw.get("context_override_json", defaults.context_override_json)).strip(),
        user_text_override=str(raw.get("user_text_override", defaults.user_text_override)).strip(),
        payload_override_json=str(raw.get("payload_override_json", defaults.payload_override_json)).strip(),
        inter_frame_ms=_coerce_float(raw.get("inter_frame_ms", defaults.inter_frame_ms), lo=1.0, hi=3000.0),
        packet_view_mode=packet_view_mode,
    )


def _action_from_payload(payload: Dict[str, Any], *, reason: str) -> ActionPlan:
    token = dict(payload)
    token.setdefault("cancel_current", False)
    token.setdefault("explanation", reason)
    action = action_plan_from_dict(token)
    if action is None:
        fallback = action_plan_from_dict(
            {
                "primitive": "hold",
                "command": {},
                "style": "calm",
                "confidence": 0.1,
                "cancel_current": False,
                "explanation": f"fallback:{reason}",
            }
        )
        assert fallback is not None
        return fallback
    action.cancel_current = False
    return action


def _action_to_dict(action: ActionPlan) -> Dict[str, Any]:
    data = to_json_dict(action)
    if isinstance(data, dict):
        return data
    return {
        "primitive": action.primitive.value,
        "style": action.style,
        "confidence": float(action.confidence),
    }


def _mode_from_transition(value: str) -> MacroMode | None:
    token = str(value or "").strip().lower()
    if token == "stay":
        return None
    if not token.startswith("to_"):
        return None
    mode_token = token[3:]
    if mode_token == MacroMode.BOOT_AWAKEN.value:
        return None
    try:
        return MacroMode(mode_token)
    except ValueError:
        return None


class ProbeFsmSimulator:
    def __init__(self, *, defaults: ProbeDefaults) -> None:
        self._lock = threading.Lock()
        self._mode_cfg = ModeFsmV4Config(
            min_mode_dwell_s=max(0.0, float(defaults.min_mode_dwell_s)),
            engage_person_conf=max(0.0, min(1.0, float(defaults.engage_person_conf))),
            disengage_person_conf=max(0.0, min(1.0, float(defaults.disengage_person_conf))),
            boot_timeout_s=6.0,
            recover_settle_s=1.0,
        )
        self._guard_cfg = ActionGuardConfig(
            min_action_dwell_s=max(0.0, float(defaults.min_action_dwell_s)),
            stale_after_s=max(0.2, float(defaults.stale_after_s)),
        )
        self._fsm = ModeFsmV4(config=self._mode_cfg)
        self._guard = ActionGuard(config=self._guard_cfg)
        self._now_mono_s = 0.0
        self._active_skill = default_skill_for_mode(MacroMode.BOOT_AWAKEN)
        self._current_action = _action_from_payload(
            default_action_payload_for_mode(MacroMode.BOOT_AWAKEN, reason="probe_init"),
            reason="probe_init",
        )
        self._last_commit_s = 0.0
        self._last_signals = ModeSignalsV4()
        self._last_transition: Dict[str, Any] = {
            "previous_mode": MacroMode.BOOT_AWAKEN.value,
            "next_mode": MacroMode.BOOT_AWAKEN.value,
            "reason": "startup",
            "transitioned": False,
            "dwell_s": 0.0,
        }
        self._last_guard_result: Dict[str, Any] | None = None
        self.reset(now_mono_s=0.0)

    def reset(self, *, now_mono_s: float = 0.0) -> Dict[str, Any]:
        with self._lock:
            now = max(0.0, float(now_mono_s))
            self._fsm = ModeFsmV4(config=self._mode_cfg)
            self._fsm.reset(now_mono_s=now)
            self._guard = ActionGuard(config=self._guard_cfg)
            self._now_mono_s = now
            self._active_skill = default_skill_for_mode(MacroMode.BOOT_AWAKEN)
            self._current_action = _action_from_payload(
                default_action_payload_for_mode(MacroMode.BOOT_AWAKEN, reason="probe_reset"),
                reason="probe_reset",
            )
            self._last_commit_s = now
            self._last_signals = ModeSignalsV4()
            self._last_transition = {
                "previous_mode": MacroMode.BOOT_AWAKEN.value,
                "next_mode": MacroMode.BOOT_AWAKEN.value,
                "reason": "reset",
                "transitioned": False,
                "dwell_s": 0.0,
            }
            self._last_guard_result = None
            return self._snapshot_locked()

    def step(
        self,
        *,
        signals: ModeSignalsV4,
        advance_s: float,
    ) -> Dict[str, Any]:
        with self._lock:
            self._now_mono_s += max(0.0, float(advance_s))
            transition = self._fsm.update(now_mono_s=self._now_mono_s, signals=signals)
            self._last_signals = signals
            self._last_transition = {
                "previous_mode": transition.previous_mode.value,
                "next_mode": transition.next_mode.value,
                "reason": transition.reason,
                "transitioned": bool(transition.transitioned),
                "dwell_s": round(float(transition.dwell_s), 3),
            }
            if transition.transitioned:
                self._active_skill = default_skill_for_mode(transition.next_mode)
                fallback = self._guard.fallback_action(mode=transition.next_mode, reason=f"mode_enter:{transition.reason}")
                self._commit_locked(action=fallback)
            return self._snapshot_locked()

    def force_mode(self, *, next_mode: MacroMode, reason: str, advance_s: float = 0.0) -> Dict[str, Any]:
        with self._lock:
            self._now_mono_s += max(0.0, float(advance_s))
            transition = self._fsm.force_mode(now_mono_s=self._now_mono_s, next_mode=next_mode, reason=reason)
            self._last_transition = {
                "previous_mode": transition.previous_mode.value,
                "next_mode": transition.next_mode.value,
                "reason": transition.reason,
                "transitioned": bool(transition.transitioned),
                "dwell_s": round(float(transition.dwell_s), 3),
            }
            if transition.transitioned:
                self._active_skill = default_skill_for_mode(transition.next_mode)
                fallback = self._guard.fallback_action(mode=transition.next_mode, reason=f"force_mode:{reason}")
                self._commit_locked(action=fallback)
            return self._snapshot_locked()

    def build_context(self) -> Dict[str, Any]:
        with self._lock:
            return self._context_locked()

    def fsm_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def apply_model_outcome(
        self,
        *,
        parsed: BehaviorDecisionParseResult | None,
        parse_error: str | None,
        model_age_s: float,
    ) -> Dict[str, Any]:
        with self._lock:
            mode_before = self._fsm.snapshot.mode
            transition_applied: Dict[str, Any] | None = None
            if parsed is None:
                reason = f"parse_error:{parse_error or 'unknown'}"
                fallback = self._guard.fallback_action(mode=mode_before, reason=reason)
                self._commit_locked(action=fallback)
                guard_result = {
                    "accepted": False,
                    "used_fallback": True,
                    "reason": reason,
                    "skill": self._active_skill,
                    "action": _action_to_dict(fallback),
                }
                self._last_guard_result = guard_result
                return {
                    "guard_result": guard_result,
                    "mode_transition_applied": transition_applied,
                    "fsm_after": self._snapshot_locked(),
                }

            target_mode = _mode_from_transition(parsed.decision.mode_transition)
            if target_mode is not None and target_mode != self._fsm.snapshot.mode:
                dwell = max(0.0, self._now_mono_s - self._fsm.snapshot.entered_mono_s)
                if target_mode == MacroMode.RECOVER_RESET or dwell >= max(0.0, float(self._mode_cfg.min_mode_dwell_s)):
                    transition = self._fsm.force_mode(
                        now_mono_s=self._now_mono_s,
                        next_mode=target_mode,
                        reason=f"model_transition:{target_mode.value}",
                    )
                    self._active_skill = default_skill_for_mode(target_mode)
                    transition_applied = {
                        "applied": True,
                        "previous_mode": transition.previous_mode.value,
                        "next_mode": transition.next_mode.value,
                        "reason": transition.reason,
                    }
                else:
                    transition_applied = {
                        "applied": False,
                        "reason": "mode_transition_min_dwell_hold",
                        "requested": target_mode.value,
                    }

            action_age_s = max(0.0, self._now_mono_s - self._last_commit_s)
            context = GuardContext(
                mode=self._fsm.snapshot.mode,
                active_skill=self._active_skill,
                current_action=self._current_action,
                action_age_s=action_age_s,
                model_age_s=max(0.0, float(model_age_s)),
                health_degraded=bool(self._last_signals.health_degraded),
                breaker_open=False,
            )
            guard = self._guard.evaluate(decision=parsed.decision, context=context, now_mono_s=self._now_mono_s)
            self._active_skill = guard.skill
            self._commit_locked(action=guard.action)
            if guard.accepted:
                self._guard.mark_committed(action=guard.action, now_mono_s=self._now_mono_s)
            guard_result = {
                "accepted": bool(guard.accepted),
                "used_fallback": bool(guard.used_fallback),
                "reason": guard.reason,
                "skill": guard.skill,
                "action": _action_to_dict(guard.action),
            }
            self._last_guard_result = guard_result
            return {
                "guard_result": guard_result,
                "mode_transition_applied": transition_applied,
                "fsm_after": self._snapshot_locked(),
            }

    def _commit_locked(self, *, action: ActionPlan) -> None:
        self._current_action = action
        self._last_commit_s = self._now_mono_s

    def _context_locked(self) -> Dict[str, Any]:
        mode = self._fsm.snapshot.mode
        allowed_skills = sorted(allowed_skills_for_mode(mode))
        allowed_primitives = sorted(allowed_primitives_for(mode, self._active_skill))
        action_summary = _action_to_dict(self._current_action)
        return {
            "mode": mode.value,
            "active_skill": self._active_skill,
            "mode_reason": self._fsm.snapshot.reason,
            "mode_contract": {
                "allowed_skills": allowed_skills,
                "allowed_primitives_for_active_skill": allowed_primitives,
            },
            "current_action": {
                "primitive": action_summary.get("primitive"),
                "style": action_summary.get("style"),
                "confidence": action_summary.get("confidence"),
            },
            "timing": {
                "action_age_s": round(max(0.0, self._now_mono_s - self._last_commit_s), 3),
                "now_mono_s": round(self._now_mono_s, 3),
            },
            "signals": {
                "person_present": bool(self._last_signals.person_present),
                "person_conf": round(float(self._last_signals.person_conf), 3),
                "search_requested": bool(self._last_signals.search_requested),
                "search_complete": bool(self._last_signals.search_complete),
                "task_active": bool(self._last_signals.task_active),
                "startup_complete": bool(self._last_signals.startup_complete),
                "health_degraded": bool(self._last_signals.health_degraded),
            },
        }

    def _snapshot_locked(self) -> Dict[str, Any]:
        snap = self._fsm.snapshot
        return {
            "now_mono_s": round(self._now_mono_s, 3),
            "mode": snap.mode.value,
            "mode_entered_mono_s": round(float(snap.entered_mono_s), 3),
            "mode_reason": snap.reason,
            "active_skill": self._active_skill,
            "current_action": _action_to_dict(self._current_action),
            "current_action_age_s": round(max(0.0, self._now_mono_s - self._last_commit_s), 3),
            "last_signals": {
                "person_present": bool(self._last_signals.person_present),
                "person_conf": round(float(self._last_signals.person_conf), 3),
                "search_requested": bool(self._last_signals.search_requested),
                "search_complete": bool(self._last_signals.search_complete),
                "task_active": bool(self._last_signals.task_active),
                "startup_complete": bool(self._last_signals.startup_complete),
                "health_degraded": bool(self._last_signals.health_degraded),
            },
            "last_transition": dict(self._last_transition),
            "last_guard_result": dict(self._last_guard_result or {}),
        }


def parse_signals_form(raw: Dict[str, Any]) -> ModeSignalsV4:
    return ModeSignalsV4(
        person_present=_coerce_bool(raw.get("person_present"), default=False),
        person_conf=_coerce_float(raw.get("person_conf", 0.0), lo=0.0, hi=1.0),
        search_requested=_coerce_bool(raw.get("search_requested"), default=False),
        search_complete=_coerce_bool(raw.get("search_complete"), default=False),
        task_active=_coerce_bool(raw.get("task_active"), default=False),
        startup_complete=_coerce_bool(raw.get("startup_complete"), default=False),
        health_degraded=_coerce_bool(raw.get("health_degraded"), default=False),
    )


def parse_force_mode(value: str) -> MacroMode | None:
    token = str(value or "").strip().lower()
    if not token:
        return None
    try:
        return MacroMode(token)
    except ValueError:
        return None


def _build_frame_timeline(*, count: int, inter_frame_ms: float) -> List[Dict[str, float]]:
    span_ms = max(1.0, float(inter_frame_ms))
    out: List[Dict[str, float]] = []
    for idx in range(count):
        age_s = ((count - 1) - idx) * (span_ms / 1000.0)
        out.append({"ordinal": idx + 1, "age_s": round(age_s, 3)})
    return out


def _build_request_payload(
    *,
    params: BehaviorProbeParams,
    user_text: str,
    image_data_urls: List[str],
) -> Dict[str, Any]:
    messages = build_messages(user_text=user_text, image_data_urls=image_data_urls)
    if messages and isinstance(messages[0], dict):
        messages[0]["content"] = params.system_prompt
    return {
        "model": str(params.model),
        "messages": messages,
        "temperature": float(params.temperature),
        "top_p": float(params.top_p),
        "presence_penalty": float(params.presence_penalty),
        "max_tokens": int(params.max_tokens),
        "stream": False,
        "response_format": behavior_decision_response_format(),
    }


def _packet_compact(
    *,
    params: BehaviorProbeParams,
    chat_url: str,
    image_count: int,
    context: Dict[str, Any],
    frame_timeline: List[Dict[str, float]],
    user_text: str,
) -> List[Dict[str, str]]:
    mode_token = str(context.get("mode") or "")
    skill_token = str(context.get("active_skill") or "")
    return [
        {"label": "Provider", "value": params.provider},
        {"label": "Model", "value": params.model},
        {"label": "Chat URL", "value": chat_url},
        {"label": "Image Count", "value": str(image_count)},
        {"label": "Mode", "value": mode_token},
        {"label": "Active Skill", "value": skill_token},
        {"label": "Timeout (s)", "value": f"{params.timeout_s:.2f}"},
        {"label": "Max Tokens", "value": str(params.max_tokens)},
        {"label": "Temperature", "value": f"{params.temperature:.3f}"},
        {"label": "Top P", "value": f"{params.top_p:.3f}"},
        {"label": "Presence Penalty", "value": f"{params.presence_penalty:.3f}"},
        {
            "label": "Frame Timeline",
            "value": ", ".join(f"#{row['ordinal']} age={row['age_s']}s" for row in frame_timeline),
        },
        {"label": "User Prompt Chars", "value": str(len(user_text))},
    ]


def _packet_expanded(
    *,
    params: BehaviorProbeParams,
    context_default: Dict[str, Any],
    context_effective: Dict[str, Any],
    frame_timeline: List[Dict[str, float]],
    images: List[Dict[str, Any]],
    response_format: Dict[str, Any],
    user_text: str,
    overrides: Dict[str, bool],
) -> List[Dict[str, Any]]:
    schema_required = []
    schema = response_format.get("json_schema", {}).get("schema")
    if isinstance(schema, dict):
        required = schema.get("required")
        if isinstance(required, list):
            schema_required = [str(item) for item in required]

    return [
        {
            "title": "Request Target",
            "rows": [
                {"label": "provider", "value": params.provider},
                {"label": "model", "value": params.model},
                {"label": "base_url", "value": params.base_url},
                {"label": "timeout_s", "value": params.timeout_s},
            ],
        },
        {
            "title": "Request Knobs",
            "rows": [
                {"label": "max_tokens", "value": params.max_tokens},
                {"label": "temperature", "value": params.temperature},
                {"label": "top_p", "value": params.top_p},
                {"label": "presence_penalty", "value": params.presence_penalty},
                {"label": "frame_max_width", "value": params.frame_max_width},
                {"label": "frame_jpeg_quality", "value": params.frame_jpeg_quality},
                {"label": "inter_frame_ms", "value": params.inter_frame_ms},
            ],
        },
        {
            "title": "Policy Fields",
            "rows": [
                {"label": "system_prompt", "value": params.system_prompt},
                {"label": "policy_identity", "value": params.policy_identity},
                {"label": "policy_capabilities", "value": params.policy_capabilities},
                {"label": "policy_safety", "value": params.policy_safety},
                {"label": "policy_style", "value": params.policy_style},
                {"label": "planner_prompt", "value": params.planner_prompt},
                {"label": "user_text", "value": user_text},
            ],
        },
        {
            "title": "Context Fields",
            "rows": [
                {"label": "context_default", "value": _json_preview(context_default, max_chars=340)},
                {"label": "context_effective", "value": _json_preview(context_effective, max_chars=340)},
                {"label": "frame_timeline", "value": _json_preview(frame_timeline, max_chars=340)},
                {"label": "context_override_used", "value": overrides.get("context_override_used")},
                {"label": "user_text_override_used", "value": overrides.get("user_text_override_used")},
                {"label": "payload_override_used", "value": overrides.get("payload_override_used")},
            ],
        },
        {
            "title": "Image Packet",
            "rows": [
                {
                    "label": f"Image #{idx + 1}",
                    "value": (
                        f"{img['filename']} | {img['encoded_width']}x{img['encoded_height']} "
                        f"(orig {img['original_width']}x{img['original_height']}) | jpeg_bytes={img['jpeg_bytes']}"
                    ),
                }
                for idx, img in enumerate(images)
            ],
        },
        {
            "title": "Response Format Fields",
            "rows": [
                {"label": "type", "value": response_format.get("type")},
                {"label": "json_schema.name", "value": response_format.get("json_schema", {}).get("name")},
                {"label": "json_schema.strict", "value": response_format.get("json_schema", {}).get("strict")},
                {"label": "schema.required", "value": ", ".join(schema_required)},
            ],
        },
    ]


def _decision_to_dict(parsed: BehaviorDecisionParseResult | None) -> Dict[str, Any] | None:
    if parsed is None:
        return None
    return asdict(parsed.decision)


def _compute_parse_error(
    *,
    response_ok: bool,
    response_error: str | None,
    parsed: BehaviorDecisionParseResult | None,
    parser: BehaviorDecisionParser,
    finish_reason: Any,
) -> str | None:
    if not response_ok:
        return str(response_error or "transport_error")
    if parsed is not None:
        return None
    detail = parser.last_parse_error or "unknown"
    if "schema:$.schema_version:" in detail:
        return (
            "decision_parse_failed:"
            + detail
            + ":hint=check_schema_version_or_click_sync_override_defaults"
        )
    if str(finish_reason or "").strip().lower() == "length":
        return f"decision_parse_failed:truncated_response:finish_reason=length:likely_token_budget_exhausted:{detail}"
    return f"decision_parse_failed:{detail}"


async def run_behavior_probe(
    *,
    params: BehaviorProbeParams,
    upload_files: List[UploadFile],
    image_order: str,
    logs_root: Path,
    simulator: ProbeFsmSimulator,
) -> BehaviorProbeRun:
    api_key = resolve_api_key()
    if api_key is None:
        raise ProbeInputError(f"Missing API key in environment variable {API_KEY_ENV_VAR}.")

    chat_url = normalize_chat_url(params.base_url, provider=params.provider)
    if not chat_url:
        raise ProbeInputError("Unable to resolve chat URL from base URL/provider settings.")

    run_id, run_dir = storage.new_run_dir(logs_root)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prepared, jpeg_payloads = await prepare_images(
        upload_files,
        image_order=image_order,
        max_width=params.frame_max_width,
        jpeg_quality=params.frame_jpeg_quality,
    )
    images_manifest, image_data_urls = _build_images_manifest(
        prepared=prepared,
        jpeg_payloads=jpeg_payloads,
        run_dir=run_dir,
        run_id=run_id,
        logs_root=logs_root,
    )

    frame_timeline = _build_frame_timeline(count=len(images_manifest), inter_frame_ms=params.inter_frame_ms)
    fsm_before = simulator.fsm_snapshot()
    context_default = simulator.build_context()
    context_effective = context_default
    context_override_used = False
    if params.context_override_json:
        candidate = _parse_json_object(params.context_override_json, field_name="context_override_json")
        if candidate != context_default:
            context_effective = candidate
            context_override_used = True

    default_user_text = build_behavior_v4_user_text(
        context=context_effective,
        policy_identity=params.policy_identity,
        policy_capabilities=params.policy_capabilities,
        policy_safety=params.policy_safety,
        policy_style=params.policy_style,
        planner_prompt=params.planner_prompt,
    )
    user_text = default_user_text
    user_text_override_used = False
    if params.user_text_override:
        if params.user_text_override.strip() != default_user_text.strip():
            user_text = params.user_text_override
            user_text_override_used = True

    default_payload = _build_request_payload(params=params, user_text=user_text, image_data_urls=image_data_urls)
    payload = default_payload
    payload_override_used = False
    if params.payload_override_json:
        candidate_payload = _parse_json_object(params.payload_override_json, field_name="payload_override_json")
        if candidate_payload != default_payload:
            payload = candidate_payload
            payload_override_used = True

    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    response_format = payload.get("response_format") if isinstance(payload.get("response_format"), dict) else {}

    result = post_chat_json(
        url=chat_url,
        payload=payload,
        timeout_s=params.timeout_s,
        api_key=api_key,
        provider=params.provider,
    )

    content, reasoning = extract_message_content(result.response_json or {}) if result.response_json else (None, None)
    finish_reason, prompt_tokens, completion_tokens, total_tokens = (
        _response_meta(result.response_json or {}) if result.response_json else (None, None, None, None)
    )

    parser = BehaviorDecisionParser()
    parsed = parser.parse(content or "")
    parse_error = _compute_parse_error(
        response_ok=bool(result.ok),
        response_error=result.error,
        parsed=parsed,
        parser=parser,
        finish_reason=finish_reason,
    )

    model_age_s = max(0.0, float(result.latency_ms) / 1000.0)
    applied = simulator.apply_model_outcome(parsed=parsed if result.ok else None, parse_error=parse_error, model_age_s=model_age_s)

    response_meta = {
        "http_ok": bool(result.ok),
        "http_status": int(result.status_code),
        "latency_ms": round(float(result.latency_ms), 1),
        "error": None if result.ok else result.error,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    request_payload_redacted = _redact_data_urls(payload)
    message_structure = _message_structure(messages)
    overrides = {
        "context_override_used": context_override_used,
        "user_text_override_used": user_text_override_used,
        "payload_override_used": payload_override_used,
    }
    packet_compact = _packet_compact(
        params=params,
        chat_url=chat_url,
        image_count=len(images_manifest),
        context=context_effective,
        frame_timeline=frame_timeline,
        user_text=user_text,
    )
    packet_expanded = _packet_expanded(
        params=params,
        context_default=context_default,
        context_effective=context_effective,
        frame_timeline=frame_timeline,
        images=images_manifest,
        response_format=response_format,
        user_text=user_text,
        overrides=overrides,
    )

    effective_inputs = {
        "target": {
            "provider": params.provider,
            "model": params.model,
            "base_url": params.base_url,
            "chat_url": chat_url,
            "timeout_s": params.timeout_s,
        },
        "knobs": {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "frame_max_width": params.frame_max_width,
            "frame_jpeg_quality": params.frame_jpeg_quality,
            "inter_frame_ms": params.inter_frame_ms,
        },
        "policy_fields": {
            "system_prompt": params.system_prompt,
            "policy_identity": params.policy_identity,
            "policy_capabilities": params.policy_capabilities,
            "policy_safety": params.policy_safety,
            "policy_style": params.policy_style,
            "planner_prompt": params.planner_prompt,
        },
        "context_default": context_default,
        "context_effective": context_effective,
        "user_text_default": default_user_text,
        "user_text_effective": user_text,
        "payload_default": _redact_data_urls(default_payload),
        "payload_effective": request_payload_redacted,
        "overrides": overrides,
        "response_format": response_format,
        "message_structure": message_structure,
        "request_payload_redacted": request_payload_redacted,
        "frame_timeline": frame_timeline,
        "images": images_manifest,
        "fsm_before": fsm_before,
        "fsm_after": applied.get("fsm_after"),
        "mode_transition_applied": applied.get("mode_transition_applied"),
    }

    parsed_output = _decision_to_dict(parsed) if result.ok else None
    parse_stage = parser.last_parse_stage if result.ok else "transport"

    run = BehaviorProbeRun(
        run_id=run_id,
        created_at_utc=created_at,
        mode="behavior_v4",
        params={
            "mode": "behavior_v4",
            "provider": params.provider,
            "model": params.model,
            "base_url": params.base_url,
            "chat_url": chat_url,
            "system_prompt": params.system_prompt,
            "timeout_s": params.timeout_s,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "frame_max_width": params.frame_max_width,
            "frame_jpeg_quality": params.frame_jpeg_quality,
            "policy_identity": params.policy_identity,
            "policy_capabilities": params.policy_capabilities,
            "policy_safety": params.policy_safety,
            "policy_style": params.policy_style,
            "planner_prompt": params.planner_prompt,
            "context_override_json": params.context_override_json,
            "user_text_override": params.user_text_override,
            "payload_override_json": params.payload_override_json,
            "inter_frame_ms": params.inter_frame_ms,
            "packet_view_mode": params.packet_view_mode,
        },
        images=images_manifest,
        packet_compact=packet_compact,
        packet_expanded=packet_expanded,
        message_structure=message_structure,
        request_payload_redacted=request_payload_redacted,
        response_meta=response_meta,
        raw_content=content,
        reasoning_content=reasoning,
        parse_ok=bool(result.ok) and parsed is not None,
        parse_stage=parse_stage,
        parse_error=parse_error,
        parsed_output=parsed_output,
        guard_result=applied.get("guard_result"),
        final_action=(applied.get("guard_result") or {}).get("action"),
        effective_inputs=effective_inputs,
        fsm_before=fsm_before,
        fsm_after=applied.get("fsm_after"),
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
        "mode": (loaded.get("summary") or {}).get("mode", "behavior_v4"),
        "params": loaded.get("run_config", {}),
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
        "guard_result": loaded.get("guard_result") or {},
        "final_action": loaded.get("final_action") or {},
        "summary": loaded.get("summary", {}),
        "effective_inputs": loaded.get("effective_inputs") or {},
        "fsm_before": loaded.get("fsm_before") or {},
        "fsm_after": loaded.get("fsm_after") or {},
    }
