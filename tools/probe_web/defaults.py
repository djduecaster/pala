from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from pala.behavior.prompts import SYSTEM_PROMPT
from pala.config import load_config

from .models import ProbeDefaults


API_KEY_ENV_VAR = "PALA_COSMOS_API_KEY"
DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
_ASSIGNMENT_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _cfg_value(obj: Any, name: str, fallback: Any) -> Any:
    if obj is None:
        return fallback
    return getattr(obj, name, fallback)


def _env_file_path() -> Path:
    override = os.getenv("PALA_ENV_FILE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "pala" / "env.sh"


def _key_from_env_file(path: Path, *, var_name: str) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in lines:
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        match = _ASSIGNMENT_RE.match(token)
        if not match:
            continue
        name = match.group(1)
        if name != var_name:
            continue
        raw = match.group(2).strip()
        if not raw:
            continue
        if raw[0] not in {"'", '"'}:
            raw = raw.split(" #", 1)[0].strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
            raw = raw[1:-1]
        value = raw.strip()
        if value:
            return value
    return None


def resolve_api_key_info() -> tuple[str | None, str]:
    key = os.getenv(API_KEY_ENV_VAR)
    if key is not None and key.strip():
        return key.strip(), f"{API_KEY_ENV_VAR} (process env)"

    env_file = _env_file_path()
    file_key = _key_from_env_file(env_file, var_name=API_KEY_ENV_VAR)
    if file_key:
        return file_key, f"{env_file} ({API_KEY_ENV_VAR} fallback)"

    return None, f"{API_KEY_ENV_VAR} / {_env_file_path()} (not found)"


def resolve_defaults() -> ProbeDefaults:
    cosmos = None
    try:
        cfg = load_config("config/robot.yaml")
        cosmos = getattr(cfg, "cosmos", None)
    except Exception:
        cosmos = None

    timeout_s = max(1.0, float(_cfg_value(cosmos, "request_timeout_ms", 20000)) / 1000.0)
    max_tokens = int(_cfg_value(cosmos, "planner_max_tokens", 1000))
    frame_max_width = int(_cfg_value(cosmos, "summary_max_width", 320))
    frame_jpeg_quality = int(_cfg_value(cosmos, "summary_jpeg_quality", 55))

    policy_identity = str(
        _cfg_value(
            cosmos,
            "policy_identity",
            "You are PALA, a social desk companion lamp that should feel alive, expressive, and safe.",
        )
    )
    policy_capabilities = str(
        _cfg_value(
            cosmos,
            "policy_capabilities",
            (
                "You can move head/neck joints via primitives: hold, breath, glance, nod, orient_to_zone. "
                "You cannot manipulate external objects, move base position, or physically touch users."
            ),
        )
    )
    policy_safety = str(
        _cfg_value(
            cosmos,
            "policy_safety",
            "Avoid sudden aggressive motion. Prefer stable conservative actions.",
        )
    )
    policy_style = str(
        _cfg_value(
            cosmos,
            "policy_style",
            "Default style is calm; use curious for gentle tracking and focused for attentive task support.",
        )
    )

    planner_prompt = str(
        _cfg_value(
            cosmos,
            "planner_prompt",
            "Prioritize calm, safe desk-companion behavior. Always choose one concrete next action.",
        )
    )

    min_action_dwell_s = float(_cfg_value(cosmos, "arbiter_min_dwell_s", 0.8))
    stale_after_s = float(_cfg_value(cosmos, "stale_expire_s", 6.0))
    min_mode_dwell_s = float(_cfg_value(cosmos, "mode_min_dwell_s", 1.2))
    engage_person_conf = float(_cfg_value(cosmos, "mode_engage_person_conf", 0.45))
    disengage_person_conf = float(_cfg_value(cosmos, "mode_disengage_person_conf", 0.25))

    base_url = (os.getenv("PALA_COSMOS_BASE_URL") or DEFAULT_BASE_URL).strip()
    model = (os.getenv("PALA_COSMOS_MODEL") or DEFAULT_MODEL).strip()
    if DEFAULT_PROVIDER == "gemini" and "gemini" in model.lower():
        max_tokens = max(max_tokens, 1000)

    api_key, api_key_source = resolve_api_key_info()
    has_api_key = api_key is not None

    return ProbeDefaults(
        provider=DEFAULT_PROVIDER,
        model=model,
        base_url=base_url,
        system_prompt=SYSTEM_PROMPT,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        frame_max_width=max(64, frame_max_width),
        frame_jpeg_quality=max(30, min(95, frame_jpeg_quality)),
        policy_identity=policy_identity,
        policy_capabilities=policy_capabilities,
        policy_safety=policy_safety,
        policy_style=policy_style,
        planner_prompt=planner_prompt,
        context_override_json="",
        user_text_override="",
        payload_override_json="",
        inter_frame_ms=1000.0,
        packet_view_mode="expanded",
        min_action_dwell_s=max(0.0, min_action_dwell_s),
        stale_after_s=max(0.2, stale_after_s),
        min_mode_dwell_s=max(0.0, min_mode_dwell_s),
        engage_person_conf=max(0.0, min(1.0, engage_person_conf)),
        disengage_person_conf=max(0.0, min(1.0, disengage_person_conf)),
        api_key_source=api_key_source,
        has_api_key=has_api_key,
    )


def resolve_api_key() -> str | None:
    key, _source = resolve_api_key_info()
    return key
