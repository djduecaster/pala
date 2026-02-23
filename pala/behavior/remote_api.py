from __future__ import annotations

from dataclasses import dataclass
import importlib
import threading
import time
from typing import Any, Dict, Optional, Tuple


@dataclass
class RemoteCallResult:
    ok: bool
    status_code: int
    latency_ms: float
    response_json: Optional[Dict[str, Any]]
    error: Optional[str]


_CLIENT_CACHE: Dict[tuple[str, str], Any] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


def normalize_chat_url(base_url: str, *, provider: Optional[str] = None) -> str:
    base = str(base_url or "").strip()
    if not base:
        return ""
    token = base.rstrip("/")
    if token.endswith("/chat/completions"):
        return token

    provider_name = _resolve_provider(base_url=token, provider=provider)
    if provider_name == "gemini":
        if token.endswith("/v1beta/openai"):
            return f"{token}/chat/completions"
        if token.endswith("/v1beta"):
            return f"{token}/openai/chat/completions"
        return f"{token}/v1beta/openai/chat/completions"

    if token.endswith("/v1"):
        return f"{token}/chat/completions"
    return f"{token}/v1/chat/completions"


def post_chat_json(
    *,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
    api_key: Optional[str],
    provider: Optional[str] = None,
) -> RemoteCallResult:
    t0 = time.monotonic()
    chat_url = normalize_chat_url(url, provider=provider)
    base_url = _chat_url_to_base_url(chat_url)

    try:
        client = _get_openai_client(base_url=base_url, api_key=api_key)
        raw_response = client.chat.completions.with_raw_response.create(timeout=timeout_s, **payload)
        status_code = int(getattr(raw_response, "status_code", 200))
        parsed = raw_response.parse()
        data = _coerce_response_dict(parsed)
        if data is None:
            return RemoteCallResult(
                ok=False,
                status_code=status_code,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                response_json=None,
                error="invalid_response_type",
            )
        return RemoteCallResult(
            ok=True,
            status_code=status_code,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            response_json=data,
            error=None,
        )
    except ModuleNotFoundError:
        return RemoteCallResult(
            ok=False,
            status_code=0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            response_json=None,
            error="transport:openai_sdk_not_installed",
        )
    except Exception as exc:  # noqa: BLE001 - transport errors must not crash loops
        status_code = _extract_status_code(exc)
        if status_code > 0:
            detail = _extract_error_detail(exc)
            return RemoteCallResult(
                ok=False,
                status_code=status_code,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                response_json=None,
                error=f"http_{status_code}:{detail}",
            )
        return RemoteCallResult(
            ok=False,
            status_code=0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            response_json=None,
            error=f"transport:{type(exc).__name__}:{exc}",
        )


def extract_message_content(response: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, None
    first = choices[0]
    if not isinstance(first, dict):
        return None, None
    message = first.get("message")
    if not isinstance(message, dict):
        return None, None
    content = _coerce_text(message.get("content"))
    reasoning = _coerce_text(message.get("reasoning_content"))
    if reasoning is None and isinstance(first.get("reasoning_content"), (str, list, dict)):
        reasoning = _coerce_text(first.get("reasoning_content"))
    return content, reasoning


def _coerce_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        token = value.strip()
        return token if token else None
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            token = text.strip()
            return token if token else None
    return None


def _chat_url_to_base_url(chat_url: str) -> str:
    token = str(chat_url or "").rstrip("/")
    suffix = "/chat/completions"
    if token.endswith(suffix):
        return token[: -len(suffix)]
    return token


def _resolve_provider(*, base_url: str, provider: Optional[str]) -> str:
    token = str(provider or "").strip().lower()
    if token in {"openai", "cosmos", "gemini"}:
        return token
    if token not in {"", "auto"}:
        return "openai"
    return _infer_provider_from_url(base_url)


def _infer_provider_from_url(base_url: str) -> str:
    token = str(base_url or "").strip().lower()
    if "generativelanguage.googleapis.com" in token:
        return "gemini"
    if token.endswith("/v1beta/openai") or "/v1beta/openai/" in token:
        return "gemini"
    return "openai"


def _get_openai_client(*, base_url: str, api_key: Optional[str]) -> Any:
    cache_key = (base_url, api_key or "")
    with _CLIENT_CACHE_LOCK:
        cached = _CLIENT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        mod = importlib.import_module("openai")
        kwargs: Dict[str, Any] = {"base_url": base_url}
        if api_key:
            kwargs["api_key"] = api_key
        client = mod.OpenAI(**kwargs)
        _CLIENT_CACHE[cache_key] = client
        return client


def _coerce_response_dict(parsed: Any) -> Optional[Dict[str, Any]]:
    if isinstance(parsed, dict):
        return parsed
    model_dump = getattr(parsed, "model_dump", None)
    if callable(model_dump):
        data = model_dump()
        return data if isinstance(data, dict) else None
    to_dict = getattr(parsed, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        return data if isinstance(data, dict) else None
    dict_fn = getattr(parsed, "dict", None)
    if callable(dict_fn):
        data = dict_fn()
        return data if isinstance(data, dict) else None
    return None


def _extract_status_code(exc: Exception) -> int:
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int) and direct > 0:
        return direct
    response = getattr(exc, "response", None)
    nested = getattr(response, "status_code", None)
    if isinstance(nested, int) and nested > 0:
        return nested
    return 0


def _extract_error_detail(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return _short_text(text.strip(), max_len=512)
    body = getattr(response, "body", None)
    if isinstance(body, str) and body.strip():
        return _short_text(body.strip(), max_len=512)
    return _short_text(str(exc), max_len=512)


def _short_text(value: str, *, max_len: int) -> str:
    token = " ".join(value.split()).strip()
    if len(token) <= max_len:
        return token
    return token[: max_len - 3] + "..."
