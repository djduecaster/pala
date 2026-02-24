from __future__ import annotations

from dataclasses import dataclass
import importlib
import threading
import time
from typing import Any, Dict, Optional

from .types import ModelRequest, ModelResponse

_CLIENT_CACHE: Dict[tuple[str, str], Any] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class OpenAICompatClient:
    base_url: str
    api_key: Optional[str]

    def chat(self, request: ModelRequest) -> ModelResponse:
        t0 = time.monotonic()
        try:
            client = _get_openai_client(base_url=self.base_url, api_key=self.api_key)
            payload = _request_payload(request)
            raw = client.chat.completions.with_raw_response.create(timeout=request.timeout_s, **payload)
            status_code = int(getattr(raw, "status_code", 200))
            parsed = raw.parse()
            data = _coerce_response_dict(parsed)
            if data is None:
                return ModelResponse(
                    ok=False,
                    status_code=status_code,
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    response_json=None,
                    error="invalid_response_type",
                )
            return ModelResponse(
                ok=True,
                status_code=status_code,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                response_json=data,
                error=None,
            )
        except ModuleNotFoundError:
            return ModelResponse(
                ok=False,
                status_code=0,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                response_json=None,
                error="transport:openai_sdk_not_installed",
            )
        except Exception as exc:  # noqa: BLE001
            status_code = _extract_status_code(exc)
            if status_code > 0:
                detail = _extract_error_detail(exc)
                return ModelResponse(
                    ok=False,
                    status_code=status_code,
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    response_json=None,
                    error=f"http_{status_code}:{detail}",
                )
            return ModelResponse(
                ok=False,
                status_code=0,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                response_json=None,
                error=f"transport:{type(exc).__name__}:{exc}",
            )


def _request_payload(request: ModelRequest) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": request.model,
        "messages": request.messages,
        "stream": bool(request.stream),
    }
    if request.response_format is not None:
        payload["response_format"] = request.response_format
    if request.max_tokens is not None:
        payload["max_tokens"] = int(request.max_tokens)
    if request.temperature is not None:
        payload["temperature"] = float(request.temperature)
    if request.top_p is not None:
        payload["top_p"] = float(request.top_p)
    if request.presence_penalty is not None:
        payload["presence_penalty"] = float(request.presence_penalty)
    if request.extra_body:
        payload["extra_body"] = dict(request.extra_body)
    return payload


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
