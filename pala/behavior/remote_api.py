from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Dict, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request


@dataclass
class RemoteCallResult:
    ok: bool
    status_code: int
    latency_ms: float
    response_json: Optional[Dict[str, Any]]
    error: Optional[str]


def normalize_chat_url(base_url: str) -> str:
    base = str(base_url or "").strip()
    if not base:
        return ""
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if base.endswith("/v1/"):
        return f"{base}chat/completions"
    return f"{base.rstrip('/')}/v1/chat/completions"


def post_chat_json(
    *,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
    api_key: Optional[str],
) -> RemoteCallResult:
    t0 = time.monotonic()
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    req = urllib_request.Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as resp:
            status_code = int(resp.status)
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return RemoteCallResult(
            ok=False,
            status_code=int(exc.code),
            latency_ms=(time.monotonic() - t0) * 1000.0,
            response_json=None,
            error=f"http_{exc.code}:{raw}",
        )
    except urllib_error.URLError as exc:
        return RemoteCallResult(
            ok=False,
            status_code=0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            response_json=None,
            error=f"transport:{exc.reason}",
        )
    except Exception as exc:  # noqa: BLE001 - transport errors must not crash loops
        return RemoteCallResult(
            ok=False,
            status_code=0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            response_json=None,
            error=f"transport:{type(exc).__name__}:{exc}",
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return RemoteCallResult(
            ok=False,
            status_code=status_code,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            response_json=None,
            error="invalid_json_response",
        )
    if not isinstance(data, dict):
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

