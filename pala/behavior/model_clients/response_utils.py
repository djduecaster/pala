from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .factory import build_model_client, normalize_chat_url
from .types import ModelRequest, ModelResponse


def post_chat_json(
    *,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
    api_key: Optional[str],
    provider: Optional[str] = None,
) -> ModelResponse:
    model = payload.get("model")
    messages = payload.get("messages")
    if not isinstance(model, str) or not isinstance(messages, list):
        return ModelResponse(
            ok=False,
            status_code=0,
            latency_ms=0.0,
            response_json=None,
            error="invalid_payload:model_or_messages",
        )

    request = ModelRequest(
        model=model,
        messages=messages,
        response_format=payload.get("response_format") if isinstance(payload.get("response_format"), dict) else None,
        timeout_s=float(timeout_s),
        max_tokens=int(payload["max_tokens"]) if payload.get("max_tokens") is not None else None,
        temperature=float(payload["temperature"]) if payload.get("temperature") is not None else None,
        top_p=float(payload["top_p"]) if payload.get("top_p") is not None else None,
        presence_penalty=float(payload["presence_penalty"]) if payload.get("presence_penalty") is not None else None,
        stream=bool(payload.get("stream", False)),
        extra_body=payload.get("extra_body") if isinstance(payload.get("extra_body"), dict) else None,
    )

    client = build_model_client(
        provider=str(provider or "auto"),
        base_url=normalize_chat_url(url, provider=provider),
        api_key=api_key,
    )
    return client.chat(request)


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
