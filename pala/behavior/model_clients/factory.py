from __future__ import annotations

from typing import Optional

from .base import BaseModelClient
from .openai_compat_client import OpenAICompatClient
from .types import ModelProviderName


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


def build_model_client(
    *,
    provider: ModelProviderName,
    base_url: str,
    api_key: Optional[str],
) -> BaseModelClient:
    chat_url = normalize_chat_url(base_url, provider=provider)
    if not chat_url:
        raise ValueError("base_url is required")
    base = _chat_url_to_base_url(chat_url)
    return OpenAICompatClient(base_url=base, api_key=api_key)


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
