from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


ModelProviderName = Literal["cosmos", "gemini", "openai", "auto"]


@dataclass(frozen=True)
class ModelRequest:
    model: str
    messages: List[Dict[str, Any]]
    response_format: Optional[Dict[str, Any]] = None
    timeout_s: float = 20.0
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    stream: bool = False
    extra_body: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ModelResponse:
    ok: bool
    status_code: int
    latency_ms: float
    response_json: Optional[Dict[str, Any]]
    error: Optional[str]
