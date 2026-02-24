from __future__ import annotations

from typing import Protocol

from .types import ModelRequest, ModelResponse


class BaseModelClient(Protocol):
    def chat(self, request: ModelRequest) -> ModelResponse:
        ...
