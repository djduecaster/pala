from __future__ import annotations

from types import SimpleNamespace

from pala.behavior.model_clients.response_utils import _coerce_text, extract_message_content, post_chat_json
from pala.behavior.model_clients.types import ModelResponse


def test_post_chat_json_rejects_invalid_payload_shape():
    result = post_chat_json(
        url="http://unit.test/v1/chat/completions",
        payload={"messages": []},
        timeout_s=2.0,
        api_key=None,
        provider="openai",
    )
    assert result.ok is False
    assert result.error == "invalid_payload:model_or_messages"

    result = post_chat_json(
        url="http://unit.test/v1/chat/completions",
        payload={"model": "x", "messages": "not-a-list"},
        timeout_s=2.0,
        api_key=None,
        provider="openai",
    )
    assert result.ok is False
    assert result.error == "invalid_payload:model_or_messages"


def test_post_chat_json_builds_request_and_delegates_to_client(monkeypatch):
    captured = {}

    class _Client:
        def chat(self, request):
            captured["request"] = request
            return ModelResponse(ok=True, status_code=201, latency_ms=12.0, response_json={"ok": True}, error=None)

    monkeypatch.setattr(
        "pala.behavior.model_clients.response_utils.normalize_chat_url",
        lambda url, provider=None: "http://normalized/chat/completions",
    )
    monkeypatch.setattr(
        "pala.behavior.model_clients.response_utils.build_model_client",
        lambda *, provider, base_url, api_key: captured.update(
            {"provider": provider, "base_url": base_url, "api_key": api_key}
        )
        or _Client(),
    )

    response = post_chat_json(
        url="http://raw.url",
        payload={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema"},
            "max_tokens": "128",
            "temperature": "0.1",
            "top_p": "0.3",
            "presence_penalty": "0",
            "stream": 1,
            "extra_body": {"debug": True},
        },
        timeout_s=3.5,
        api_key="secret",
        provider="gemini",
    )
    assert response.ok is True
    assert response.status_code == 201
    assert response.response_json == {"ok": True}

    req = captured["request"]
    assert req.model == "demo-model"
    assert req.messages[0]["content"] == "hi"
    assert req.response_format == {"type": "json_schema"}
    assert req.timeout_s == 3.5
    assert req.max_tokens == 128
    assert abs(req.temperature - 0.1) < 1e-9
    assert abs(req.top_p - 0.3) < 1e-9
    assert req.presence_penalty == 0.0
    assert req.stream is True
    assert req.extra_body == {"debug": True}

    assert captured["provider"] == "gemini"
    assert captured["base_url"] == "http://normalized/chat/completions"
    assert captured["api_key"] == "secret"


def test_post_chat_json_ignores_non_dict_optional_payload_fields(monkeypatch):
    seen = {}

    class _Client:
        def chat(self, request):
            seen["request"] = request
            return ModelResponse(ok=True, status_code=200, latency_ms=1.0, response_json=None, error=None)

    monkeypatch.setattr(
        "pala.behavior.model_clients.response_utils.build_model_client",
        lambda **kwargs: _Client(),
    )

    post_chat_json(
        url="http://u",
        payload={
            "model": "m",
            "messages": [],
            "response_format": "json_object",
            "extra_body": "not-a-dict",
            "stream": 0,
        },
        timeout_s=1.0,
        api_key=None,
        provider=None,
    )
    req = seen["request"]
    assert req.response_format is None
    assert req.extra_body is None
    assert req.stream is False


def test_extract_message_content_and_coerce_text_paths():
    content, reasoning = extract_message_content(
        {
            "choices": [
                {
                    "message": {
                        "content": [{"text": "line1"}, {"text": "line2"}],
                        "reasoning_content": {"text": "r1"},
                    }
                }
            ]
        }
    )
    assert content == "line1\nline2"
    assert reasoning == "r1"

    # Fallback reasoning from top-level choice when message.reasoning_content is absent.
    content, reasoning = extract_message_content(
        {
            "choices": [
                {
                    "message": {"content": "  ok  "},
                    "reasoning_content": [{"text": "step1"}, {"text": "step2"}],
                }
            ]
        }
    )
    assert content == "ok"
    assert reasoning == "step1\nstep2"

    assert extract_message_content({"choices": []}) == (None, None)
    assert extract_message_content({"choices": [1]}) == (None, None)
    assert extract_message_content({"choices": [{"message": []}]}) == (None, None)

    assert _coerce_text("  x  ") == "x"
    assert _coerce_text("   ") is None
    assert _coerce_text([{"text": "a"}, {"bad": "x"}, {"text": "b"}]) == "a\nb"
    assert _coerce_text([{"bad": "x"}]) is None
    assert _coerce_text({"text": " y "}) == "y"
    assert _coerce_text({"text": "   "}) is None
    assert _coerce_text(SimpleNamespace(text="z")) is None
