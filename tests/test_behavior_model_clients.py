from __future__ import annotations

from types import SimpleNamespace

import pytest

from pala.behavior.model_clients.factory import build_model_client, normalize_chat_url
from pala.behavior.model_clients.openai_compat_client import (
    OpenAICompatClient,
    _CLIENT_CACHE,
    _coerce_response_dict,
    _extract_error_detail,
    _extract_status_code,
    _get_openai_client,
    _request_payload,
    _short_text,
)
from pala.behavior.model_clients.types import ModelRequest


def test_normalize_chat_url_provider_aware():
    assert normalize_chat_url("http://x") == "http://x/v1/chat/completions"
    assert normalize_chat_url("http://x/v1") == "http://x/v1/chat/completions"
    assert normalize_chat_url("https://generativelanguage.googleapis.com/v1beta/openai") == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert normalize_chat_url("https://proxy.example/service", provider="gemini") == (
        "https://proxy.example/service/v1beta/openai/chat/completions"
    )


def test_build_model_client_returns_openai_compat():
    client = build_model_client(provider="cosmos", base_url="http://x", api_key=None)
    assert isinstance(client, OpenAICompatClient)


def test_normalize_chat_url_and_build_client_error_paths():
    assert normalize_chat_url("", provider="openai") == ""
    assert normalize_chat_url("http://x/chat/completions") == "http://x/chat/completions"
    assert normalize_chat_url("https://generativelanguage.googleapis.com/v1beta", provider="auto").endswith(
        "/v1beta/openai/chat/completions"
    )
    assert normalize_chat_url("http://x", provider="unknown-provider") == "http://x/v1/chat/completions"
    with pytest.raises(ValueError, match="base_url is required"):
        build_model_client(provider="openai", base_url="", api_key=None)


def test_request_payload_and_response_coercion_helpers():
    req = ModelRequest(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        response_format={"type": "json_object"},
        timeout_s=2.0,
        max_tokens=123.9,
        temperature=0.2,
        top_p=0.6,
        presence_penalty=0.1,
        stream=True,
        extra_body={"a": 1},
    )
    payload = _request_payload(req)
    assert payload["model"] == "m"
    assert payload["stream"] is True
    assert payload["max_tokens"] == 123
    assert payload["extra_body"] == {"a": 1}

    assert _coerce_response_dict({"ok": True}) == {"ok": True}
    assert _coerce_response_dict(SimpleNamespace(model_dump=lambda: {"a": 1})) == {"a": 1}
    assert _coerce_response_dict(SimpleNamespace(to_dict=lambda: {"b": 2})) == {"b": 2}
    assert _coerce_response_dict(SimpleNamespace(dict=lambda: {"c": 3})) == {"c": 3}
    assert _coerce_response_dict(SimpleNamespace(model_dump=lambda: "bad")) is None


def test_status_and_error_detail_helpers():
    class _Resp:
        def __init__(self):
            self.status_code = 429
            self.text = "x" * 600
            self.body = ""

    class _Exc(Exception):
        def __init__(self):
            super().__init__("too many requests")
            self.response = _Resp()

    exc = _Exc()
    assert _extract_status_code(exc) == 429
    detail = _extract_error_detail(exc)
    assert len(detail) <= 512
    assert detail.endswith("...")
    assert _short_text("a\n b", max_len=10) == "a b"

    class _BodyResp:
        status_code = 0
        text = ""
        body = "from_body"

    class _BodyExc(Exception):
        def __init__(self):
            super().__init__("fallback")
            self.response = _BodyResp()

    assert _extract_status_code(_BodyExc()) == 0
    assert _extract_error_detail(_BodyExc()) == "from_body"


def test_get_openai_client_caches_instances(monkeypatch):
    _CLIENT_CACHE.clear()
    calls = []

    class _OpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(
        "pala.behavior.model_clients.openai_compat_client.importlib.import_module",
        lambda name: SimpleNamespace(OpenAI=_OpenAI),
    )

    c1 = _get_openai_client(base_url="http://x/v1", api_key=None)
    c2 = _get_openai_client(base_url="http://x/v1", api_key=None)
    c3 = _get_openai_client(base_url="http://x/v1", api_key="k")
    assert c1 is c2
    assert c1 is not c3
    assert calls[0] == {"base_url": "http://x/v1"}
    assert calls[1] == {"base_url": "http://x/v1", "api_key": "k"}


def test_openai_compat_client_chat_success_and_error_paths(monkeypatch):
    class _Raw:
        def __init__(self, parsed, status_code=201):
            self._parsed = parsed
            self.status_code = status_code

        def parse(self):
            return self._parsed

    class _Completions:
        def __init__(self, parsed):
            self.with_raw_response = SimpleNamespace(create=lambda timeout, **payload: _Raw(parsed=parsed, status_code=202))

    class _Client:
        def __init__(self, parsed):
            self.chat = SimpleNamespace(completions=_Completions(parsed))

    req = ModelRequest(model="m", messages=[{"role": "user", "content": "x"}], timeout_s=1.0)
    client = OpenAICompatClient(base_url="http://x/v1", api_key=None)

    monkeypatch.setattr("pala.behavior.model_clients.openai_compat_client._get_openai_client", lambda **_: _Client({"ok": 1}))
    ok = client.chat(req)
    assert ok.ok is True
    assert ok.status_code == 202
    assert ok.response_json == {"ok": 1}
    assert ok.error is None

    monkeypatch.setattr(
        "pala.behavior.model_clients.openai_compat_client._get_openai_client",
        lambda **_: _Client(SimpleNamespace(model_dump=lambda: "bad")),
    )
    bad_type = client.chat(req)
    assert bad_type.ok is False
    assert bad_type.error == "invalid_response_type"

    monkeypatch.setattr(
        "pala.behavior.model_clients.openai_compat_client._get_openai_client",
        lambda **_: (_ for _ in ()).throw(ModuleNotFoundError("openai missing")),
    )
    missing_sdk = client.chat(req)
    assert missing_sdk.ok is False
    assert missing_sdk.error == "transport:openai_sdk_not_installed"

    class _HttpExc(Exception):
        def __init__(self):
            super().__init__("denied")
            self.status_code = 401
            self.response = SimpleNamespace(text="denied", body="")

    monkeypatch.setattr(
        "pala.behavior.model_clients.openai_compat_client._get_openai_client",
        lambda **_: (_ for _ in ()).throw(_HttpExc()),
    )
    http_err = client.chat(req)
    assert http_err.ok is False
    assert http_err.status_code == 401
    assert http_err.error == "http_401:denied"

    monkeypatch.setattr(
        "pala.behavior.model_clients.openai_compat_client._get_openai_client",
        lambda **_: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    transport = client.chat(req)
    assert transport.ok is False
    assert transport.status_code == 0
    assert (transport.error or "").startswith("transport:RuntimeError:")
