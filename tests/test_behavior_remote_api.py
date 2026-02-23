from __future__ import annotations

from dataclasses import dataclass

from pala.behavior import remote_api
from pala.behavior.remote_api import extract_message_content, normalize_chat_url, post_chat_json


def test_normalize_chat_url_variants():
    assert normalize_chat_url("") == ""
    assert normalize_chat_url("  ") == ""
    assert normalize_chat_url("http://x") == "http://x/v1/chat/completions"
    assert normalize_chat_url("http://x/") == "http://x/v1/chat/completions"
    assert normalize_chat_url("http://x/v1") == "http://x/v1/chat/completions"
    assert normalize_chat_url("http://x/v1/") == "http://x/v1/chat/completions"
    assert normalize_chat_url("http://x/v1/chat/completions") == "http://x/v1/chat/completions"
    assert (
        normalize_chat_url("https://generativelanguage.googleapis.com")
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert (
        normalize_chat_url("https://generativelanguage.googleapis.com/v1beta/openai/")
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert (
        normalize_chat_url("https://proxy.example/custom/chat/completions", provider="gemini")
        == "https://proxy.example/custom/chat/completions"
    )
    assert (
        normalize_chat_url("https://proxy.example/service", provider="gemini")
        == "https://proxy.example/service/v1beta/openai/chat/completions"
    )


def test_extract_message_content_handles_string_list_dict_and_fallback_reasoning():
    content, reasoning = extract_message_content(
        {
            "choices": [
                {
                    "message": {
                        "content": [{"text": " line-1 "}, {"text": "line-2"}],
                        "reasoning_content": "",
                    },
                    "reasoning_content": {"text": "fallback reasoning"},
                }
            ]
        }
    )
    assert content == "line-1\nline-2"
    assert reasoning == "fallback reasoning"

    content2, reasoning2 = extract_message_content(
        {
            "choices": [
                {
                    "message": {
                        "content": "  hello  ",
                        "reasoning_content": {"text": "  message reasoning  "},
                    }
                }
            ]
        }
    )
    assert content2 == "hello"
    assert reasoning2 == "message reasoning"


def test_extract_message_content_missing_shapes_fail_closed():
    assert extract_message_content({}) == (None, None)
    assert extract_message_content({"choices": []}) == (None, None)
    assert extract_message_content({"choices": ["bad"]}) == (None, None)
    assert extract_message_content({"choices": [{"message": "bad"}]}) == (None, None)


@dataclass
class _Parsed:
    payload: dict

    def model_dump(self):
        return dict(self.payload)


@dataclass
class _RawResponse:
    status_code: int
    parsed: object

    def parse(self):
        return self.parsed


class _WithRawResponse:
    def __init__(self, factory):
        self._factory = factory

    def create(self, **kwargs):
        return self._factory(**kwargs)


class _Completions:
    def __init__(self, factory):
        self.with_raw_response = _WithRawResponse(factory)


class _Chat:
    def __init__(self, factory):
        self.completions = _Completions(factory)


class _Client:
    def __init__(self, factory):
        self.chat = _Chat(factory)


def test_post_chat_json_success(monkeypatch):
    captured = {}
    client_base = {}

    def _factory(**kwargs):
        captured.update(kwargs)
        return _RawResponse(
            status_code=200,
            parsed=_Parsed({"choices": [{"message": {"content": "ok"}}]}),
        )

    monkeypatch.setattr(
        remote_api,
        "_get_openai_client",
        lambda *, base_url, api_key: client_base.update({"base_url": base_url, "api_key": api_key}) or _Client(_factory),
    )

    result = post_chat_json(
        url="http://unit.test/v1/chat/completions",
        payload={"model": "x", "messages": []},
        timeout_s=1.25,
        api_key="abc",
    )
    assert result.ok is True
    assert result.status_code == 200
    assert result.response_json == {"choices": [{"message": {"content": "ok"}}]}
    assert result.error is None
    assert captured["timeout"] == 1.25
    assert captured["model"] == "x"
    assert client_base["base_url"] == "http://unit.test/v1"


def test_post_chat_json_provider_routes_gemini_base(monkeypatch):
    client_base = {}

    monkeypatch.setattr(
        remote_api,
        "_get_openai_client",
        lambda *, base_url, api_key: client_base.update({"base_url": base_url}) or _Client(
            lambda **kwargs: _RawResponse(status_code=200, parsed={"choices": [{"message": {"content": "ok"}}]})
        ),
    )
    result = post_chat_json(
        url="https://generativelanguage.googleapis.com",
        payload={"messages": []},
        timeout_s=1.0,
        api_key="k",
        provider="gemini",
    )
    assert result.ok is True
    assert client_base["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_post_chat_json_rejects_non_object_payload(monkeypatch):
    monkeypatch.setattr(
        remote_api,
        "_get_openai_client",
        lambda *, base_url, api_key: _Client(lambda **kwargs: _RawResponse(status_code=200, parsed=[])),
    )
    result = post_chat_json(
        url="http://unit.test",
        payload={"messages": []},
        timeout_s=1.0,
        api_key=None,
    )
    assert result.ok is False
    assert result.status_code == 200
    assert result.error == "invalid_response_type"


def test_post_chat_json_status_error(monkeypatch):
    class _StatusErr(Exception):
        def __init__(self):
            self.status_code = 502
            self.response = type("_Resp", (), {"text": "proxy down"})()
            super().__init__("bad gateway")

    def _factory(**kwargs):  # noqa: ARG001
        raise _StatusErr()

    monkeypatch.setattr(
        remote_api,
        "_get_openai_client",
        lambda *, base_url, api_key: _Client(_factory),
    )
    result = post_chat_json(
        url="http://unit.test",
        payload={"messages": []},
        timeout_s=1.0,
        api_key=None,
    )
    assert result.ok is False
    assert result.status_code == 502
    assert result.error == "http_502:proxy down"


def test_post_chat_json_transport_error(monkeypatch):
    def _factory(**kwargs):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(
        remote_api,
        "_get_openai_client",
        lambda *, base_url, api_key: _Client(_factory),
    )
    result = post_chat_json(
        url="http://unit.test",
        payload={"messages": []},
        timeout_s=1.0,
        api_key=None,
    )
    assert result.ok is False
    assert result.status_code == 0
    assert result.error is not None
    assert "transport:RuntimeError:boom" in result.error
