from __future__ import annotations

import io
import json
from urllib import error as urllib_error

from pala.behavior.remote_api import extract_message_content, normalize_chat_url, post_chat_json


class _Resp:
    def __init__(self, *, status: int, body: str):
        self.status = status
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


def test_normalize_chat_url_variants():
    assert normalize_chat_url("") == ""
    assert normalize_chat_url("  ") == ""
    assert normalize_chat_url("http://x") == "http://x/v1/chat/completions"
    assert normalize_chat_url("http://x/") == "http://x/v1/chat/completions"
    assert normalize_chat_url("http://x/v1") == "http://x/v1/chat/completions"
    assert normalize_chat_url("http://x/v1/") == "http://x/v1/chat/completions"
    assert normalize_chat_url("http://x/v1/chat/completions") == "http://x/v1/chat/completions"


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


def test_post_chat_json_success(monkeypatch):
    captured = {}

    def _urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = req.data.decode("utf-8")
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _Resp(status=200, body=json.dumps({"choices": [{"message": {"content": "ok"}}]}))

    monkeypatch.setattr("pala.behavior.remote_api.urllib_request.urlopen", _urlopen)

    result = post_chat_json(
        url="http://unit.test/v1/chat/completions",
        payload={"model": "x", "messages": []},
        timeout_s=1.25,
        api_key="abc",
    )
    assert result.ok is True
    assert result.status_code == 200
    assert result.response_json is not None
    assert result.error is None
    assert captured["url"] == "http://unit.test/v1/chat/completions"
    assert captured["timeout"] == 1.25
    assert captured["headers"]["authorization"] == "Bearer abc"
    assert captured["headers"]["content-type"] == "application/json"
    assert json.loads(captured["body"])["model"] == "x"


def test_post_chat_json_invalid_json_response(monkeypatch):
    monkeypatch.setattr(
        "pala.behavior.remote_api.urllib_request.urlopen",
        lambda req, timeout: _Resp(status=200, body="not-json"),  # noqa: ARG005
    )
    result = post_chat_json(
        url="http://unit.test",
        payload={"messages": []},
        timeout_s=1.0,
        api_key=None,
    )
    assert result.ok is False
    assert result.error == "invalid_json_response"


def test_post_chat_json_rejects_non_object_json(monkeypatch):
    monkeypatch.setattr(
        "pala.behavior.remote_api.urllib_request.urlopen",
        lambda req, timeout: _Resp(status=200, body="[]"),  # noqa: ARG005
    )
    result = post_chat_json(
        url="http://unit.test",
        payload={"messages": []},
        timeout_s=1.0,
        api_key=None,
    )
    assert result.ok is False
    assert result.error == "invalid_response_type"


def test_post_chat_json_http_error(monkeypatch):
    def _raise_http(req, timeout):  # noqa: ARG001
        raise urllib_error.HTTPError(
            url="http://unit.test",
            code=502,
            msg="bad gateway",
            hdrs=None,
            fp=io.BytesIO(b"proxy down"),
        )

    monkeypatch.setattr("pala.behavior.remote_api.urllib_request.urlopen", _raise_http)
    result = post_chat_json(
        url="http://unit.test",
        payload={"messages": []},
        timeout_s=1.0,
        api_key=None,
    )
    assert result.ok is False
    assert result.status_code == 502
    assert result.error is not None
    assert result.error.startswith("http_502:proxy down")


def test_post_chat_json_url_error(monkeypatch):
    def _raise_url(req, timeout):  # noqa: ARG001
        raise urllib_error.URLError("unreachable")

    monkeypatch.setattr("pala.behavior.remote_api.urllib_request.urlopen", _raise_url)
    result = post_chat_json(
        url="http://unit.test",
        payload={"messages": []},
        timeout_s=1.0,
        api_key=None,
    )
    assert result.ok is False
    assert result.status_code == 0
    assert result.error is not None
    assert result.error.startswith("transport:unreachable")


def test_post_chat_json_generic_exception(monkeypatch):
    def _raise(req, timeout):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr("pala.behavior.remote_api.urllib_request.urlopen", _raise)
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
