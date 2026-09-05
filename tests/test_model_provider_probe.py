from __future__ import annotations

from tools.model_provider_probe import _json_schema_payload, _safe_json_parse, _text_payload


def test_provider_probe_builds_generic_text_payload() -> None:
    payload = _text_payload("test-model", max_tokens=32)
    assert payload["model"] == "test-model"
    assert payload["messages"][0]["role"] == "system"
    assert "READY" in payload["messages"][1]["content"]


def test_provider_probe_builds_provider_compatible_json_payload() -> None:
    gemini = _json_schema_payload("gemini-model", max_tokens=64, strict=True, provider="gemini")
    assert gemini["response_format"] == {"type": "json_object"}

    generic = _json_schema_payload("generic-model", max_tokens=64, strict=True, provider="openai")
    assert generic["response_format"]["type"] == "json_schema"
    assert generic["response_format"]["json_schema"]["strict"] is True


def test_provider_probe_accepts_only_json_objects() -> None:
    assert _safe_json_parse('{"ok":true}')[0] is True
    assert _safe_json_parse("[]")[0] is False
    assert _safe_json_parse("not json")[0] is False
