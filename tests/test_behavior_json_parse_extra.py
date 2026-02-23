from __future__ import annotations

from pala.behavior import json_parse


def test_parse_json_flexible_empty_and_coercion_paths():
    data, err, stage = json_parse.parse_json_flexible(None)
    assert data is None
    assert err == "empty_response"
    assert stage == "raw"

    data, err, stage = json_parse.parse_json_flexible("   ")
    assert data is None
    assert err == "empty_response"
    assert stage == "raw"

    class _JsonText:
        def __str__(self) -> str:
            return '{"coerced": true}'

    data, err, stage = json_parse.parse_json_flexible(_JsonText())
    assert err is None
    assert stage == "raw"
    assert data == {"coerced": True}


def test_parse_json_flexible_defence_error_then_extracted_error():
    data, err, stage = json_parse.parse_json_flexible("<answer>```json\n{bad}\n```</answer>")
    assert data is None
    assert stage == "extracted"
    assert (err or "").startswith("json_decode:")


def test_try_json_loads_reports_decode_error_shape():
    data, err = json_parse._try_json_loads("{bad}")
    assert data is None
    assert (err or "").startswith("json_decode:")
    assert "@" in (err or "")


def test_strip_common_wrappers():
    text = "  <answer>\n```json\n{\"ok\": true}\n```\n</answer> "
    assert json_parse._strip_common_wrappers(text) == '{"ok": true}'


def test_extract_first_json_value_branches():
    assert json_parse._extract_first_json_value("no json here") is None

    balanced = json_parse._extract_first_json_value('prefix {"k":"v[1]","arr":[1,2]} suffix')
    assert balanced == '{"k":"v[1]","arr":[1,2]}'

    mismatched = json_parse._extract_first_json_value('prefix {"a":[1,2} tail')
    assert mismatched is None

    unbalanced = json_parse._extract_first_json_value("prefix {\"a\": [1, 2] tail")
    assert unbalanced is None
