from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple

_OPEN_ANSWER_RE = re.compile(r"^\s*<answer>\s*", flags=re.IGNORECASE)
_CLOSE_ANSWER_RE = re.compile(r"\s*</answer>\s*$", flags=re.IGNORECASE)
_FENCE_HEAD_RE = re.compile(r"^\s*```(?:[a-zA-Z0-9_-]+)?\s*")
_FENCE_TAIL_RE = re.compile(r"\s*```\s*$")


def parse_json_flexible(raw_text: Any) -> Tuple[Optional[Any], Optional[str], str]:
    """
    Parse model output as JSON with deterministic, limited sanitization.

    Stages:
      - "raw": parsed without modification
      - "defenced": parsed after removing common wrappers
      - "extracted": parsed after extracting first balanced JSON value
    """
    text = _coerce_text(raw_text)
    if text is None:
        return None, "empty_response", "raw"
    token = text.strip()
    if not token:
        return None, "empty_response", "raw"

    data, err = _try_json_loads(token)
    if err is None:
        return data, None, "raw"

    sanitized = _strip_common_wrappers(token)
    if sanitized != token:
        data, err2 = _try_json_loads(sanitized)
        if err2 is None:
            return data, None, "defenced"
        err = err2

    candidate = _extract_first_json_value(sanitized)
    if candidate:
        data, err3 = _try_json_loads(candidate)
        if err3 is None:
            return data, None, "extracted"
        err = err3

    return None, err or "json_decode:unknown", "extracted"


def _coerce_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return str(value)


def _strip_common_wrappers(text: str) -> str:
    token = _OPEN_ANSWER_RE.sub("", text)
    token = _CLOSE_ANSWER_RE.sub("", token)
    token = _FENCE_HEAD_RE.sub("", token)
    token = _FENCE_TAIL_RE.sub("", token)
    return token.strip()


def _try_json_loads(text: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"json_decode:{exc.msg}@{exc.lineno}:{exc.colno}"


def _extract_first_json_value(text: str) -> Optional[str]:
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [idx for idx in (start_obj, start_arr) if idx >= 0]
    if not starts:
        return None
    start = min(starts)

    stack: list[str] = []
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in "{[":
            stack.append(ch)
            continue
        if ch in "}]":
            if not stack:
                return None
            left = stack.pop()
            if (left, ch) not in (("{", "}"), ("[", "]")):
                return None
            if not stack:
                return text[start : index + 1].strip()
    return None
