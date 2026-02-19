from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple


_EXPR_RE = re.compile(
    r"^(?P<source>[a-zA-Z0-9_]+)\.(?P<path>[a-zA-Z0-9_.]+)\s*(?P<op>!=|=|>|<|~)\s*(?P<value>.+)$"
)


@dataclass(frozen=True)
class FieldFilter:
    source: str
    path: Tuple[str, ...]
    op: str
    raw_value: str
    value: Any
    regex: Optional[re.Pattern[str]] = None


def _coerce_scalar(text: str) -> Any:
    raw = text.strip()
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if "." in raw or "e" in lowered:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_field_filter(expr: str) -> FieldFilter:
    text = str(expr or "").strip()
    match = _EXPR_RE.match(text)
    if not match:
        raise ValueError(
            f"invalid field filter '{text}'. expected source.path<op>value with op in =,!=,<,>,~"
        )

    source = match.group("source")
    path = tuple(part for part in match.group("path").split(".") if part)
    op = match.group("op")
    raw_value = match.group("value").strip()
    if not path:
        raise ValueError(f"invalid field filter '{text}': empty path")
    if not raw_value:
        raise ValueError(f"invalid field filter '{text}': empty value")

    if op == "~":
        try:
            compiled = re.compile(raw_value)
        except re.error as exc:
            raise ValueError(f"invalid regex in field filter '{text}': {exc}") from exc
        return FieldFilter(
            source=source,
            path=path,
            op=op,
            raw_value=raw_value,
            value=raw_value,
            regex=compiled,
        )

    return FieldFilter(
        source=source,
        path=path,
        op=op,
        raw_value=raw_value,
        value=_coerce_scalar(raw_value),
    )


def parse_field_filters(expressions: Sequence[str] | None) -> List[FieldFilter]:
    return [parse_field_filter(expr) for expr in (expressions or []) if str(expr).strip()]


def _lookup_payload_value(payload: Mapping[str, Any], path: Tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping):
            return False, None
        if part not in current:
            return False, None
        current = current[part]
    return True, current


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _match_filter(msg: Mapping[str, Any], flt: FieldFilter) -> bool:
    payload = msg.get("payload")
    if not isinstance(payload, Mapping):
        return False
    ok, current = _lookup_payload_value(payload, flt.path)
    if not ok:
        return False

    if flt.op == "=":
        return current == flt.value
    if flt.op == "!=":
        return current != flt.value
    if flt.op in {">", "<"}:
        lhs = _numeric(current)
        rhs = _numeric(flt.value)
        if lhs is None or rhs is None:
            return False
        if flt.op == ">":
            return lhs > rhs
        return lhs < rhs
    if flt.op == "~":
        if flt.regex is None:
            return False
        return flt.regex.search(str(current)) is not None
    return False


def matches_field_filters(msg: Mapping[str, Any], filters: Sequence[FieldFilter]) -> bool:
    if not filters:
        return True
    source = msg.get("source")
    if not isinstance(source, str):
        return False
    scoped = [flt for flt in filters if flt.source == source]
    if not scoped:
        return True
    return all(_match_filter(msg, flt) for flt in scoped)
