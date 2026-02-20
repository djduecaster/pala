from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ReasoningEvent:
    source: str
    ts_wall_s: Optional[float]
    req_id: Optional[int]
    phase: str
    status: str
    latency_ms: Optional[float]
    primitive: Optional[str]
    confidence: Optional[float]
    target_zone: Optional[str]
    model: Optional[str]
    provider: Optional[str]
    snippet: str
    severity: str


_SENSITIVE_PAIR_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|authorization)\b\s*[:=]\s*([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._\-+/=]+)")
_QUERY_SECRET_RE = re.compile(r"(?i)([?&](?:api[_-]?key|token|secret|key)=)([^&\s]+)")
_LONG_BLOB_RE = re.compile(r"\b[A-Za-z0-9._\-+/=]{40,}\b")


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pick_first(values: list[Any], fn) -> Any:
    for item in values:
        parsed = fn(item)
        if parsed is not None:
            return parsed
    return None


def _classify_severity(phase: str, status: str) -> str:
    text = f"{phase} {status}".lower()
    if any(key in text for key in ("fail", "error", "invalid", "timeout", "stale", "no_content", "crash")):
        return "error"
    if any(key in text for key in ("warn", "drop", "retry")):
        return "warning"
    return "info"


def _extract_snippet(data: Dict[str, Any], payload: Dict[str, Any], msg_payload: Dict[str, Any]) -> str:
    for key in ("reasoning", "rationale", "explanation", "detail", "error", "preview", "message", "text"):
        text = _as_optional_str(data.get(key))
        if text:
            return text
    for key in ("reasoning", "rationale", "explanation", "detail", "error", "message", "text"):
        text = _as_optional_str(payload.get(key))
        if text:
            return text
    for key in ("line", "error"):
        text = _as_optional_str(msg_payload.get(key))
        if text:
            return text
    return ""


def normalize_reasoning_message(msg: Dict[str, Any]) -> Optional[ReasoningEvent]:
    source = _as_optional_str(msg.get("source"))
    if not source:
        return None
    ts_wall_s = _as_optional_float(msg.get("ts_wall_s"))
    msg_payload = _as_dict(msg.get("payload"))

    if source == "timeline_log":
        data = _as_dict(msg_payload.get("data"))
        if not data:
            return None
        data_payload = _as_dict(data.get("payload"))
        phase = _as_optional_str(data.get("type")) or "timeline"
        status = _pick_first(
            [data_payload.get("status"), data.get("status"), data_payload.get("result")],
            _as_optional_str,
        ) or ""
        return ReasoningEvent(
            source=source,
            ts_wall_s=ts_wall_s,
            req_id=_pick_first(
                [
                    data_payload.get("request_id"),
                    data_payload.get("req_id"),
                    data_payload.get("id"),
                    data.get("request_id"),
                    data.get("req_id"),
                    data.get("id"),
                ],
                _as_optional_int,
            ),
            phase=phase,
            status=status,
            latency_ms=_pick_first(
                [data_payload.get("latency_ms"), data.get("latency_ms")],
                _as_optional_float,
            ),
            primitive=_pick_first(
                [data_payload.get("primitive"), data_payload.get("primitive_hint"), data_payload.get("active_primitive")],
                _as_optional_str,
            ),
            confidence=_pick_first(
                [data_payload.get("confidence"), data.get("confidence")],
                _as_optional_float,
            ),
            target_zone=_pick_first(
                [data_payload.get("target_zone"), data_payload.get("zone")],
                _as_optional_str,
            ),
            model=_pick_first([data_payload.get("model"), data.get("model")], _as_optional_str),
            provider=_pick_first([data_payload.get("provider"), data.get("provider")], _as_optional_str),
            snippet=_extract_snippet(data_payload, data, msg_payload),
            severity=_classify_severity(phase, status),
        )

    if source == "actions_log":
        data = _as_dict(msg_payload.get("data"))
        if not data:
            return None
        return ReasoningEvent(
            source=source,
            ts_wall_s=ts_wall_s,
            req_id=_pick_first([data.get("request_id"), data.get("req_id")], _as_optional_int),
            phase="action_plan",
            status="ok",
            latency_ms=None,
            primitive=_as_optional_str(data.get("primitive")),
            confidence=_as_optional_float(data.get("confidence")),
            target_zone=_as_optional_str(data.get("target_zone")),
            model=None,
            provider=None,
            snippet=_extract_snippet(data, data, msg_payload),
            severity="info",
        )

    if source == "agent":
        detail = _as_optional_str(msg_payload.get("error"))
        if not detail:
            return None
        return ReasoningEvent(
            source=source,
            ts_wall_s=ts_wall_s,
            req_id=None,
            phase="agent_error",
            status="error",
            latency_ms=None,
            primitive=None,
            confidence=None,
            target_zone=None,
            model=None,
            provider=None,
            snippet=detail,
            severity="error",
        )

    return None


def redact_reasoning_text(text: str) -> str:
    out = str(text)
    out = _SENSITIVE_PAIR_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", out)
    out = _BEARER_RE.sub("Bearer [REDACTED]", out)
    out = _QUERY_SECRET_RE.sub(lambda m: f"{m.group(1)}[REDACTED]", out)
    out = _LONG_BLOB_RE.sub("[REDACTED_BLOB]", out)
    return out


def format_reasoning_snippet(text: str, *, max_chars: int, redact: bool) -> str:
    out = " ".join(str(text).strip().split())
    if not out:
        return ""
    if redact:
        out = redact_reasoning_text(out)
    limit = max(24, int(max_chars))
    if len(out) <= limit:
        return out
    return out[: limit - 3] + "..."
