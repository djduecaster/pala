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
    component: Optional[str] = None
    delta_score: Optional[float] = None
    mode: Optional[str] = None
    active_skill: Optional[str] = None
    guard_accepted: Optional[bool] = None
    guard_fallback: Optional[bool] = None
    guard_reason: Optional[str] = None
    guard_skill: Optional[str] = None
    guard_primitive: Optional[str] = None
    planner_enabled: Optional[bool] = None
    planner_inflight: Optional[bool] = None
    planner_pending: Optional[bool] = None
    planner_last_parse_stage: Optional[str] = None
    planner_error: Optional[str] = None
    planner_next_allowed_in_s: Optional[float] = None
    mode_transition_from: Optional[str] = None
    mode_transition_to: Optional[str] = None
    mode_transition_reason: Optional[str] = None
    mode_transitioned: Optional[bool] = None


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


def _as_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


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
    if any(key in text for key in ("fail", "error", "invalid", "timeout", "stale", "no_content", "crash", "fallback", "reject")):
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


def _decision_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    raw = data.get("decision_json")
    if isinstance(raw, dict):
        return raw
    return {}


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
            component="timeline",
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
            component="action",
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
            component="agent",
        )

    if source in {"behavior_env_log", "behavior_env"}:
        data = _as_dict(msg_payload.get("data"))
        if not data:
            data = msg_payload
        if not data:
            return None
        phase = _pick_first([data.get("phase"), data.get("stage"), data.get("module")], _as_optional_str) or "env_processor"
        status = _pick_first([data.get("status"), data.get("parse_status"), data.get("result")], _as_optional_str) or "ok"
        return ReasoningEvent(
            source=source,
            ts_wall_s=ts_wall_s,
            req_id=_pick_first([data.get("request_id"), data.get("req_id"), data.get("id")], _as_optional_int),
            phase=phase,
            status=status,
            latency_ms=_pick_first([data.get("latency_ms"), data.get("duration_ms")], _as_optional_float),
            primitive=None,
            confidence=None,
            target_zone=_pick_first([data.get("target_zone"), data.get("zone_hint")], _as_optional_str),
            model=_pick_first([data.get("model"), data.get("model_name")], _as_optional_str),
            provider=_pick_first([data.get("provider"), data.get("vendor")], _as_optional_str),
            snippet=_extract_snippet(data, data, msg_payload),
            severity=_classify_severity(phase, status),
            component="env_processor",
            delta_score=_as_optional_float(data.get("delta_score")),
            mode=_as_optional_str(data.get("mode")),
            active_skill=_as_optional_str(data.get("active_skill")),
        )

    if source in {"behavior_planner_log", "behavior_planner"}:
        data = _as_dict(msg_payload.get("data"))
        if not data:
            data = msg_payload
        if not data:
            return None
        decision = _decision_payload(data)
        phase = _pick_first([data.get("phase"), data.get("stage"), data.get("module")], _as_optional_str) or "planner_v4"
        status = _pick_first([data.get("status"), data.get("parse_status"), data.get("result")], _as_optional_str) or "ok"
        command = _as_dict(decision.get("command"))
        planner_error = _pick_first([data.get("error"), data.get("planner_error")], _as_optional_str)
        parse_stage = _pick_first([data.get("parse_stage"), data.get("last_parse_stage")], _as_optional_str)
        return ReasoningEvent(
            source=source,
            ts_wall_s=ts_wall_s,
            req_id=_pick_first([data.get("request_id"), data.get("req_id"), data.get("id")], _as_optional_int),
            phase=phase,
            status=status,
            latency_ms=_pick_first([data.get("latency_ms"), data.get("duration_ms")], _as_optional_float),
            primitive=_pick_first([data.get("primitive"), decision.get("primitive")], _as_optional_str),
            confidence=_pick_first([data.get("confidence"), decision.get("confidence")], _as_optional_float),
            target_zone=_pick_first([data.get("target_zone"), command.get("target_zone"), command.get("zone")], _as_optional_str),
            model=_pick_first([data.get("model"), data.get("model_name")], _as_optional_str),
            provider=_pick_first([data.get("provider"), data.get("vendor")], _as_optional_str),
            snippet=_extract_snippet(decision, data, msg_payload),
            severity=_classify_severity(phase, f"{status} {planner_error or ''} {parse_stage or ''}"),
            component=_pick_first([data.get("component"), data.get("module")], _as_optional_str) or "planner_v4",
            mode=_as_optional_str(data.get("mode")),
            active_skill=_as_optional_str(data.get("skill")),
            planner_last_parse_stage=parse_stage,
            planner_error=planner_error,
        )

    if source in {"behavior_reasoning_log", "behavior_reasoning"}:
        data = _as_dict(msg_payload.get("data"))
        if not data:
            data = msg_payload
        if not data:
            return None
        phase = _pick_first([data.get("phase"), data.get("module"), data.get("stage")], _as_optional_str) or "behavior_reasoning"
        status = _pick_first([data.get("status"), data.get("result")], _as_optional_str) or "ok"
        return ReasoningEvent(
            source=source,
            ts_wall_s=ts_wall_s,
            req_id=_pick_first([data.get("request_id"), data.get("req_id"), data.get("id")], _as_optional_int),
            phase=phase,
            status=status,
            latency_ms=_pick_first([data.get("latency_ms"), data.get("duration_ms")], _as_optional_float),
            primitive=_as_optional_str(data.get("primitive")),
            confidence=_as_optional_float(data.get("confidence")),
            target_zone=_pick_first([data.get("target_zone"), data.get("zone_hint")], _as_optional_str),
            model=_pick_first([data.get("model"), data.get("model_name")], _as_optional_str),
            provider=_pick_first([data.get("provider"), data.get("vendor")], _as_optional_str),
            snippet=_extract_snippet(data, data, msg_payload),
            severity=_classify_severity(phase, status),
            component=_pick_first([data.get("component"), data.get("module")], _as_optional_str) or "planner_v4",
            mode=_as_optional_str(data.get("mode")),
            active_skill=_pick_first([data.get("skill"), data.get("active_skill")], _as_optional_str),
        )

    if source in {"behavior_trace_log", "behavior_trace"}:
        data = _as_dict(msg_payload.get("data"))
        if not data:
            data = msg_payload
        if not data:
            return None
        planner = _as_dict(data.get("planner"))
        guard = _as_dict(data.get("guard"))
        mode_transition = _as_dict(data.get("mode_transition"))
        current_action = _as_dict(data.get("current_action"))

        mode = _as_optional_str(data.get("mode"))
        active_skill = _pick_first([data.get("active_skill"), data.get("skill")], _as_optional_str)
        planner_error = _pick_first([planner.get("last_error"), data.get("planner_error"), data.get("error")], _as_optional_str)
        parse_stage = _pick_first([planner.get("last_parse_stage"), data.get("parse_stage")], _as_optional_str)
        guard_reason = _pick_first([guard.get("reason"), data.get("guard_reason")], _as_optional_str)
        guard_accepted = _as_optional_bool(guard.get("accepted"))
        guard_fallback = _as_optional_bool(guard.get("fallback"))

        if guard_accepted is True:
            status = "committed"
        elif guard_fallback is True:
            status = "fallback"
        elif planner_error:
            status = "planner_error"
        elif parse_stage and any(tok in parse_stage.lower() for tok in ("fail", "invalid", "error")):
            status = "parse_fail"
        else:
            status = _as_optional_str(data.get("status")) or "trace"

        phase = _pick_first([mode_transition.get("to"), data.get("mode"), data.get("phase")], _as_optional_str) or "behavior_trace"

        snippet = guard_reason or planner_error or _as_optional_str(mode_transition.get("reason")) or ""
        if not snippet:
            snippet = _extract_snippet(data, planner, msg_payload)
        if not snippet:
            prim_text = _as_optional_str(current_action.get("primitive")) or "-"
            snippet = f"mode={mode or '-'} skill={active_skill or '-'} action={prim_text}"

        return ReasoningEvent(
            source=source,
            ts_wall_s=ts_wall_s,
            req_id=_pick_first([data.get("request_id"), data.get("req_id"), data.get("id")], _as_optional_int),
            phase=phase,
            status=status,
            latency_ms=_pick_first(
                [planner.get("last_latency_ms"), data.get("latency_ms"), data.get("duration_ms")],
                _as_optional_float,
            ),
            primitive=_pick_first([guard.get("primitive"), current_action.get("primitive")], _as_optional_str),
            confidence=_as_optional_float(current_action.get("confidence")),
            target_zone=_pick_first([data.get("target_zone"), data.get("zone_hint")], _as_optional_str),
            model=None,
            provider=None,
            snippet=snippet,
            severity=_classify_severity(phase, f"{status} {guard_reason or ''} {planner_error or ''}"),
            component="arbiter",
            mode=mode,
            active_skill=active_skill,
            guard_accepted=guard_accepted,
            guard_fallback=guard_fallback,
            guard_reason=guard_reason,
            guard_skill=_as_optional_str(guard.get("skill")),
            guard_primitive=_as_optional_str(guard.get("primitive")),
            planner_enabled=_as_optional_bool(planner.get("enabled")),
            planner_inflight=_as_optional_bool(planner.get("inflight")),
            planner_pending=_as_optional_bool(planner.get("pending")),
            planner_last_parse_stage=parse_stage,
            planner_error=planner_error,
            planner_next_allowed_in_s=_as_optional_float(planner.get("next_allowed_in_s")),
            mode_transition_from=_as_optional_str(mode_transition.get("from")),
            mode_transition_to=_as_optional_str(mode_transition.get("to")),
            mode_transition_reason=_as_optional_str(mode_transition.get("reason")),
            mode_transitioned=_as_optional_bool(mode_transition.get("transitioned")),
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
