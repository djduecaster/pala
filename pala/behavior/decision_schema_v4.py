from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import ValidationError, validate

from .json_parse import parse_json_flexible

_MODES = (
    "boot_awaken",
    "idle_presence",
    "social_interact",
    "search_assist",
    "task_lighting",
    "return_home",
    "recover_reset",
)
_MOODS = ("calm", "curious", "excited", "focused", "neutral")
_SKILLS = (
    "wake_sequence",
    "observe_settle",
    "idle_presence",
    "idle_scan",
    "greet_user",
    "social_ack",
    "expressive_search",
    "point_and_hold",
    "task_light_adjust",
    "return_home",
    "recover_hold",
)
_PRIMITIVES = ("hold", "home", "breath", "glance", "nod", "orient_to_zone")
_STYLES = ("calm", "curious", "focused")
_MODE_TRANSITIONS = ("stay",) + tuple(f"to_{mode}" for mode in _MODES)


BEHAVIOR_DECISION_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "pala.behavior_decision.v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "mode",
        "mood",
        "skill",
        "action",
        "confidence",
        "rationale_short",
        "mode_transition",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "pala.behavior_decision.v1"},
        "mode": {"type": "string", "enum": list(_MODES)},
        "mood": {"type": "string", "enum": list(_MOODS)},
        "skill": {"type": "string", "enum": list(_SKILLS)},
        "action": {
            "type": "object",
            "additionalProperties": False,
            "required": ["primitive", "command", "style"],
            "properties": {
                "primitive": {"type": "string", "enum": list(_PRIMITIVES)},
                "command": {"type": "object"},
                "style": {"type": "string", "enum": list(_STYLES)},
            },
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale_short": {"type": "string", "minLength": 1, "maxLength": 220},
        "mode_transition": {"type": "string", "enum": list(_MODE_TRANSITIONS)},
        "alternatives": {
            "type": "array",
            "minItems": 0,
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["skill", "primitive", "rationale_short"],
                "properties": {
                    "skill": {"type": "string", "enum": list(_SKILLS)},
                    "primitive": {"type": "string", "enum": list(_PRIMITIVES)},
                    "rationale_short": {"type": "string", "minLength": 1, "maxLength": 120},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class BehaviorActionDecision:
    primitive: str
    command: Dict[str, Any]
    style: str


@dataclass(frozen=True)
class BehaviorDecisionAlternative:
    skill: str
    primitive: str
    rationale_short: str


@dataclass(frozen=True)
class BehaviorDecision:
    schema_version: str
    mode: str
    mood: str
    skill: str
    action: BehaviorActionDecision
    confidence: float
    rationale_short: str
    mode_transition: str
    alternatives: List[BehaviorDecisionAlternative] = field(default_factory=list)


@dataclass(frozen=True)
class BehaviorDecisionParseResult:
    decision: BehaviorDecision
    raw_text: str
    parse_stage: str = "raw"


class BehaviorDecisionParser:
    def __init__(self) -> None:
        self._last_parse_error: Optional[str] = None
        self._last_parse_stage: str = "raw"

    @property
    def last_parse_error(self) -> Optional[str]:
        return self._last_parse_error

    @property
    def last_parse_stage(self) -> str:
        return self._last_parse_stage

    def parse(self, raw_text: str) -> Optional[BehaviorDecisionParseResult]:
        parsed, err, stage = _parse_behavior_decision_response_with_error(raw_text)
        self._last_parse_error = err
        self._last_parse_stage = stage
        return parsed


def behavior_decision_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pala_behavior_decision_v1",
            "strict": True,
            "schema": BEHAVIOR_DECISION_SCHEMA,
        },
    }


def parse_behavior_decision_response(raw_text: str) -> Optional[BehaviorDecisionParseResult]:
    parsed, _, _ = _parse_behavior_decision_response_with_error(raw_text)
    return parsed


def _parse_behavior_decision_response_with_error(
    raw_text: str,
) -> Tuple[Optional[BehaviorDecisionParseResult], Optional[str], str]:
    token = str(raw_text or "").strip()
    if not token:
        return None, "empty_response", "raw"

    data, err, stage = parse_json_flexible(token)
    if data is None:
        return None, err or "json_decode:unknown", stage

    payload = _canonicalize_payload(data)
    if payload is None:
        return None, "payload_not_object", stage
    _normalize_payload_tokens(payload)

    try:
        validate(instance=payload, schema=BEHAVIOR_DECISION_SCHEMA)
    except ValidationError as exc:
        return None, f"schema:{_json_path(exc)}:{_short_text(exc.message, max_len=160)}", stage

    action_raw = payload.get("action")
    if not isinstance(action_raw, dict):
        return None, "action_not_object", stage

    action = BehaviorActionDecision(
        primitive=str(action_raw.get("primitive", "")).strip().lower(),
        command=dict(action_raw.get("command", {})),
        style=str(action_raw.get("style", "")).strip().lower(),
    )
    decision = BehaviorDecision(
        schema_version=str(payload.get("schema_version", "pala.behavior_decision.v1")),
        mode=str(payload.get("mode", "")).strip().lower(),
        mood=str(payload.get("mood", "")).strip().lower(),
        skill=str(payload.get("skill", "")).strip().lower(),
        action=action,
        confidence=_clamp01(payload.get("confidence"), default=0.0),
        rationale_short=_clean_text(payload.get("rationale_short"), max_len=220),
        mode_transition=str(payload.get("mode_transition", "stay")).strip().lower(),
        alternatives=_parse_alternatives(payload.get("alternatives")),
    )
    return BehaviorDecisionParseResult(decision=decision, raw_text=token, parse_stage=stage), None, stage


def _parse_alternatives(value: Any) -> List[BehaviorDecisionAlternative]:
    if not isinstance(value, list):
        return []
    out: List[BehaviorDecisionAlternative] = []
    for item in value[:2]:
        if not isinstance(item, dict):
            continue
        out.append(
            BehaviorDecisionAlternative(
                skill=str(item.get("skill", "")).strip().lower(),
                primitive=str(item.get("primitive", "")).strip().lower(),
                rationale_short=_clean_text(item.get("rationale_short"), max_len=120),
            )
        )
    return out


def _canonicalize_payload(data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    if "pala.behavior_decision.v1" in data and isinstance(data.get("pala.behavior_decision.v1"), dict):
        wrapped = dict(data.get("pala.behavior_decision.v1"))
        wrapped.setdefault("schema_version", "pala.behavior_decision.v1")
        return wrapped
    return dict(data)


def _normalize_payload_tokens(payload: Dict[str, Any]) -> None:
    for key in ("schema_version", "mode", "mood", "skill", "mode_transition"):
        if key not in payload:
            continue
        if key == "schema_version":
            payload[key] = _normalize_schema_version(payload.get(key))
        else:
            payload[key] = _norm_token(payload.get(key))

    action = payload.get("action")
    if isinstance(action, dict):
        if "primitive" in action:
            action["primitive"] = _norm_token(action.get("primitive"))
        if "style" in action:
            action["style"] = _norm_token(action.get("style"))

    alternatives = payload.get("alternatives")
    if isinstance(alternatives, list):
        for item in alternatives:
            if not isinstance(item, dict):
                continue
            if "skill" in item:
                item["skill"] = _norm_token(item.get("skill"))
            if "primitive" in item:
                item["primitive"] = _norm_token(item.get("primitive"))


def _norm_token(value: Any) -> str:
    token = str(value or "").strip().lower().replace(" ", "_")
    return token


def _normalize_schema_version(value: Any) -> str:
    token = str(value or "").strip().lower()
    compact = token.replace(" ", "").replace("-", "_")
    aliases = {
        "pala.behavior_decision.v1",
        "pala_behavior_decision_v1",
        "pala/behavior_decision/v1",
        "pala:behavior_decision:v1",
    }
    if compact in aliases:
        return "pala.behavior_decision.v1"
    return token


def _clean_text(value: Any, *, max_len: int) -> str:
    token = " ".join(str(value or "").split()).strip()
    return token[:max_len]


def _clamp01(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _short_text(value: Any, *, max_len: int) -> str:
    token = " ".join(str(value or "").split()).strip()
    if len(token) <= max_len:
        return token
    return token[: max_len - 3] + "..."


def _json_path(exc: ValidationError) -> str:
    if not exc.absolute_path:
        return "$(root)"
    parts = ["$"]
    for part in exc.absolute_path:
        if isinstance(part, int):
            parts.append(f"[{part}]")
        else:
            parts.append(f".{part}")
    return "".join(parts)
