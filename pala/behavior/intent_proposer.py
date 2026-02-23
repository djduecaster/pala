from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from jsonschema import ValidationError, validate

from .json_parse import parse_json_flexible
from .schemas import INTENT_PROPOSALS_SCHEMA
from .types import IntentProposal, ProposerResponse, clamp01, clamp_float, clamp_int

_ALLOWED_INTENTS = {
    "idle_presence",
    "acknowledge_presence",
    "track_user",
    "scan_environment",
    "react_to_change",
    "reset_pose",
    "affirmation",
}
_ALLOWED_PRIMITIVES = {"hold", "home", "breath", "glance", "nod", "orient_to_zone"}
_ALLOWED_STYLES = {"calm", "curious", "focused"}
_ALLOWED_RISKS = {"low", "medium", "high"}
_ALLOWED_DIRECTIONS = {"left", "right", "up", "down"}
_ALLOWED_ZONES = {"left", "center", "right"}


@dataclass
class IntentProposerParseResult:
    response: ProposerResponse
    raw_text: str
    parse_stage: str = "raw"


class IntentProposer:
    """Latest-only async request bookkeeping for the remote intent proposer."""

    def __init__(self) -> None:
        self._inflight = False
        self._pending_payload: Optional[Mapping[str, Any]] = None
        self._last_parse_error: Optional[str] = None
        self._last_parse_stage: str = "raw"

    @property
    def in_flight(self) -> bool:
        return self._inflight

    @property
    def last_parse_error(self) -> Optional[str]:
        return self._last_parse_error

    @property
    def last_parse_stage(self) -> str:
        return self._last_parse_stage

    def submit_or_replace(self, payload: Mapping[str, Any]) -> bool:
        if not self._inflight:
            self._inflight = True
            self._pending_payload = None
            return True
        self._pending_payload = dict(payload)
        return False

    def mark_pending(self, payload: Mapping[str, Any]) -> None:
        self._pending_payload = dict(payload)

    def complete_request(self, raw_text: str) -> Optional[IntentProposerParseResult]:
        self._inflight = False
        parsed, err, stage = _parse_intent_proposer_response_with_error(raw_text)
        self._last_parse_error = err
        self._last_parse_stage = stage
        return parsed

    def take_latest_pending(self) -> Optional[Mapping[str, Any]]:
        payload = self._pending_payload
        self._pending_payload = None
        return payload


def parse_intent_proposer_response(
    raw_text: str,
) -> Optional[IntentProposerParseResult]:
    parsed, _, _ = _parse_intent_proposer_response_with_error(raw_text)
    return parsed


def _parse_intent_proposer_response_with_error(
    raw_text: str,
) -> tuple[Optional[IntentProposerParseResult], Optional[str], str]:
    token = str(raw_text or "").strip()
    if not token:
        return None, "empty_response", "raw"
    data, parse_error, stage = parse_json_flexible(token)
    if data is None:
        return None, parse_error or "json_decode:unknown", stage
    canonical = _canonicalize_payload(data)
    if canonical is None:
        return None, "proposals_missing_or_empty", stage
    schema_error: Optional[str] = None
    try:
        validate(instance=canonical, schema=INTENT_PROPOSALS_SCHEMA)
    except ValidationError as exc:
        schema_error = f"schema:{_json_path(exc)}:{_short_text(exc.message, max_len=160)}"

    schema_version = str(canonical.get("schema_version", "pala.intent_proposals.v1")).strip()
    raw_proposals = canonical.get("proposals")

    if not isinstance(raw_proposals, list) or not raw_proposals:
        return None, "proposals_missing_or_empty", stage

    parsed = []
    invalid_indices = []
    for idx, item in enumerate(raw_proposals):
        proposal = _parse_proposal(item)
        if proposal is None:
            invalid_indices.append(idx)
            continue
        parsed.append(proposal)

    if invalid_indices:
        if schema_error is not None:
            return None, schema_error, stage
        return None, f"proposal_invalid:indices={invalid_indices}", stage

    if not parsed:
        if schema_error is not None:
            return None, schema_error, stage
        return None, "proposals_missing_or_empty", stage

    if not any(_is_non_idle_proposal(item) for item in parsed):
        return None, "non_idle_proposal_missing", stage

    notes_short = _clean_text(canonical.get("notes_short"))
    response = ProposerResponse(schema_version=schema_version, proposals=parsed, notes_short=notes_short)
    return IntentProposerParseResult(response=response, raw_text=token, parse_stage=stage), None, stage


def _parse_proposal(item: Any) -> Optional[IntentProposal]:
    if not isinstance(item, dict):
        return None

    required = (
        "intent",
        "primitive",
        "command",
        "style",
        "score",
        "confidence",
        "urgency",
        "risk",
        "allow_interrupt",
        "evidence",
        "rationale_short",
    )
    if any(key not in item for key in required):
        return None

    intent = str(item.get("intent", "")).strip().lower().replace(" ", "_")
    primitive = str(item.get("primitive", "")).strip().lower().replace(" ", "_")
    if primitive in {"idle_presence", "idle", "none"}:
        primitive = "breath"
    style = str(item.get("style", item.get("_style_", ""))).strip().lower()
    risk = _normalize_risk(item.get("risk", "low"))

    if intent not in _ALLOWED_INTENTS:
        return None
    if primitive not in _ALLOWED_PRIMITIVES:
        return None
    if style not in _ALLOWED_STYLES:
        style = "calm"

    command_raw = item.get("command", {})
    if not isinstance(command_raw, dict):
        return None
    command = _normalize_command(primitive, command_raw)
    if command is None:
        return None

    evidence = item.get("evidence")
    evidence_out = []
    if isinstance(evidence, str):
        text = " ".join(evidence.split()).strip()
        if text:
            evidence_out.append(text[:64])
    elif isinstance(evidence, list):
        for token in evidence[:8]:
            text = " ".join(str(token).split()).strip()
            if text:
                evidence_out.append(text[:64])

    rationale_short = _clean_text(item.get("rationale_short"))
    if not rationale_short:
        return None

    allow_interrupt = bool(item.get("allow_interrupt"))

    min_dwell_raw = item.get("min_dwell_ms")
    max_duration_raw = item.get("max_duration_ms")
    min_dwell_ms = clamp_int(min_dwell_raw, lo=0, hi=30000, default=0) if min_dwell_raw is not None else None
    max_duration_ms = (
        clamp_int(max_duration_raw, lo=0, hi=60000, default=0) if max_duration_raw is not None else None
    )

    score = clamp01(item.get("score"), default=0.0)
    confidence = clamp01(item.get("confidence"), default=0.0)
    urgency = clamp01(item.get("urgency"), default=0.0)

    return IntentProposal(
        intent=intent,
        primitive=primitive,
        command=command,
        style=style,
        score=score,
        confidence=confidence,
        urgency=urgency,
        risk=risk,
        allow_interrupt=allow_interrupt,
        min_dwell_ms=min_dwell_ms,
        max_duration_ms=max_duration_ms,
        evidence=evidence_out,
        rationale_short=rationale_short,
    )


def _normalize_command(primitive: str, command_raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cmd: Dict[str, Any] = {}

    if primitive == "hold":
        return {}

    if primitive == "home":
        cmd["rate_rad_s"] = clamp_float(command_raw.get("rate_rad_s", 1.5), lo=0.2, hi=5.0, default=1.5)
        return cmd

    if primitive == "breath":
        cmd["amp_rad"] = clamp_float(command_raw.get("amp_rad", 0.08), lo=0.0, hi=0.35, default=0.08)
        cmd["period_s"] = clamp_float(command_raw.get("period_s", 7.0), lo=1.5, hi=20.0, default=7.0)
        cmd["rate_rad_s"] = clamp_float(command_raw.get("rate_rad_s", 1.0), lo=0.2, hi=5.0, default=1.0)
        return cmd

    if primitive == "glance":
        direction_raw = command_raw.get("direction")
        direction = str(direction_raw).strip().lower() if direction_raw is not None else "left"
        if direction not in _ALLOWED_DIRECTIONS:
            return None
        cmd["direction"] = direction
        cmd["amp_rad"] = clamp_float(command_raw.get("amp_rad", 0.35), lo=0.0, hi=0.8, default=0.35)
        cmd["duration_s"] = clamp_float(command_raw.get("duration_s", 0.6), lo=0.1, hi=2.0, default=0.6)
        cmd["rate_rad_s"] = clamp_float(command_raw.get("rate_rad_s", 1.8), lo=0.2, hi=5.0, default=1.8)
        return cmd

    if primitive == "nod":
        cmd["amp_rad"] = clamp_float(command_raw.get("amp_rad", 0.2), lo=0.0, hi=0.6, default=0.2)
        cmd["duration_s"] = clamp_float(command_raw.get("duration_s", 0.4), lo=0.1, hi=2.0, default=0.4)
        cmd["cycles"] = clamp_int(command_raw.get("cycles", 1), lo=1, hi=3, default=1)
        cmd["rate_rad_s"] = clamp_float(command_raw.get("rate_rad_s", 1.8), lo=0.2, hi=5.0, default=1.8)
        return cmd

    if primitive == "orient_to_zone":
        zone_raw = command_raw.get("zone")
        zone = str(zone_raw if zone_raw is not None else "").strip().lower()
        if zone not in _ALLOWED_ZONES:
            return None
        cmd["zone"] = zone
        cmd["amp_rad"] = clamp_float(command_raw.get("amp_rad", 0.25), lo=0.0, hi=0.8, default=0.25)
        cmd["rate_rad_s"] = clamp_float(command_raw.get("rate_rad_s", 1.4), lo=0.2, hi=5.0, default=1.4)
        return cmd

    return None


def _clean_text(value: Any) -> str:
    token = " ".join(str(value or "").split()).strip()
    return token[:220]


def _normalize_risk(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
        if score <= 0.33:
            return "low"
        if score <= 0.66:
            return "medium"
        return "high"
    token = str(value or "low").strip().lower()
    if token in _ALLOWED_RISKS:
        return token
    if token in {"minimal", "safe"}:
        return "low"
    return "medium"


def _short_text(value: Any, *, max_len: int) -> str:
    token = " ".join(str(value or "").split()).strip()
    if len(token) <= max_len:
        return token
    return token[: max_len - 3] + "..."


def _json_path(exc: ValidationError) -> str:
    if not exc.absolute_path:
        return "$"
    parts = ["$"]
    for part in exc.absolute_path:
        if isinstance(part, int):
            parts.append(f"[{part}]")
        else:
            parts.append(f".{part}")
    return "".join(parts)


def _looks_like_single_proposal(data: Mapping[str, Any]) -> bool:
    return "intent" in data and "primitive" in data


def _canonicalize_payload(data: Any) -> Optional[Dict[str, Any]]:
    schema_version = "pala.intent_proposals.v1"
    notes_short = ""
    raw_proposals: Any = None

    if isinstance(data, dict):
        notes_short = _clean_text(data.get("notes_short"))
        raw_proposals = data.get("proposals")
        if raw_proposals is None:
            raw_proposals = data.get("intent_proposals")
        if raw_proposals is None and "pala.intent_proposals.v1" in data:
            wrapped = data.get("pala.intent_proposals.v1")
            if isinstance(wrapped, dict):
                notes_short = _clean_text(wrapped.get("notes_short") or notes_short)
                raw_proposals = wrapped.get("proposals")
            else:
                raw_proposals = wrapped
        if raw_proposals is None and isinstance(data.get("proposal"), dict):
            raw_proposals = [data["proposal"]]
        if raw_proposals is None and _looks_like_single_proposal(data):
            raw_proposals = [data]
    elif isinstance(data, list):
        raw_proposals = data

    if not isinstance(raw_proposals, list):
        return None
    normalized: list[Any] = []
    for item in raw_proposals:
        normalized.append(_canonicalize_proposal_item(item))
    return {
        "schema_version": schema_version,
        "notes_short": notes_short,
        "proposals": normalized,
    }


def _canonicalize_proposal_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item

    intent = str(item.get("intent", "")).strip().lower().replace(" ", "_")
    primitive = str(item.get("primitive", "")).strip().lower().replace(" ", "_")
    primitive = _normalize_primitive_alias(primitive, intent=intent)
    intent = _normalize_intent_alias(intent=intent, primitive=primitive)

    command_raw = item.get("command")
    if not isinstance(command_raw, dict):
        command_raw = {}
    command = _normalize_command(primitive, command_raw)
    if command is None:
        # Preserve invalid payload shape so schema validation can report the exact violation.
        command = dict(command_raw)

    style_raw = item.get("style")
    style: Optional[str] = None
    if style_raw is not None:
        token = str(style_raw).strip().lower()
        style = token if token in _ALLOWED_STYLES else "calm"
    risk_raw = item.get("risk")
    risk: Optional[str] = _normalize_risk(risk_raw) if risk_raw is not None else None

    evidence_raw = item.get("evidence", [])
    evidence: list[str] = []
    if isinstance(evidence_raw, str):
        token = " ".join(evidence_raw.split()).strip()
        if token:
            evidence.append(token[:64])
    elif isinstance(evidence_raw, list):
        for token in evidence_raw[:8]:
            norm = " ".join(str(token).split()).strip()
            if norm:
                evidence.append(norm[:64])

    rationale = _clean_text(item.get("rationale_short"))

    out: Dict[str, Any] = {
        "intent": intent,
        "primitive": primitive,
        "command": command,
    }
    if style is not None:
        out["style"] = style
    if risk is not None:
        out["risk"] = risk
    if "allow_interrupt" in item:
        out["allow_interrupt"] = bool(item.get("allow_interrupt"))
    if "evidence" in item:
        out["evidence"] = evidence
    if rationale:
        out["rationale_short"] = rationale
    if "score" in item:
        out["score"] = clamp01(item.get("score"), default=0.0)
    if "confidence" in item:
        out["confidence"] = clamp01(item.get("confidence"), default=0.0)
    if "urgency" in item:
        out["urgency"] = clamp01(item.get("urgency"), default=0.0)

    min_dwell_raw = item.get("min_dwell_ms")
    max_duration_raw = item.get("max_duration_ms")
    if min_dwell_raw is not None:
        out["min_dwell_ms"] = clamp_int(min_dwell_raw, lo=0, hi=30000, default=0)
    if max_duration_raw is not None:
        out["max_duration_ms"] = clamp_int(max_duration_raw, lo=0, hi=60000, default=0)

    return out


def _normalize_intent_alias(*, intent: str, primitive: str) -> str:
    if intent in _ALLOWED_INTENTS:
        return intent
    fallback = {
        "hold": "idle_presence",
        "home": "reset_pose",
        "breath": "idle_presence",
        "glance": "acknowledge_presence",
        "nod": "affirmation",
        "orient_to_zone": "track_user",
    }
    return fallback.get(primitive, "idle_presence")


def _normalize_primitive_alias(primitive: str, *, intent: str) -> str:
    if primitive in _ALLOWED_PRIMITIVES:
        return primitive
    if primitive in {"idle_presence", "idle", "none"}:
        return "breath"
    if primitive in {"reset_pose"}:
        return "home"
    if primitive in {"track_user"}:
        return "orient_to_zone"
    if intent == "reset_pose":
        return "home"
    return primitive


def _is_non_idle_proposal(item: IntentProposal) -> bool:
    if item.primitive in {"hold", "breath"}:
        return False
    if item.intent == "idle_presence":
        return False
    return True
