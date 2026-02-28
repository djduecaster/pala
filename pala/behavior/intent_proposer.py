from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from jsonschema import ValidationError, validate

from .json_parse import parse_json_flexible
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
_INTENT_PROPOSALS_ENVELOPE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "proposals"],
    "properties": {
        "schema_version": {"type": "string", "const": "pala.intent_proposals.v2"},
        "notes_short": {"type": "string", "maxLength": 280},
        "proposals": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            # Validate proposals individually below so one malformed entry
            # does not drop the whole response.
            "items": {},
        },
    },
}


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


def parse_intent_proposer_response(raw_text: str) -> Optional[IntentProposerParseResult]:
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

    try:
        validate(instance=canonical, schema=_INTENT_PROPOSALS_ENVELOPE_SCHEMA)
    except ValidationError as exc:
        return None, f"schema:{_json_path(exc)}:{_short_text(exc.message, max_len=160)}", stage

    raw_proposals = canonical.get("proposals")
    if not isinstance(raw_proposals, list) or not raw_proposals:
        return None, "proposals_missing_or_empty", stage

    parsed: list[IntentProposal] = []
    for item in raw_proposals:
        proposal = _parse_proposal(item)
        if proposal is None:
            continue
        parsed.append(proposal)

    if not parsed:
        return None, "all_proposals_invalid", stage

    notes_short = _clean_text(canonical.get("notes_short"))
    response = ProposerResponse(
        schema_version=str(canonical.get("schema_version", "pala.intent_proposals.v2")),
        proposals=parsed,
        notes_short=notes_short,
    )
    return IntentProposerParseResult(response=response, raw_text=token, parse_stage=stage), None, stage


def _parse_proposal(item: Any) -> Optional[IntentProposal]:
    if not isinstance(item, dict):
        return None

    intent = str(item.get("intent", "")).strip().lower().replace(" ", "_")
    primitive = str(item.get("primitive", "")).strip().lower().replace(" ", "_")
    style = str(item.get("style", "")).strip().lower()
    risk = str(item.get("risk", "")).strip().lower()

    if intent not in _ALLOWED_INTENTS:
        return None
    if primitive not in _ALLOWED_PRIMITIVES:
        return None
    if style not in _ALLOWED_STYLES:
        return None
    if risk not in _ALLOWED_RISKS:
        return None

    command_raw = item.get("command")
    if not isinstance(command_raw, dict):
        return None
    command = _normalize_command(primitive, command_raw)
    if command is None:
        return None

    evidence_raw = item.get("evidence")
    if not isinstance(evidence_raw, list):
        return None
    evidence: list[str] = []
    for token in evidence_raw[:8]:
        norm = " ".join(str(token).split()).strip()
        if norm:
            evidence.append(norm[:64])

    rationale_short = _clean_text(item.get("rationale_short"))
    if not rationale_short:
        return None

    allow_interrupt = item.get("allow_interrupt")
    if not isinstance(allow_interrupt, bool):
        return None

    min_dwell_raw = item.get("min_dwell_ms")
    max_duration_raw = item.get("max_duration_ms")
    min_dwell_ms = clamp_int(min_dwell_raw, lo=0, hi=30000, default=0) if min_dwell_raw is not None else None
    max_duration_ms = (
        clamp_int(max_duration_raw, lo=50, hi=60000, default=2000) if max_duration_raw is not None else None
    )

    return IntentProposal(
        intent=intent,
        primitive=primitive,
        command=command,
        style=style,
        score=clamp01(item.get("score"), default=0.0),
        confidence=clamp01(item.get("confidence"), default=0.0),
        urgency=clamp01(item.get("urgency"), default=0.0),
        risk=risk,
        allow_interrupt=allow_interrupt,
        min_dwell_ms=min_dwell_ms,
        max_duration_ms=max_duration_ms,
        evidence=evidence,
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
        direction = str(direction_raw).strip().lower() if direction_raw is not None else ""
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


def _canonicalize_payload(data: Any) -> Optional[Dict[str, Any]]:
    if isinstance(data, list):
        return {
            "schema_version": "pala.intent_proposals.v2",
            "notes_short": "",
            "proposals": data,
        }

    if not isinstance(data, dict):
        return None

    if "pala.intent_proposals.v2" in data:
        wrapped = data.get("pala.intent_proposals.v2")
        if isinstance(wrapped, dict):
            payload = dict(wrapped)
            payload.setdefault("schema_version", "pala.intent_proposals.v2")
            return payload
        if isinstance(wrapped, list):
            return {
                "schema_version": "pala.intent_proposals.v2",
                "notes_short": "",
                "proposals": wrapped,
            }

    payload = dict(data)
    if "proposals" not in payload and isinstance(payload.get("intent_proposals"), list):
        payload["proposals"] = payload.get("intent_proposals")
    payload.pop("intent_proposals", None)
    if "schema_version" not in payload:
        payload["schema_version"] = "pala.intent_proposals.v2"
    payload["schema_version"] = _normalize_schema_version(payload.get("schema_version"))
    payload.pop("pala.intent_proposals.v2", None)
    return payload


def _normalize_schema_version(raw: Any) -> str:
    token = str(raw or "").strip().lower()
    if token in {"pala.intent_proposals.v2", "v2", "2", "2.0", "2.0.0"}:
        return "pala.intent_proposals.v2"
    return "pala.intent_proposals.v2"
