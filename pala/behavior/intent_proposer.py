from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Mapping, Optional

from jsonschema import ValidationError, validate

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


class IntentProposer:
    """Latest-only async request bookkeeping for the remote intent proposer."""

    def __init__(self) -> None:
        self._inflight = False
        self._pending_payload: Optional[Mapping[str, Any]] = None
        self._last_parse_error: Optional[str] = None

    @property
    def in_flight(self) -> bool:
        return self._inflight

    @property
    def last_parse_error(self) -> Optional[str]:
        return self._last_parse_error

    def submit_or_replace(self, payload: Mapping[str, Any]) -> bool:
        if not self._inflight:
            self._inflight = True
            self._pending_payload = None
            return True
        self._pending_payload = dict(payload)
        return False

    def complete_request(self, raw_text: str) -> Optional[IntentProposerParseResult]:
        self._inflight = False
        parsed, err = _parse_intent_proposer_response_with_error(raw_text)
        self._last_parse_error = err
        return parsed

    def take_latest_pending(self) -> Optional[Mapping[str, Any]]:
        payload = self._pending_payload
        self._pending_payload = None
        return payload


def parse_intent_proposer_response(
    raw_text: str,
) -> Optional[IntentProposerParseResult]:
    parsed, _ = _parse_intent_proposer_response_with_error(raw_text)
    return parsed


def _parse_intent_proposer_response_with_error(
    raw_text: str,
) -> tuple[Optional[IntentProposerParseResult], Optional[str]]:
    token = str(raw_text or "").strip()
    if not token:
        return None, "empty_response"
    data, parse_error = _parse_json_flexible(token)
    if data is None:
        return None, parse_error or "json_decode:unknown"
    canonical = _canonicalize_payload(data)
    if canonical is None:
        return None, "proposals_missing_or_empty"
    schema_error: Optional[str] = None
    try:
        validate(instance=canonical, schema=INTENT_PROPOSALS_SCHEMA)
    except ValidationError as exc:
        schema_error = f"schema:{_json_path(exc)}:{_short_text(exc.message, max_len=160)}"

    schema_version = str(canonical.get("schema_version", "pala.intent_proposals.v1")).strip()
    raw_proposals = canonical.get("proposals")

    if not isinstance(raw_proposals, list) or not raw_proposals:
        return None, "proposals_missing_or_empty"

    parsed = []
    invalid_indices = []
    for idx, item in enumerate(raw_proposals):
        proposal = _parse_proposal(item)
        if proposal is None:
            invalid_indices.append(idx)
            continue
        parsed.append(proposal)

    if not parsed:
        return None, schema_error or f"proposal_invalid:indices={invalid_indices}"

    notes_short = _clean_text(canonical.get("notes_short"))
    response = ProposerResponse(schema_version=schema_version, proposals=parsed, notes_short=notes_short)
    # Keep schema validation strict but non-blocking for bring-up compatibility;
    # decision safety still flows through deterministic governor/arbiter checks.
    return IntentProposerParseResult(response=response, raw_text=token), None


def _parse_proposal(item: Any) -> Optional[IntentProposal]:
    if not isinstance(item, dict):
        return None

    intent = str(item.get("intent", "")).strip().lower().replace(" ", "_")
    primitive = str(item.get("primitive", "")).strip().lower().replace(" ", "_")
    if primitive in {"idle_presence", "idle", "none"}:
        primitive = "breath"
    style = str(item.get("style", item.get("_style_", "calm"))).strip().lower()
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

    rationale_short = _clean_text(
        item.get("rationale_short")
        or item.get("rationale")
        or item.get("notes_short")
        or f"{intent}:{primitive}"
    )
    if not rationale_short:
        return None

    allow_interrupt = bool(item.get("allow_interrupt", True))

    min_dwell_raw = item.get("min_dwell_ms")
    max_duration_raw = item.get("max_duration_ms")
    min_dwell_ms = clamp_int(min_dwell_raw, lo=0, hi=30000, default=0) if min_dwell_raw is not None else None
    max_duration_ms = (
        clamp_int(max_duration_raw, lo=0, hi=60000, default=0) if max_duration_raw is not None else None
    )

    score = clamp01(item.get("score", item.get("confidence", 0.35)), default=0.35)
    confidence = clamp01(item.get("confidence", item.get("score", 0.55)), default=0.55)
    urgency = clamp01(item.get("urgency", 0.2), default=0.2)

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
        direction = str(command_raw.get("direction", "left")).strip().lower()
        if direction not in _ALLOWED_DIRECTIONS:
            direction = "left"
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
        zone = str(command_raw.get("zone", "center")).strip().lower()
        if zone not in _ALLOWED_ZONES:
            zone = "center"
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


def _parse_json_flexible(raw_text: str) -> tuple[Optional[Any], Optional[str]]:
    first_error: Optional[str] = None
    for candidate in _json_candidates(raw_text):
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as exc:
            if first_error is None:
                first_error = f"json_decode:{exc.msg}@{exc.lineno}:{exc.colno}"
    return None, first_error or "json_decode:unknown"


def _json_candidates(raw_text: str) -> list[str]:
    token = raw_text.strip()
    out: list[str] = []
    if token:
        out.append(token)

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", token, flags=re.IGNORECASE | re.DOTALL):
        inner = match.group(1).strip()
        if inner:
            out.append(inner)

    first_obj = token.find("{")
    last_obj = token.rfind("}")
    if first_obj >= 0 and last_obj > first_obj:
        out.append(token[first_obj : last_obj + 1].strip())

    first_arr = token.find("[")
    last_arr = token.rfind("]")
    if first_arr >= 0 and last_arr > first_arr:
        out.append(token[first_arr : last_arr + 1].strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


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
    if primitive in {"idle_presence", "idle", "none"}:
        primitive = "breath"
    intent = _normalize_intent_alias(intent=intent, primitive=primitive)

    command = item.get("command")
    if not isinstance(command, dict):
        command = {}

    out: Dict[str, Any] = {
        "intent": intent,
        "primitive": primitive,
        "command": command,
    }

    for key in (
        "style",
        "score",
        "confidence",
        "urgency",
        "risk",
        "allow_interrupt",
        "min_dwell_ms",
        "max_duration_ms",
        "evidence",
        "rationale_short",
    ):
        if key in item:
            out[key] = item.get(key)

    if "rationale_short" not in out:
        rationale = item.get("rationale") or item.get("notes_short")
        if rationale is not None:
            out["rationale_short"] = rationale

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
