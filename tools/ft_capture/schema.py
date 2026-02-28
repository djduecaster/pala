from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, Mapping, Optional

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
_ALLOWED_STATUS = {"unlabeled", "labeled", "discarded"}
_ALLOWED_QUALITY = {"usable", "discard"}
_SCENARIO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


@dataclass(frozen=True)
class ExpectedAction:
    intent: str
    primitive: str
    command: Dict[str, Any]
    style: str
    confidence: float


@dataclass(frozen=True)
class LabelRecord:
    status: str
    quality_flag: str
    expected_action: Optional[ExpectedAction]
    rationale_text: str
    annotator: str
    notes: str
    updated_at_utc: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_label_record() -> Dict[str, Any]:
    return {
        "status": "unlabeled",
        "quality_flag": "usable",
        "expected_action": None,
        "rationale_text": "",
        "annotator": "",
        "notes": "",
        "updated_at_utc": utc_now_iso(),
    }


def scenario_id_is_valid(value: str) -> bool:
    return bool(_SCENARIO_ID_RE.match(str(value or "").strip()))


def validate_expected_action(value: Mapping[str, Any]) -> ExpectedAction:
    if not isinstance(value, Mapping):
        raise ValueError("expected_action must be an object")

    intent = str(value.get("intent", "")).strip().lower()
    if intent not in _ALLOWED_INTENTS:
        raise ValueError(f"expected_action.intent must be one of {sorted(_ALLOWED_INTENTS)}")

    primitive = str(value.get("primitive", "")).strip().lower()
    if primitive not in _ALLOWED_PRIMITIVES:
        raise ValueError(f"expected_action.primitive must be one of {sorted(_ALLOWED_PRIMITIVES)}")

    command_raw = value.get("command")
    if not isinstance(command_raw, Mapping):
        raise ValueError("expected_action.command must be an object")
    command = {str(k): command_raw[k] for k in command_raw.keys()}

    style = str(value.get("style", "")).strip().lower()
    if style not in _ALLOWED_STYLES:
        raise ValueError(f"expected_action.style must be one of {sorted(_ALLOWED_STYLES)}")

    confidence_raw = value.get("confidence")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        raise ValueError("expected_action.confidence must be a number in [0,1]") from None
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("expected_action.confidence must be in [0,1]")

    return ExpectedAction(
        intent=intent,
        primitive=primitive,
        command=command,
        style=style,
        confidence=confidence,
    )


def parse_expected_action_json(text: str) -> ExpectedAction:
    token = str(text or "").strip()
    if not token:
        raise ValueError("expected_action JSON is required")
    try:
        parsed = json.loads(token)
    except json.JSONDecodeError as exc:
        raise ValueError(f"expected_action JSON parse error: {exc}") from exc
    return validate_expected_action(parsed)


def validate_label_record(
    value: Mapping[str, Any],
    *,
    require_expected_action_for_labeled: bool = True,
) -> LabelRecord:
    if not isinstance(value, Mapping):
        raise ValueError("label must be an object")

    status = str(value.get("status", "unlabeled")).strip().lower()
    if status not in _ALLOWED_STATUS:
        raise ValueError(f"label.status must be one of {sorted(_ALLOWED_STATUS)}")

    quality_flag = str(value.get("quality_flag", "usable")).strip().lower()
    if quality_flag not in _ALLOWED_QUALITY:
        raise ValueError(f"label.quality_flag must be one of {sorted(_ALLOWED_QUALITY)}")

    expected_action: Optional[ExpectedAction] = None
    expected_raw = value.get("expected_action")
    if expected_raw is not None:
        if not isinstance(expected_raw, Mapping):
            raise ValueError("label.expected_action must be null or an object")
        expected_action = validate_expected_action(expected_raw)

    rationale_text = " ".join(str(value.get("rationale_text", "")).split()).strip()
    annotator = str(value.get("annotator", "")).strip()
    notes = str(value.get("notes", "")).strip()
    updated = str(value.get("updated_at_utc", "")).strip() or utc_now_iso()

    if status == "labeled":
        if require_expected_action_for_labeled and expected_action is None:
            raise ValueError("labeled records require expected_action")
        if not rationale_text:
            raise ValueError("labeled records require rationale_text")

    return LabelRecord(
        status=status,
        quality_flag=quality_flag,
        expected_action=expected_action,
        rationale_text=rationale_text,
        annotator=annotator,
        notes=notes,
        updated_at_utc=updated,
    )


def label_record_to_dict(record: LabelRecord) -> Dict[str, Any]:
    expected_payload: Optional[Dict[str, Any]] = None
    if record.expected_action is not None:
        expected_payload = {
            "intent": record.expected_action.intent,
            "primitive": record.expected_action.primitive,
            "command": dict(record.expected_action.command),
            "style": record.expected_action.style,
            "confidence": float(record.expected_action.confidence),
        }
    return {
        "status": record.status,
        "quality_flag": record.quality_flag,
        "expected_action": expected_payload,
        "rationale_text": record.rationale_text,
        "annotator": record.annotator,
        "notes": record.notes,
        "updated_at_utc": record.updated_at_utc,
    }


def normalize_label_payload(value: Mapping[str, Any]) -> Dict[str, Any]:
    record = validate_label_record(value)
    return label_record_to_dict(record)
