from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, Mapping, Optional

from pala.behavior.decision_schema_v4 import BehaviorDecision, BehaviorDecisionParser

_ALLOWED_STATUS = {"unlabeled", "labeled", "discarded"}
_ALLOWED_QUALITY = {"usable", "discard"}
_SCENARIO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


@dataclass(frozen=True)
class LabelRecord:
    status: str
    quality_flag: str
    expected_decision: Optional[BehaviorDecision]
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
        "expected_decision": None,
        "rationale_text": "",
        "annotator": "",
        "notes": "",
        "updated_at_utc": utc_now_iso(),
    }


def scenario_id_is_valid(value: str) -> bool:
    return bool(_SCENARIO_ID_RE.match(str(value or "").strip()))


def validate_expected_decision(value: Mapping[str, Any]) -> BehaviorDecision:
    if not isinstance(value, Mapping):
        raise ValueError("expected_decision must be an object")

    parser = BehaviorDecisionParser()
    raw = json.dumps(dict(value), ensure_ascii=True, separators=(",", ":"))
    parsed = parser.parse(raw)
    if parsed is None:
        detail = parser.last_parse_error or "unknown_error"
        stage = parser.last_parse_stage
        raise ValueError(f"expected_decision invalid at parse stage '{stage}': {detail}")
    return parsed.decision


def parse_expected_decision_json(text: str) -> BehaviorDecision:
    token = str(text or "").strip()
    if not token:
        raise ValueError("expected_decision JSON is required")
    try:
        parsed = json.loads(token)
    except json.JSONDecodeError as exc:
        raise ValueError(f"expected_decision JSON parse error: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("expected_decision JSON must decode to an object")
    return validate_expected_decision(parsed)


def validate_label_record(
    value: Mapping[str, Any],
    *,
    require_expected_decision_for_labeled: bool = True,
) -> LabelRecord:
    if not isinstance(value, Mapping):
        raise ValueError("label must be an object")
    if "expected_action" in value:
        raise ValueError("label.expected_action is deprecated; use label.expected_decision")

    status = str(value.get("status", "unlabeled")).strip().lower()
    if status not in _ALLOWED_STATUS:
        raise ValueError(f"label.status must be one of {sorted(_ALLOWED_STATUS)}")

    quality_flag = str(value.get("quality_flag", "usable")).strip().lower()
    if quality_flag not in _ALLOWED_QUALITY:
        raise ValueError(f"label.quality_flag must be one of {sorted(_ALLOWED_QUALITY)}")

    expected_decision: Optional[BehaviorDecision] = None
    expected_raw = value.get("expected_decision")
    if expected_raw is not None:
        if not isinstance(expected_raw, Mapping):
            raise ValueError("label.expected_decision must be null or an object")
        expected_decision = validate_expected_decision(expected_raw)

    rationale_text = " ".join(str(value.get("rationale_text", "")).split()).strip()
    annotator = str(value.get("annotator", "")).strip()
    notes = str(value.get("notes", "")).strip()
    updated = str(value.get("updated_at_utc", "")).strip() or utc_now_iso()

    if status == "labeled":
        if require_expected_decision_for_labeled and expected_decision is None:
            raise ValueError("labeled records require expected_decision")
        if not rationale_text:
            raise ValueError("labeled records require rationale_text")

    return LabelRecord(
        status=status,
        quality_flag=quality_flag,
        expected_decision=expected_decision,
        rationale_text=rationale_text,
        annotator=annotator,
        notes=notes,
        updated_at_utc=updated,
    )


def decision_to_dict(decision: BehaviorDecision) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": decision.schema_version,
        "mode": decision.mode,
        "mood": decision.mood,
        "skill": decision.skill,
        "action": {
            "primitive": decision.action.primitive,
            "command": dict(decision.action.command),
            "style": decision.action.style,
        },
        "confidence": float(decision.confidence),
        "rationale_short": decision.rationale_short,
        "mode_transition": decision.mode_transition,
    }
    if decision.alternatives:
        payload["alternatives"] = [
            {
                "skill": item.skill,
                "primitive": item.primitive,
                "rationale_short": item.rationale_short,
            }
            for item in decision.alternatives
        ]
    return payload


def label_record_to_dict(record: LabelRecord) -> Dict[str, Any]:
    expected_payload: Optional[Dict[str, Any]] = None
    if record.expected_decision is not None:
        expected_payload = decision_to_dict(record.expected_decision)
    return {
        "status": record.status,
        "quality_flag": record.quality_flag,
        "expected_decision": expected_payload,
        "rationale_text": record.rationale_text,
        "annotator": record.annotator,
        "notes": record.notes,
        "updated_at_utc": record.updated_at_utc,
    }


def normalize_label_payload(value: Mapping[str, Any]) -> Dict[str, Any]:
    record = validate_label_record(value)
    return label_record_to_dict(record)
