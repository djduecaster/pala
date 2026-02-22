from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Mapping, Optional

from jsonschema import ValidationError, validate

from .schemas import ENV_SUMMARY_SCHEMA
from .types import EnvSummary, clamp01


@dataclass
class EnvSummarizerParseResult:
    summary: EnvSummary
    raw_text: str


class EnvSummarizer:
    """Latest-only async request bookkeeping for the remote env summarizer."""

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

    def complete_request(self, raw_text: str) -> Optional[EnvSummarizerParseResult]:
        self._inflight = False
        parsed, err = _parse_env_summary_response_with_error(raw_text)
        self._last_parse_error = err
        return parsed

    def take_latest_pending(self) -> Optional[Mapping[str, Any]]:
        payload = self._pending_payload
        self._pending_payload = None
        return payload


def parse_env_summary_response(raw_text: str) -> Optional[EnvSummarizerParseResult]:
    parsed, _ = _parse_env_summary_response_with_error(raw_text)
    return parsed


def _parse_env_summary_response_with_error(
    raw_text: str,
) -> tuple[Optional[EnvSummarizerParseResult], Optional[str]]:
    token = str(raw_text or "").strip()
    if not token:
        return None, "empty_response"

    data, parse_error = _parse_json_flexible(token)
    if data is None:
        return None, parse_error or "json_decode:unknown"

    canonical = _canonicalize_env_payload(data)
    if canonical is None:
        return None, "json_root_not_object"

    try:
        validate(instance=canonical, schema=ENV_SUMMARY_SCHEMA)
    except ValidationError as exc:
        return None, f"schema:{_json_path(exc)}:{_short_text(exc.message, max_len=160)}"

    parsed = EnvSummary(
        scene=str(canonical["scene"]),
        events=str(canonical["events"]),
        hypotheses=str(canonical["hypotheses"]),
        summary_short=str(canonical["summary_short"]),
        delta_score=clamp01(canonical["delta_score"], default=0.0),
        features=dict(canonical["features"]),
    )
    return EnvSummarizerParseResult(summary=parsed, raw_text=token), None


def _canonicalize_env_payload(data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None

    root = data
    wrapped = data.get("pala.env_summary.v1")
    if isinstance(wrapped, dict):
        root = wrapped

    scene_raw = _pick_value(root, "scene", "description")
    events_raw = _pick_value(root, "events", "changes", "event_context")
    hypotheses_raw = _pick_value(root, "hypotheses", "inferences", "hypothesis")
    summary_raw = _pick_value(root, "summary_short", "summary", "caption")

    scene = _clean_text(_coerce_text(scene_raw, prefer=("scene", "description", "summary", "caption")), max_len=1200)
    events = _clean_text(_coerce_text(events_raw, prefer=("events", "description", "summary", "caption")), max_len=800)
    hypotheses = _clean_text(
        _coerce_text(hypotheses_raw, prefer=("hypotheses", "inference", "reasoning", "summary")),
        max_len=800,
    )
    summary_short = _clean_text(_coerce_text(summary_raw, prefer=("summary", "caption", "text")), max_len=180)

    if not scene:
        scene = _clean_text(summary_short or events, max_len=1200)
    if not events:
        events = _clean_text(summary_short or scene, max_len=800)
    if not hypotheses:
        hypotheses = "user intent uncertain from available evidence"
    if not summary_short:
        summary_short = _clean_text(events or scene, max_len=180)

    features_raw = root.get("features")
    if not isinstance(features_raw, dict) and isinstance(scene_raw, dict):
        features_raw = scene_raw
    if not isinstance(features_raw, dict):
        features_raw = {}

    zone_hint = str(features_raw.get("zone_hint", "unknown")).strip().lower()
    if zone_hint not in {"left", "center", "right", "unknown"}:
        zone_hint = "unknown"

    features = {
        "person_present": bool(features_raw.get("person_present", False)),
        "zone_hint": zone_hint,
        "activity_level": clamp01(features_raw.get("activity_level", 0.0), default=0.0),
        "novelty": clamp01(features_raw.get("novelty", 0.0), default=0.0),
    }

    delta_score = clamp01(root.get("delta_score", features["novelty"]), default=0.0)

    return {
        "schema_version": "pala.env_summary.v1",
        "scene": scene,
        "events": events,
        "hypotheses": hypotheses,
        "summary_short": summary_short,
        "delta_score": delta_score,
        "features": features,
    }


def _clean_text(value: Any, *, max_len: int) -> str:
    token = " ".join(str(value or "").split()).strip()
    return token[:max_len]


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

    for match in re.finditer(r"```(?:json)?\\s*(.*?)```", token, flags=re.IGNORECASE | re.DOTALL):
        inner = match.group(1).strip()
        if inner:
            out.append(inner)

    first_obj = token.find("{")
    last_obj = token.rfind("}")
    if first_obj >= 0 and last_obj > first_obj:
        out.append(token[first_obj : last_obj + 1].strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _pick_value(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data.get(key)
    return ""


def _coerce_text(value: Any, *, prefer: tuple[str, ...]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in prefer:
            if key in value:
                token = _coerce_text(value.get(key), prefer=prefer)
                if token:
                    return token
        parts = [_coerce_text(item, prefer=prefer) for item in value.values()]
        joined = " ".join(part for part in parts if part)
        return joined
    if isinstance(value, list):
        parts = [_coerce_text(item, prefer=prefer) for item in value]
        return " ".join(part for part in parts if part)
    return str(value)
