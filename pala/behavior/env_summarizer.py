from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Mapping, Optional

from jsonschema import ValidationError, validate

from .json_parse import parse_json_flexible
from .schemas import ENV_SUMMARY_SCHEMA
from .types import EnvSummary, clamp01


@dataclass
class EnvSummarizerParseResult:
    summary: EnvSummary
    raw_text: str
    parse_stage: str = "raw"


class EnvSummarizer:
    """Latest-only async request bookkeeping for the remote env summarizer."""

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

    def complete_request(self, raw_text: str) -> Optional[EnvSummarizerParseResult]:
        self._inflight = False
        parsed, err, stage = _parse_env_summary_response_with_error(raw_text)
        self._last_parse_error = err
        self._last_parse_stage = stage
        return parsed

    def take_latest_pending(self) -> Optional[Mapping[str, Any]]:
        payload = self._pending_payload
        self._pending_payload = None
        return payload


def parse_env_summary_response(raw_text: str) -> Optional[EnvSummarizerParseResult]:
    parsed, _, _ = _parse_env_summary_response_with_error(raw_text)
    return parsed


def _parse_env_summary_response_with_error(
    raw_text: str,
) -> tuple[Optional[EnvSummarizerParseResult], Optional[str], str]:
    token = str(raw_text or "").strip()
    if not token:
        return None, "empty_response", "raw"

    data, parse_error, stage = parse_json_flexible(token)
    if data is None:
        return None, parse_error or "json_decode:unknown", stage

    canonical = _canonicalize_env_payload(data)
    if canonical is None:
        return None, "json_root_not_object", stage

    try:
        validate(instance=canonical, schema=ENV_SUMMARY_SCHEMA)
    except ValidationError as exc:
        return None, f"schema:{_json_path(exc)}:{_short_text(exc.message, max_len=160)}", stage

    parsed = EnvSummary(
        scene=str(canonical["scene"]),
        events=str(canonical["events"]),
        hypotheses=str(canonical["hypotheses"]),
        summary_short=str(canonical["summary_short"]),
        delta_score=clamp01(canonical["delta_score"], default=0.0),
        features=dict(canonical["features"]),
    )
    return EnvSummarizerParseResult(summary=parsed, raw_text=token, parse_stage=stage), None, stage


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

    scene = _clean_text(_coerce_text(scene_raw, prefer=("scene", "description", "summary", "caption")), max_len=360)
    events = _clean_text(_coerce_text(events_raw, prefer=("events", "description", "summary", "caption")), max_len=220)
    hypotheses = _clean_text(
        _coerce_text(hypotheses_raw, prefer=("hypotheses", "inference", "reasoning", "summary")),
        max_len=220,
    )
    summary_short = _clean_text(_coerce_text(summary_raw, prefer=("summary", "caption", "text")), max_len=120)

    if not scene:
        scene = _clean_text(summary_short or events, max_len=360)
    if not events:
        events = _clean_text(summary_short or scene, max_len=220)
    if not hypotheses:
        hypotheses = "user intent uncertain from available evidence"
    if not summary_short:
        summary_short = _clean_text(events or scene, max_len=120)

    features_raw = root.get("features")
    if not isinstance(features_raw, dict) and isinstance(scene_raw, dict):
        features_raw = scene_raw
    if not isinstance(features_raw, dict):
        features_raw = {}

    zone_hint = str(features_raw.get("zone_hint", "unknown")).strip().lower()
    if zone_hint not in {"left", "center", "right", "unknown"}:
        zone_hint = "unknown"
    if zone_hint == "unknown":
        zone_hint = _infer_zone_hint_from_text(scene, events, summary_short, hypotheses)

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


def _infer_zone_hint_from_text(*texts: str) -> str:
    token = " ".join(_clean_text(item, max_len=240) for item in texts if item).lower()
    if not token:
        return "unknown"

    zone_patterns = {
        "left": (
            r"\bto my left\b",
            r"\bon my left\b",
            r"\bleft side\b",
            r"\bleft\b",
        ),
        "right": (
            r"\bto my right\b",
            r"\bon my right\b",
            r"\bright side\b",
            r"\bright\b",
        ),
        "center": (
            r"\bin front of me\b",
            r"\bahead of me\b",
            r"\bcenter\b",
            r"\bmiddle\b",
        ),
    }

    earliest: Dict[str, int] = {}
    for zone, patterns in zone_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, token)
            if match is None:
                continue
            at = int(match.start())
            previous = earliest.get(zone)
            if previous is None or at < previous:
                earliest[zone] = at

    if not earliest:
        return "unknown"
    return min(earliest.items(), key=lambda item: item[1])[0]
