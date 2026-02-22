from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Mapping, Optional

from .world_state_store import EnvironmentSnapshot


_TAG_PATTERN = re.compile(r"<([a-z_]+)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_FENCE_PATTERN = re.compile(r"```(?:[a-z0-9_+-]+)?|```", re.IGNORECASE)
_TEMPLATE_ECHO_PATTERN = re.compile(
    r"<(?:scene|events|hypotheses)>\s*\.\.\.\s*</(?:scene|events|hypotheses)>",
    re.IGNORECASE,
)
_INSTRUCTION_PATTERN = re.compile(
    r"\b(optional tags?|required tags?|rules?|return exactly|do not output|output format)\b",
    re.IGNORECASE,
)
_ORDINAL_FRAME_PATTERN = re.compile(r"\b(first|second|third|fourth)\s+(?:image|frame)\b", re.IGNORECASE)


@dataclass
class EnvProcessorConfig:
    max_inflight: int = 1
    event_delta_threshold: float = 0.65


@dataclass
class EnvProcessorParseResult:
    snapshot: EnvironmentSnapshot
    reasoning_text: Optional[str]
    raw_text: str


class CosmosEnvProcessor:
    """
    Phase-0/1 environment processor scaffolding.

    This class manages latest-only in-flight bookkeeping and tagged text parsing.
    Transport/network dispatch is intentionally deferred to phase 2.
    """

    def __init__(self, config: Optional[EnvProcessorConfig] = None):
        self._cfg = config or EnvProcessorConfig()
        self._inflight = False
        self._pending_payload: Optional[Mapping[str, Any]] = None

    @property
    def in_flight(self) -> bool:
        return self._inflight

    def submit_or_replace(self, payload: Mapping[str, Any]) -> bool:
        if not self._inflight:
            self._inflight = True
            self._pending_payload = None
            return True
        self._pending_payload = dict(payload)
        return False

    def complete_request(self, raw_text: str) -> Optional[EnvProcessorParseResult]:
        self._inflight = False
        return parse_env_processor_response(raw_text)

    def take_latest_pending(self) -> Optional[Mapping[str, Any]]:
        payload = self._pending_payload
        self._pending_payload = None
        return payload


def parse_env_processor_response(raw_text: str) -> Optional[EnvProcessorParseResult]:
    strict_blocks = _extract_tag_blocks(raw_text)
    fuzzy_blocks = _extract_fuzzy_tag_blocks(
        raw_text,
        (
            "scene",
            "events",
            "hypotheses",
            "opportunities",
            "uncertainties",
            "delta_score",
            "summary",
            "think",
            "answer",
        ),
    )
    blocks = {**fuzzy_blocks, **strict_blocks}
    core = ("scene", "events", "hypotheses")
    reasoning = _coerce_reasoning(blocks)
    scene = _normalize_sequence_language(blocks.get("scene"), field="scene")
    events = _normalize_sequence_language(blocks.get("events"), field="events")
    hypotheses = _normalize_sequence_language(blocks.get("hypotheses"), field="hypotheses")
    opportunities = _normalize_text(blocks.get("opportunities"))
    uncertainties = _normalize_text(blocks.get("uncertainties"))
    summary = _normalize_sequence_language(blocks.get("summary"), field="summary")

    scene_ok = _is_meaningful_text(scene)
    events_ok = _is_meaningful_text(events)
    hypotheses_ok = _is_meaningful_text(hypotheses)
    summary_ok = _is_meaningful_text(summary)

    if any(tag in blocks for tag in core) and (scene_ok or events_ok or hypotheses_ok):
        if not scene_ok:
            scene = summary if summary_ok else (events if events_ok else "unspecified_scene")
        if not events_ok:
            events = summary if summary_ok else (scene if scene_ok else "unspecified_events")
        if not hypotheses_ok:
            hypotheses = "inferred from recent scene observations"
        if not summary_ok:
            summary = events or scene or hypotheses
        scene = _normalize_sequence_language(scene, field="scene")
        events = _normalize_sequence_language(events, field="events")
        hypotheses = _normalize_sequence_language(hypotheses, field="hypotheses")
        summary = _normalize_sequence_language(summary, field="summary")
        opportunities = opportunities if _is_meaningful_text(opportunities) else "unspecified_opportunities"
        uncertainties = uncertainties if _is_meaningful_text(uncertainties) else "unspecified_uncertainties"

        delta = _coerce_delta_score(
            blocks.get("delta_score"),
            scene=scene,
            events=events,
            summary=summary,
        )
        snapshot = EnvironmentSnapshot(
            scene=scene,
            events=events,
            hypotheses=hypotheses,
            opportunities=opportunities,
            uncertainties=uncertainties,
            summary=summary,
            delta_score=delta,
        )
        return EnvProcessorParseResult(snapshot=snapshot, reasoning_text=reasoning, raw_text=raw_text)

    payload_text = blocks.get("answer", raw_text)
    data = _extract_first_json_dict(payload_text)
    if data is not None:
        snapshot = _snapshot_from_json(data)
        if snapshot is not None:
            return EnvProcessorParseResult(snapshot=snapshot, reasoning_text=reasoning, raw_text=raw_text)

    snapshot = _snapshot_from_unstructured_text(payload_text)
    if snapshot is None:
        return None
    return EnvProcessorParseResult(snapshot=snapshot, reasoning_text=reasoning, raw_text=raw_text)


def _extract_tag_blocks(raw_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for match in _TAG_PATTERN.finditer(raw_text or ""):
        tag = match.group(1).strip().lower()
        body = match.group(2)
        out[tag] = body
    return out


def _extract_fuzzy_tag_blocks(raw_text: str, expected_tags: tuple[str, ...]) -> Dict[str, str]:
    expected = set(expected_tags)
    text = str(raw_text or "")
    openings = list(re.finditer(r"<([a-z_]+)>", text, flags=re.IGNORECASE))
    out: Dict[str, str] = {}
    for idx, match in enumerate(openings):
        tag = match.group(1).strip().lower()
        if tag not in expected:
            continue
        body_start = match.end()
        body_end = openings[idx + 1].start() if (idx + 1) < len(openings) else len(text)
        body = text[body_start:body_end]
        body = re.sub(r"</[a-z_]+>\s*$", "", body, flags=re.IGNORECASE).strip()
        if body and tag not in out:
            out[tag] = body
    return out


def _coerce_reasoning(blocks: Dict[str, str]) -> Optional[str]:
    reasoning = blocks.get("think")
    if reasoning is None:
        return None
    token = reasoning.strip()
    return token if token else None


def _snapshot_from_json(data: Dict[str, Any]) -> Optional[EnvironmentSnapshot]:
    def lookup(*keys: str) -> Any:
        for key in keys:
            if key in data:
                return data[key]
        for value in data.values():
            if isinstance(value, dict):
                for key in keys:
                    if key in value:
                        return value[key]
        return None

    def text_field(*keys: str) -> str:
        value = lookup(*keys)
        if value is None:
            return ""
        if isinstance(value, str):
            return " ".join(value.split()).strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, (list, dict)):
            try:
                return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
            except TypeError:
                return str(value)
        return str(value)

    scene = text_field("scene", "scene_description", "environment", "environment_content")
    events = text_field("events", "event_context", "event_summary", "activity", "activities")
    hypotheses = text_field("hypotheses", "inferences", "hypothesis")
    opportunities = text_field("opportunities", "help_opportunities", "support_opportunities")
    uncertainties = text_field("uncertainties", "unknowns", "limits")
    summary = text_field("summary", "scene_summary", "synthesis", "inference")

    delta_raw = lookup("delta_score", "delta", "change_score", "event_delta")
    try:
        delta = float(delta_raw) if delta_raw is not None else 0.2
    except (TypeError, ValueError):
        delta = 0.2
    delta = max(0.0, min(1.0, delta))

    scene = _normalize_text(scene)
    events = _normalize_text(events)
    hypotheses = _normalize_text(hypotheses)
    opportunities = _normalize_text(opportunities)
    uncertainties = _normalize_text(uncertainties)
    summary = _normalize_text(summary)

    scene = _normalize_sequence_language(scene, field="scene")
    events = _normalize_sequence_language(events, field="events")
    hypotheses = _normalize_sequence_language(hypotheses, field="hypotheses")
    summary = _normalize_sequence_language(summary, field="summary")

    if not any((scene, events, hypotheses, opportunities, uncertainties, summary)):
        return None
    if not _is_meaningful_text(summary):
        summary = next((token for token in (events, scene, hypotheses, opportunities) if _is_meaningful_text(token)), "")
    if not summary:
        return None

    return EnvironmentSnapshot(
        scene=scene or "unspecified_scene",
        events=events or "unspecified_events",
        hypotheses=hypotheses or "unspecified_hypotheses",
        opportunities=opportunities or "unspecified_opportunities",
        uncertainties=uncertainties or "unspecified_uncertainties",
        summary=summary,
        delta_score=delta,
    )


def _snapshot_from_unstructured_text(raw_text: str) -> Optional[EnvironmentSnapshot]:
    if not raw_text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", " ", raw_text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?answer>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = _FENCE_PATTERN.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split()).strip()
    if len(cleaned) < 12 or not _is_meaningful_text(cleaned):
        return None
    summary = _normalize_sequence_language(cleaned[:420], field="summary")
    return EnvironmentSnapshot(
        scene=_normalize_sequence_language(summary, field="scene"),
        events=_normalize_sequence_language(summary, field="events"),
        hypotheses="inferred from unstructured model output",
        opportunities="unknown",
        uncertainties="parser_fallback_unstructured_response",
        summary=summary,
        delta_score=0.2,
    )


def _extract_first_json_dict(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    text = str(raw_text)
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for end_idx in range(start_idx, len(text)):
            cur = text[end_idx]
            if in_str:
                if esc:
                    esc = False
                elif cur == "\\":
                    esc = True
                elif cur == '"':
                    in_str = False
                continue
            if cur == '"':
                in_str = True
                continue
            if cur == "{":
                depth += 1
            elif cur == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx : end_idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
    return None


def _normalize_text(value: Any) -> str:
    token = str(value or "")
    token = _FENCE_PATTERN.sub(" ", token)
    token = " ".join(token.split()).strip()
    if token in {"...", "<...>", "_empty_"}:
        return ""
    if _TEMPLATE_ECHO_PATTERN.search(token):
        return ""
    if _INSTRUCTION_PATTERN.search(token) and len(token) < 180:
        return ""
    return token


def _normalize_sequence_language(value: Any, *, field: str) -> str:
    token = _normalize_text(value)
    if not token:
        return ""

    rewrites = (
        (r"\bmultiple images depict\b", "Across the recent frame sequence,"),
        (r"\bmultiple images show\b", "Across the recent frame sequence,"),
        (r"\bthere are four frames depicting\b", "Across the recent frame sequence,"),
        (r"\bvariations of the same\b", "a short temporal sequence of the same"),
        (r"\bdifferent perspective(?:s)?\b", "different moments in the sequence"),
        (r"\bin the first (?:frame|image)\b", "earlier in the sequence"),
        (r"\bin the second (?:frame|image)\b", "later in the sequence"),
        (r"\bin the third (?:frame|image)\b", "later in the sequence"),
        (r"\bin the fourth (?:frame|image)\b", "later in the sequence"),
    )
    for pattern, repl in rewrites:
        token = re.sub(pattern, repl, token, flags=re.IGNORECASE)

    def _ordinal_repl(match: re.Match[str]) -> str:
        ordinal = match.group(1).strip().lower()
        if ordinal == "first":
            return "earlier in the sequence"
        return "later in the sequence"

    token = _ORDINAL_FRAME_PATTERN.sub(_ordinal_repl, token)
    token = " ".join(token.split()).strip()

    if field == "events":
        lower = token.lower()
        if lower.startswith("the image") or lower.startswith("the video"):
            token = f"Across the sequence, {token[4:]}"
    return token


def _coerce_delta_score(raw_value: Any, *, scene: str, events: str, summary: str) -> float:
    token = str(raw_value).strip().lower()
    if token:
        if token in {"low", "small", "minor"}:
            return 0.2
        if token in {"medium", "moderate"}:
            return 0.5
        if token in {"high", "large", "major"}:
            return 0.75
    try:
        delta = float(raw_value)
    except (TypeError, ValueError):
        delta = 0.2
    return max(0.0, min(1.0, delta))


def _is_meaningful_text(value: str) -> bool:
    token = " ".join(str(value or "").split()).strip()
    if len(token) < 8:
        return False
    if token.lower() in {"unknown", "n/a", "none", "_empty_"}:
        return False
    if _TEMPLATE_ECHO_PATTERN.search(token):
        return False
    if token in {"...", "<...>"}:
        return False
    if _INSTRUCTION_PATTERN.search(token) and len(token) < 220:
        return False
    return True
