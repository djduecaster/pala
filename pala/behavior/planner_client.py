from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Mapping, Optional


_TAG_PATTERN = re.compile(r"<([a-z_]+)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_FENCE_PATTERN = re.compile(r"```(?:[a-z0-9_+-]+)?|```", re.IGNORECASE)


@dataclass(frozen=True)
class _ResponseFormatSignals:
    has_decision_open: bool
    has_decision_close: bool
    has_rationale_open: bool
    has_rationale_close: bool
    has_fence: bool
    has_markdown_heading: bool

    @property
    def strict_tagged(self) -> bool:
        return self.has_decision_open and self.has_decision_close and self.has_rationale_open and self.has_rationale_close


@dataclass
class PlannerClientConfig:
    max_inflight: int = 1


@dataclass
class PlannerDecision:
    act_now: bool
    primitive: Optional[str]
    command: Dict[str, Any]
    style: str
    confidence: float
    rationale_short: str
    reasoning_text: Optional[str]
    raw_text: str


class CosmosPlannerClient:
    """
    Phase-0/1 planner scaffolding.

    Handles latest-only in-flight bookkeeping and strict tag parsing.
    Transport/network dispatch is intentionally deferred to phase 2.
    """

    def __init__(self, config: Optional[PlannerClientConfig] = None):
        self._cfg = config or PlannerClientConfig()
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

    def complete_request(self, raw_text: str) -> Optional[PlannerDecision]:
        self._inflight = False
        return parse_planner_response(raw_text)

    def take_latest_pending(self) -> Optional[Mapping[str, Any]]:
        payload = self._pending_payload
        self._pending_payload = None
        return payload


def parse_planner_response(raw_text: str) -> Optional[PlannerDecision]:
    signals = _response_format_signals(raw_text)
    strict_blocks = _extract_tag_blocks(raw_text)
    fuzzy_blocks = _extract_fuzzy_tag_blocks(raw_text, ("decision_json", "rationale_short", "think", "answer"))
    blocks = {**fuzzy_blocks, **strict_blocks}
    reasoning = _coerce_reasoning(blocks)
    raw_lower = str(raw_text or "").lower()
    has_tag_markers = (
        any(tag in blocks for tag in ("decision_json", "rationale_short", "answer"))
        or "<decision_json>" in raw_lower
        or "<rationale_short>" in raw_lower
    )
    if "decision_json" in blocks and "rationale_short" in blocks:
        decision_payload = _strip_markdown_fences(blocks["decision_json"])
        data = _parse_json_object(decision_payload)
        if isinstance(data, dict):
            return _planner_decision_from_json(
                data,
                raw_text=raw_text,
                reasoning=reasoning,
                rationale_override=str(blocks["rationale_short"]).strip(),
                signals=signals,
            )

    if "decision_json" in blocks:
        decision_payload = _strip_markdown_fences(blocks["decision_json"])
        data = _extract_first_json_dict(decision_payload)
        if isinstance(data, dict):
            return _planner_decision_from_json(
                data,
                raw_text=raw_text,
                reasoning=reasoning,
                rationale_override=_compact_text(blocks.get("rationale_short"))
                or _extract_rationale_from_raw(raw_text),
                signals=signals,
            )

    payload_text = _strip_markdown_fences(blocks.get("answer", raw_text))
    data = _extract_first_json_dict(payload_text)
    if data is not None:
        return _planner_decision_from_json(
            data,
            raw_text=raw_text,
            reasoning=reasoning,
            rationale_override=_extract_rationale_from_raw(raw_text),
            signals=signals,
        )

    if has_tag_markers:
        return None

    fallback_rationale = _extract_rationale_from_raw(payload_text) or _compact_text(payload_text)
    if not fallback_rationale:
        return None
    return PlannerDecision(
        act_now=False,
        primitive=None,
        command={},
        style="calm",
        confidence=0.0,
        rationale_short=fallback_rationale[:220],
        reasoning_text=reasoning,
        raw_text=raw_text,
    )


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


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y", "on"}:
            return True
        if token in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


def _coerce_reasoning(blocks: Dict[str, str]) -> Optional[str]:
    reasoning = blocks.get("think")
    if reasoning is None:
        return None
    token = reasoning.strip()
    return token if token else None


def _planner_decision_from_json(
    data: Dict[str, Any],
    *,
    raw_text: str,
    reasoning: Optional[str],
    rationale_override: Optional[str],
    signals: _ResponseFormatSignals,
) -> Optional[PlannerDecision]:
    work = dict(data)

    decision_json = work.get("decision_json")
    if isinstance(decision_json, str):
        try:
            parsed = json.loads(decision_json)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            work = parsed
    elif isinstance(decision_json, dict):
        work = dict(decision_json)

    prediction_block = work.get("prediction")
    if isinstance(prediction_block, dict):
        merged = dict(work)
        merged.update(prediction_block)
        work = merged

    action_details = work.get("action_details")
    if isinstance(action_details, dict):
        merged = dict(work)
        for key, value in action_details.items():
            merged.setdefault(key, value)
        work = merged

    primitive_raw = _lookup(work, "primitive", "primitive_hint", "action", "action_name")
    if primitive_raw is None:
        prediction_raw = _lookup(work, "prediction")
        if isinstance(prediction_raw, str):
            primitive_raw = prediction_raw
    primitive = _normalize_primitive(primitive_raw)

    rationale_short = rationale_override or _compact_text(
        _lookup(work, "rationale_short", "rationale", "inference", "reason", "explanation")
    )
    if not rationale_short:
        rationale_short = _extract_rationale_from_raw(raw_text)
    if not rationale_short:
        rationale_short = "no rationale provided"

    command = _coerce_command(
        work,
        primitive=primitive,
        raw_text=raw_text,
        rationale_text=rationale_short,
    )

    style_raw = _lookup(work, "style", "mood")
    style = _compact_text(style_raw or "calm").lower() or "calm"

    confidence = _coerce_confidence(
        _lookup(work, "confidence", "inference_confidence", "score", "probability"),
        default=0.5,
    )

    act_now_raw = _lookup(work, "act_now", "should_act", "execute", "take_action")
    if act_now_raw is None:
        act_now = primitive is not None
    else:
        act_now = _coerce_bool(act_now_raw, default=(primitive is not None))

    decision = PlannerDecision(
        act_now=act_now,
        primitive=primitive,
        command=command,
        style=style,
        confidence=confidence,
        rationale_short=rationale_short[:220],
        reasoning_text=reasoning,
        raw_text=raw_text,
    )
    return _apply_low_quality_guard(decision, signals=signals)


def _coerce_command(
    data: Dict[str, Any],
    *,
    primitive: Optional[str],
    raw_text: str,
    rationale_text: str,
) -> Dict[str, Any]:
    command: Dict[str, Any] = {}
    command_raw = data.get("command")
    if isinstance(command_raw, dict):
        source_items = dict(command_raw)
    else:
        source_items = {}

    for key in ("zone", "target_zone", "target", "direction", "amp_rad", "amplitude", "rate_rad_s", "rate", "speed", "duration_s"):
        value = source_items.get(key)
        if value is None:
            value = data.get(key)
        if value is None:
            continue
        if key in {"target_zone", "target", "direction"}:
            key = "zone"
        elif key in {"amplitude"}:
            key = "amp_rad"
        elif key in {"rate", "speed"}:
            key = "rate_rad_s"
        command[key] = value

    if primitive == "orient_to_zone":
        zone = _normalize_zone(command.get("zone"))
        if zone is None:
            zone = _infer_zone_from_text(raw_text, rationale_text)
        if zone is not None:
            command["zone"] = zone
        else:
            command.pop("zone", None)
    return command


def _lookup(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _normalize_primitive(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = _compact_text(value).lower()
    if token in {"", "none", "null"}:
        return None
    return token


def _coerce_confidence(value: Any, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    return max(0.0, min(1.0, out))


def _compact_text(value: Any) -> str:
    if value is None:
        return ""
    token = _strip_markdown_fences(str(value))
    token = _FENCE_PATTERN.sub(" ", token)
    token = " ".join(token.split()).strip()
    return token


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
                    parsed = _loads_relaxed_json(candidate)
                    if parsed is None:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
    return None


def _extract_rationale_from_raw(raw_text: str) -> str:
    text = str(raw_text or "")
    if not text:
        return ""
    work = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    work = re.sub(r"<decision_json>.*?(?:</[a-z_]+>|$)", " ", work, flags=re.IGNORECASE | re.DOTALL)
    work = re.sub(r"</?rationale_short>", " ", work, flags=re.IGNORECASE)
    work = re.sub(r"</?answer>", " ", work, flags=re.IGNORECASE)
    work = re.sub(r"</?[a-z_]+>", " ", work, flags=re.IGNORECASE)
    work = _FENCE_PATTERN.sub(" ", work)
    work = re.sub(r"\{.*\}", " ", work, flags=re.DOTALL)
    work = " ".join(work.split()).strip()
    low = work.lower()
    if not work:
        return ""
    if "decision_json" in low or "rationale_short" in low:
        return ""
    return work[:220]


def _response_format_signals(raw_text: str) -> _ResponseFormatSignals:
    text = str(raw_text or "").lower()
    return _ResponseFormatSignals(
        has_decision_open="<decision_json>" in text,
        has_decision_close="</decision_json>" in text,
        has_rationale_open="<rationale_short>" in text,
        has_rationale_close="</rationale_short>" in text,
        has_fence="```" in text,
        has_markdown_heading="###" in text,
    )


def _apply_low_quality_guard(decision: PlannerDecision, *, signals: _ResponseFormatSignals) -> PlannerDecision:
    # De-prioritize low-quality, fence-heavy outputs that frequently repeat low-energy primitives.
    # Keep valid tagged responses and responses with explicit rationale.
    if signals.strict_tagged:
        return decision
    if not (signals.has_fence or signals.has_markdown_heading):
        return decision
    if decision.rationale_short and decision.rationale_short != "no rationale provided":
        return decision
    if decision.primitive not in {"breath", "hold", "home"}:
        return decision
    if decision.command:
        return decision
    return PlannerDecision(
        act_now=False,
        primitive=None,
        command={},
        style=decision.style,
        confidence=min(0.4, decision.confidence),
        rationale_short=decision.rationale_short,
        reasoning_text=decision.reasoning_text,
        raw_text=decision.raw_text,
    )


def _normalize_zone(value: Any) -> Optional[str]:
    token = _compact_text(value).lower()
    if token in {"", "none", "null"}:
        return None
    if token in {"left", "l", "left_side"}:
        return "left"
    if token in {"right", "r", "right_side"}:
        return "right"
    if token in {"center", "centre", "middle", "forward", "front", "straight"}:
        return "center"
    if "left" in token:
        return "left"
    if "right" in token:
        return "right"
    if any(word in token for word in ("center", "centre", "middle", "forward", "front")):
        return "center"
    return None


def _infer_zone_from_text(*segments: str) -> Optional[str]:
    text = " ".join(_compact_text(seg) for seg in segments if seg).lower()
    if not text:
        return None
    if re.search(r"\bleft\b|\bleft-side\b|\bleftward\b", text):
        return "left"
    if re.search(r"\bright\b|\bright-side\b|\brightward\b", text):
        return "right"
    if re.search(r"\bcenter\b|\bcentre\b|\bmiddle\b|\bforward\b|\bfront\b", text):
        return "center"
    return None


def _strip_markdown_fences(text: str) -> str:
    token = str(text or "")
    if "```" not in token:
        return token
    return _FENCE_PATTERN.sub(" ", token)


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    parsed = _loads_relaxed_json(text)
    if isinstance(parsed, dict):
        return parsed
    return None


def _loads_relaxed_json(candidate: str) -> Optional[Any]:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        normalized = re.sub(r"\btrue\b", "True", candidate, flags=re.IGNORECASE)
        normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bnull\b", "None", normalized, flags=re.IGNORECASE)
        try:
            parsed = ast.literal_eval(normalized)
        except (ValueError, SyntaxError):
            return None
    return parsed
