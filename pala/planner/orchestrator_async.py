from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import threading
import time
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..perception.frame_cache import LatestFrameCache
from ..types import (
    ActionPlan,
    PrimitiveKind,
    HoldCommand,
    BreathCommand,
    GlanceCommand,
    NodCommand,
    OrientToZoneCommand,
    PerceptionState,
)
from .heuristic import HeuristicPlanner
from .protocol import PlannerInterface
from .state_models import OrchestratorDecision, SceneSummary, SessionMemory
from .summarizer_async import AsyncSceneSummarizer
from .session_memory import SessionMemoryManager

logger = logging.getLogger(__name__)


@dataclass
class _OrchestratorRequest:
    summary: SceneSummary
    memory: SessionMemory


class AsyncOrchestratorPlanner(PlannerInterface):
    """Two-agent stack: async summarizer + async orchestrator with deterministic memory."""

    def __init__(
        self,
        *,
        frame_cache: LatestFrameCache,
        fallback: Optional[PlannerInterface] = None,
        provider: str = "brev",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "nvidia/cosmos-reason2-2b",
        planner_prompt: Optional[str] = None,
        summarizer_hz: float = 2.0,
        orchestrator_hz: float = 1.0,
        request_timeout_ms: int = 5000,
        response_ttl_ms: int = 1500,
    ) -> None:
        self._fallback = fallback or HeuristicPlanner()
        self._provider = str(provider).strip().lower()
        self._chat_url = _normalize_chat_url(base_url)
        self._api_key = api_key
        self._model = str(model or "nvidia/cosmos-reason2-2b")
        self._planner_prompt = (planner_prompt or "").strip()
        self._request_timeout_s = max(0.05, float(request_timeout_ms) / 1000.0)
        self._response_ttl_s = max(0.1, float(response_ttl_ms) / 1000.0)
        self._orchestrator_period_s = 1.0 / max(0.2, float(orchestrator_hz))
        self._remote_enabled = self._provider in {"brev", "cosmos", "openai"} and self._chat_url is not None
        self._request_count = 0
        self._success_count = 0
        self._last_stats_log_s = 0.0

        self._summarizer = AsyncSceneSummarizer(
            frame_cache=frame_cache,
            max_hz=summarizer_hz,
            response_ttl_ms=max(int(response_ttl_ms), 1200),
        )
        self._memory = SessionMemoryManager()

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._pending: Optional[_OrchestratorRequest] = None
        self._last_submit_s = 0.0

        self._latest_decision: Optional[OrchestratorDecision] = None
        self._latest_action: Optional[ActionPlan] = None
        self._latest_action_ts_s: Optional[float] = None
        self._latest_summary: Optional[SceneSummary] = None
        self._latest_memory: Optional[SessionMemory] = None

        if self._remote_enabled:
            logger.info(
                "orchestrator remote enabled provider=%s url=%s model=%s",
                self._provider,
                self._chat_url,
                self._model,
            )
        else:
            logger.info("orchestrator remote disabled provider=%s base_url=%s", self._provider, self._chat_url)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def plan(self, st: PerceptionState) -> ActionPlan:
        now = time.monotonic()
        self._summarizer.update(st)
        summary = self._summarizer.latest()
        if summary is None:
            return self._fallback.plan(st)
        memory = self._memory.update(summary)
        self._latest_summary = summary
        self._latest_memory = memory

        if (now - self._last_submit_s) >= self._orchestrator_period_s:
            with self._lock:
                self._pending = _OrchestratorRequest(summary=summary, memory=memory)
                self._last_submit_s = now
                self._cond.notify_all()

        with self._lock:
            action = self._latest_action
            action_ts = self._latest_action_ts_s

        if action is not None and action_ts is not None and (now - action_ts) <= self._response_ttl_s:
            return action

        decision = _local_decision(summary, memory)
        return _decision_to_action(decision)

    def snapshot(self) -> tuple[Optional[SceneSummary], Optional[SessionMemory], Optional[OrchestratorDecision]]:
        return self._latest_summary, self._latest_memory, self._latest_decision

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)
        self._summarizer.shutdown()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                while not self._stop.is_set() and self._pending is None:
                    self._cond.wait(timeout=0.1)
                if self._stop.is_set():
                    break
                req = self._pending
                self._pending = None

            if req is None:
                continue

            try:
                if self._remote_enabled:
                    decision = self._remote_decision(req.summary, req.memory)
                    if decision is None:
                        decision = _local_decision(req.summary, req.memory)
                else:
                    decision = _local_decision(req.summary, req.memory)
                action = _decision_to_action(decision)
            except Exception as exc:
                logger.warning("orchestrator planning failed: %s", exc)
                decision = _local_decision(req.summary, req.memory)
                action = _decision_to_action(decision)

            self._memory.note_intent(decision.intent)
            self._latest_decision = decision
            with self._lock:
                self._latest_action = action
                self._latest_action_ts_s = time.monotonic()

            now = time.monotonic()
            if now - self._last_stats_log_s >= 10.0:
                logger.info(
                    "orchestrator stats requests=%d successes=%d source=%s intent=%s style=%s",
                    self._request_count,
                    self._success_count,
                    decision.source,
                    decision.intent,
                    decision.style,
                )
                self._last_stats_log_s = now

    def _remote_decision(self, summary: SceneSummary, memory: SessionMemory) -> Optional[OrchestratorDecision]:
        assert self._chat_url is not None
        self._request_count += 1
        payload = self._build_payload(summary, memory)
        response = _post_json(
            self._chat_url,
            payload,
            timeout_s=self._request_timeout_s,
            api_key=self._api_key,
        )
        content = _extract_content(response)
        if content is None:
            return None
        decision = _parse_decision_content(content)
        if decision is None:
            return None
        self._success_count += 1
        return decision

    def _build_payload(self, summary: SceneSummary, memory: SessionMemory) -> dict[str, Any]:
        context = {
            "summary": {
                "person_present": summary.person_present,
                "zone_hint": summary.zone_hint,
                "primary_person_conf": summary.primary_person_conf,
                "activity_hint": summary.activity_hint,
                "uncertainty_flags": summary.uncertainty_flags,
                "frame_age_ms": summary.frame_age_ms,
            },
            "memory": {
                "interaction_state": memory.interaction_state,
                "task_hypothesis": memory.task_hypothesis,
                "staleness_ms": memory.staleness_ms,
                "recent_intents": memory.recent_intents[-5:],
            },
        }

        system_prompt = (
            "You are an interaction orchestrator for a social desk robot lamp. "
            "Return JSON only with keys: intent, style, primitive_hint, target_zone, confidence, rationale. "
            "style must be one of ['calm','curious','focused']. "
            "primitive_hint should be one of ['hold','breath','glance','nod','orient_to_zone'] or null. "
            "target_zone should be one of ['left','center','right'] or null. "
            "confidence in [0,1]. Keep rationale concise."
        )
        if self._planner_prompt:
            system_prompt += f" Operator guidance: {self._planner_prompt}"

        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, separators=(",", ":"), ensure_ascii=True)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 180,
            "stream": False,
        }


def _local_decision(summary: SceneSummary, memory: SessionMemory) -> OrchestratorDecision:
    if not summary.person_present:
        if memory.interaction_state == "searching":
            return OrchestratorDecision(
                intent="reacquire_attention",
                style="focused",
                primitive_hint="orient_to_zone",
                target_zone=summary.zone_hint,
                confidence=0.55,
                rationale="person likely nearby but currently out of frame",
                source="local",
            )
        return OrchestratorDecision(
            intent="idle_presence",
            style="calm",
            primitive_hint="breath",
            target_zone=None,
            confidence=0.6,
            rationale="no person present",
            source="local",
        )

    if summary.activity_hint == "transitioning":
        return OrchestratorDecision(
            intent="track_transition",
            style="curious",
            primitive_hint="glance",
            target_zone=summary.zone_hint,
            confidence=0.7,
            rationale="person motion implies context shift",
            source="local",
        )

    if summary.zone_hint == "center":
        return OrchestratorDecision(
            intent="engaged_focus",
            style="focused",
            primitive_hint="nod",
            target_zone=summary.zone_hint,
            confidence=0.72,
            rationale="stable centered engagement",
            source="local",
        )

    return OrchestratorDecision(
        intent="maintain_presence",
        style="curious",
        primitive_hint="orient_to_zone",
        target_zone=summary.zone_hint,
        confidence=0.66,
        rationale="person present off-center",
        source="local",
    )


def _decision_to_action(decision: OrchestratorDecision) -> ActionPlan:
    hint = (decision.primitive_hint or "").strip().lower()
    if hint == "nod":
        return ActionPlan(
            primitive=PrimitiveKind.NOD,
            command=NodCommand(amp_rad=0.2, duration_s=0.5, cycles=1, rate_rad_s=1.8),
            confidence=decision.confidence,
            explanation=f"{decision.source}:{decision.rationale}",
            style=decision.style,
        )
    if hint == "glance":
        direction = "left"
        if decision.target_zone == "right":
            direction = "right"
        elif decision.target_zone == "left":
            direction = "left"
        elif "right" in decision.rationale:
            direction = "right"
        return ActionPlan(
            primitive=PrimitiveKind.GLANCE,
            command=GlanceCommand(direction=direction, amp_rad=0.24, duration_s=0.55, rate_rad_s=1.6),
            confidence=decision.confidence,
            explanation=f"{decision.source}:{decision.rationale}",
            style=decision.style,
        )
    if hint == "orient_to_zone":
        zone = decision.target_zone if decision.target_zone in {"left", "center", "right"} else "center"
        if zone == "center" and "left" in decision.rationale:
            zone = "left"
        elif zone == "center" and "right" in decision.rationale:
            zone = "right"
        return ActionPlan(
            primitive=PrimitiveKind.ORIENT_TO_ZONE,
            command=OrientToZoneCommand(zone=zone, amp_rad=0.2, rate_rad_s=1.4),
            confidence=decision.confidence,
            explanation=f"{decision.source}:{decision.rationale}",
            style=decision.style,
        )
    if hint == "hold":
        return ActionPlan(
            primitive=PrimitiveKind.HOLD,
            command=HoldCommand(),
            confidence=decision.confidence,
            explanation=f"{decision.source}:{decision.rationale}",
            style=decision.style,
        )
    return ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command=BreathCommand(amp_rad=0.08, period_s=6.5, rate_rad_s=1.0),
        confidence=decision.confidence,
        explanation=f"{decision.source}:{decision.rationale}",
        style=decision.style,
    )


def _normalize_chat_url(base_url: Optional[str]) -> Optional[str]:
    if base_url is None:
        return None
    base = str(base_url).strip()
    if not base:
        return None
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if base.endswith("/v1/"):
        return f"{base}chat/completions"
    return f"{base.rstrip('/')}/v1/chat/completions"


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float, api_key: Optional[str]) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    req = urllib_request.Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid JSON response from orchestrator endpoint") from exc


def _extract_content(response: dict[str, Any]) -> Optional[str]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return None
    message = choice0.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) and content.strip() else None


def _parse_decision_content(content: str) -> Optional[OrchestratorDecision]:
    cleaned = content.strip()
    data = _parse_json_obj(cleaned)
    if data is None:
        candidate = _extract_first_json_object(cleaned)
        data = None if candidate is None else _parse_json_obj(candidate)
    if data is None or not isinstance(data, dict):
        return None

    if isinstance(data.get("decision"), dict):
        data = data["decision"]

    intent = str(data.get("intent", "")).strip()
    style = str(data.get("style", "calm")).strip().lower()
    primitive_hint = data.get("primitive_hint")
    primitive_hint_str = None if primitive_hint in (None, "") else str(primitive_hint).strip().lower()
    target_zone_raw = data.get("target_zone", data.get("zone_hint"))
    target_zone = None if target_zone_raw in (None, "") else str(target_zone_raw).strip().lower()
    if target_zone not in {None, "left", "center", "right"}:
        target_zone = None
    rationale = str(data.get("rationale", "")).strip()
    if not intent or not rationale:
        return None
    try:
        confidence = float(data.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    if style not in {"calm", "curious", "focused"}:
        style = "calm"
    return OrchestratorDecision(
        intent=intent,
        style=style,
        primitive_hint=primitive_hint_str,
        target_zone=target_zone,
        confidence=confidence,
        rationale=rationale,
        source="remote",
    )


def _parse_json_obj(raw: str) -> Optional[dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_first_json_object(raw: str) -> Optional[str]:
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None
