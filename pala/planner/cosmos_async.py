from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Optional
import threading
import time
from urllib import request as urllib_request
from urllib import error as urllib_error

from ..perception.frame_cache import LatestFrameCache
from ..types import (
    ActionPlan,
    PerceptionState,
    PrimitiveKind,
    HoldCommand,
    BreathCommand,
    action_plan_from_dict,
)
from .heuristic import HeuristicPlanner
from .protocol import PlannerInterface

logger = logging.getLogger(__name__)


@dataclass
class _CosmosRequest:
    state: PerceptionState
    frame_mono_ns: Optional[int]
    frame_shape: Optional[tuple[int, int, int]]
    frame_age_ms: Optional[float]


class AsyncCosmosPlanner(PlannerInterface):
    """Async planner for Cosmos/Brev integration.

    A background worker consumes the latest perception snapshot and attempts
    a remote planning call. If no fresh remote action is available, caller
    gets fallback planner output.
    """

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
        max_hz: float = 1.0,
        max_frame_age_ms: int = 500,
        request_timeout_ms: int = 5000,
        mock_latency_ms: int = 150,
        response_ttl_ms: int = 1500,
    ) -> None:
        self._frame_cache = frame_cache
        self._fallback = fallback or HeuristicPlanner()
        self._provider = str(provider).strip().lower()
        self._chat_url = _normalize_chat_url(base_url)
        self._api_key = api_key
        self._model = str(model or "nvidia/cosmos-reason2-2b")
        self._planner_prompt = (planner_prompt or "").strip()
        self._max_hz = max(0.1, float(max_hz))
        self._max_frame_age_ms = int(max_frame_age_ms)
        self._request_timeout_s = max(0.05, float(request_timeout_ms) / 1000.0)
        self._mock_latency_s = max(0.0, float(mock_latency_ms) / 1000.0)
        self._response_ttl_s = max(0.1, float(response_ttl_ms) / 1000.0)
        self._remote_enabled = self._provider in {"brev", "cosmos", "openai"} and self._chat_url is not None
        self._warned_unconfigured = False
        self._request_count = 0
        self._success_count = 0
        self._last_stats_log_s = 0.0

        if self._remote_enabled:
            logger.info(
                "cosmos remote planner enabled provider=%s url=%s model=%s max_hz=%.2f timeout_ms=%d",
                self._provider,
                self._chat_url,
                self._model,
                self._max_hz,
                int(self._request_timeout_s * 1000.0),
            )
        else:
            logger.info(
                "cosmos remote planner disabled provider=%s base_url=%s (using mock/fallback)",
                self._provider,
                self._chat_url,
            )

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._pending: Optional[_CosmosRequest] = None
        self._last_submit_s = 0.0
        self._latest_action: Optional[ActionPlan] = None
        self._latest_action_ts_s: Optional[float] = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def plan(self, st: PerceptionState) -> ActionPlan:
        now = time.monotonic()
        submit_period_s = 1.0 / self._max_hz
        if (now - self._last_submit_s) >= submit_period_s:
            snap = self._frame_cache.get(max_age_ms=self._max_frame_age_ms)
            frame_shape = None if snap is None else tuple(snap.frame.shape)
            frame_age_ms = None if snap is None else (time.monotonic_ns() - snap.mono_ns) / 1_000_000.0
            req = _CosmosRequest(
                state=st,
                frame_mono_ns=None if snap is None else snap.mono_ns,
                frame_shape=frame_shape,
                frame_age_ms=frame_age_ms,
            )
            with self._lock:
                self._pending = req
                self._last_submit_s = now
                self._cond.notify_all()

        with self._lock:
            action = self._latest_action
            action_ts = self._latest_action_ts_s

        if action is not None and action_ts is not None and (now - action_ts) <= self._response_ttl_s:
            return action
        return self._fallback.plan(st)

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)

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

            action: Optional[ActionPlan] = None
            try:
                if self._remote_enabled:
                    action = self._remote_plan(req)
                else:
                    if not self._warned_unconfigured and self._provider in {"brev", "cosmos", "openai"}:
                        logger.warning(
                            "cosmos provider=%s selected but base URL is missing; using mock planner",
                            self._provider,
                        )
                        self._warned_unconfigured = True
                    if self._mock_latency_s > 0:
                        time.sleep(self._mock_latency_s)
                    action = self._mock_plan(req)
            except Exception as exc:
                logger.warning("cosmos planning failed; using fallback planner path: %s", exc)
                action = None

            if action is None:
                continue
            with self._lock:
                self._latest_action = action
                self._latest_action_ts_s = time.monotonic()

            now = time.monotonic()
            if now - self._last_stats_log_s >= 10.0:
                logger.info(
                    "cosmos stats requests=%d successes=%d",
                    self._request_count,
                    self._success_count,
                )
                self._last_stats_log_s = now

    def _mock_plan(self, req: _CosmosRequest) -> ActionPlan:
        if req.state.primary_person is None:
            return ActionPlan(
                primitive=PrimitiveKind.HOLD,
                command=HoldCommand(),
                confidence=0.25,
                explanation="cosmos_mock:no_person",
            )

        has_frame = req.frame_mono_ns is not None and req.frame_shape is not None
        return ActionPlan(
            primitive=PrimitiveKind.BREATH,
            command=BreathCommand(amp_rad=0.1, period_s=6.0, rate_rad_s=1.1),
            confidence=0.45,
            explanation=f"cosmos_mock:presence frame={has_frame}",
        )

    def _remote_plan(self, req: _CosmosRequest) -> Optional[ActionPlan]:
        assert self._chat_url is not None
        self._request_count += 1
        payload = self._build_payload(req)
        response = _post_json(
            self._chat_url,
            payload,
            timeout_s=self._request_timeout_s,
            api_key=self._api_key,
        )
        content = _extract_content(response)
        if content is None:
            logger.warning("cosmos response missing choices[0].message.content")
            return None
        action = _parse_action_content(content)
        if action is None:
            preview = content.strip().replace("\n", "\\n")
            if len(preview) > 240:
                preview = preview[:240] + "..."
            logger.warning("cosmos response could not be parsed into ActionPlan content=%s", preview)
            return None
        if action.explanation is None:
            action.explanation = "cosmos_remote"
        elif not action.explanation.startswith("cosmos_remote"):
            action.explanation = f"cosmos_remote:{action.explanation}"
        self._success_count += 1
        return action

    def _build_payload(self, req: _CosmosRequest) -> dict[str, Any]:
        state = req.state
        person = state.primary_person
        primary_person = None
        if person is not None:
            primary_person = {
                "cx": float(person.cx),
                "cy": float(person.cy),
                "w": float(person.w),
                "h": float(person.h),
                "conf": None if state.primary_person_conf is None else float(state.primary_person_conf),
            }

        context = {
            "zone_hint": (state.debug or {}).get("zone_hint"),
            "primary_person": primary_person,
            "latency_ms": state.latency_ms,
            "fps": state.fps,
            "frame": {
                "present": req.frame_mono_ns is not None and req.frame_shape is not None,
                "shape": None if req.frame_shape is None else list(req.frame_shape),
                "age_ms": req.frame_age_ms,
            },
        }
        if self._planner_prompt:
            context["planner_prompt"] = self._planner_prompt

        system_prompt = (
            "You are a robotics planner. Return only JSON with keys "
            "primitive, command, confidence, explanation. "
            f"Allowed primitive values: {[k.value for k in PrimitiveKind]}. "
            "Confidence must be 0..1. Keep command compact. "
            "No markdown, no prose, no code fences. "
            "Example: "
            '{"primitive":"breath","command":{"amp_rad":0.08,"period_s":7.0,"rate_rad_s":1.0},'
            '"confidence":0.5,"explanation":"short reason"}'
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
            "max_tokens": 120,
            "stream": False,
        }


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
        raise RuntimeError("invalid JSON response from cosmos endpoint") from exc


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


def _parse_action_content(content: str) -> Optional[ActionPlan]:
    cleaned = content.strip()

    data = _parse_json_obj(cleaned)
    if data is None:
        candidate = _extract_first_json_object(cleaned)
        data = None if candidate is None else _parse_json_obj(candidate)
    if data is None:
        return None

    if not isinstance(data, dict):
        return None
    return action_plan_from_dict(data)


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
