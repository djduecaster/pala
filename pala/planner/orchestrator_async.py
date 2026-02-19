from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
import io
import json
import logging
import threading
import time
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from PIL import Image

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
from .memory_manager import MemoryManager, MemoryManagerConfig
from .protocol import PlannerInterface
from .state_models import OrchestratorDecision
from .timeline import TimelineConfig, TimelineWriter

logger = logging.getLogger(__name__)

_TARGET_STATES = {"idle", "user_detected", "engaging", "tracking", "acknowledging", "reacquiring"}
_STYLES = {"calm", "curious", "focused"}
_URGENCY = {"low", "medium", "high"}
_PRIMITIVE_HINTS = {"hold", "breath", "glance", "nod", "orient_to_zone"}


@dataclass
class _OrchestratorRequest:
    state: PerceptionState
    frames: list[np.ndarray]
    frame_age_ms: Optional[float]
    control_active_primitive: Optional[str]
    control_active_age_s: Optional[float]


class AsyncOrchestratorPlanner(PlannerInterface):
    """Async orchestrator with transcript context and latest-only remote requests."""

    owns_semantic_behavior = True

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
        policy_version: str = "v1",
        policy_identity: str = (
            "You are PALA, a social desk companion lamp that should feel alive, expressive, and safe."
        ),
        policy_capabilities: str = (
            "You can move head/neck joints via primitives: hold, breath, glance, nod, orient_to_zone. "
            "You cannot manipulate external objects, move base position, or physically touch users."
        ),
        policy_safety: str = (
            "Avoid sudden aggressive motion. Prefer stable behavior. If uncertain, choose conservative actions."
        ),
        policy_style: str = (
            "Default style is calm; use curious for gentle tracking and focused for attentive task support."
        ),
        policy_output_contract: str = (
            "Return JSON only with target_state,intent,style,primitive_hint,target_zone,allow_interrupt,urgency,confidence,rationale."
        ),
        orchestrator_hz: float = 1.0,
        max_frame_age_ms: int = 500,
        video_window_s: float = 8.0,
        video_max_frames: int = 8,
        video_max_width: int = 320,
        video_jpeg_quality: int = 60,
        request_timeout_ms: int = 5000,
        response_ttl_ms: int = 1500,
        transcript_max_items: int = 80,
        context_max_transcript_items: int = 0,
        context_transcript_max_items: int = 24,
        context_transcript_per_type_max_items: int = 8,
        context_transcript_max_chars: int = 4000,
        context_memory_digest_max_items: int = 3,
        memory_enabled: bool = True,
        memory_jsonl_path: str = "logs/orchestrator_memory.jsonl",
        memory_recent_events: int = 10,
        memory_digest_items: int = 3,
        memory_distill_every_n_events: int = 20,
        decision_repeat_detector_window: int = 6,
        timeline_jsonl_path: str = "logs/orchestrator_timeline.jsonl",
        inflight_guard_enabled: bool = True,
        request_min_fresh_frames: int = 1,
        reasoning_probe_enabled: bool = False,
        reasoning_probe_hz: float = 0.1,
        reasoning_probe_timeout_ms: int = 8000,
        reasoning_probe_max_tokens: int = 1024,
    ) -> None:
        self._frame_cache = frame_cache
        self._fallback = fallback or HeuristicPlanner()
        self._provider = str(provider).strip().lower()
        self._chat_url = _normalize_chat_url(base_url)
        self._api_key = api_key
        self._model = str(model or "nvidia/cosmos-reason2-2b")
        self._planner_prompt = (planner_prompt or "").strip()
        self._policy_version = str(policy_version or "v1").strip() or "v1"
        self._policy_identity = str(policy_identity or "").strip()
        self._policy_capabilities = str(policy_capabilities or "").strip()
        self._policy_safety = str(policy_safety or "").strip()
        self._policy_style = str(policy_style or "").strip()
        self._policy_output_contract = str(policy_output_contract or "").strip()
        self._max_frame_age_ms = max(50, int(max_frame_age_ms))
        self._video_window_s = max(0.1, float(video_window_s))
        self._video_max_frames = max(1, int(video_max_frames))
        self._video_max_width = max(64, int(video_max_width))
        self._video_jpeg_quality = max(1, min(100, int(video_jpeg_quality)))
        self._request_timeout_s = max(0.05, float(request_timeout_ms) / 1000.0)
        self._response_ttl_s = max(0.1, float(response_ttl_ms) / 1000.0)
        self._orchestrator_period_s = 1.0 / max(0.2, float(orchestrator_hz))
        legacy_max_items = max(0, int(context_max_transcript_items))
        default_max_items = max(1, int(context_transcript_max_items))
        self._context_transcript_max_items = legacy_max_items if legacy_max_items > 0 else default_max_items
        self._context_transcript_per_type_max_items = max(1, int(context_transcript_per_type_max_items))
        self._context_transcript_max_chars = max(64, int(context_transcript_max_chars))
        self._context_memory_digest_max_items = max(1, int(context_memory_digest_max_items))
        self._timeline = TimelineWriter(TimelineConfig(enabled=True, jsonl_path=str(timeline_jsonl_path)))
        self._inflight_guard_enabled = bool(inflight_guard_enabled)
        self._request_min_fresh_frames = max(0, int(request_min_fresh_frames))
        self._reasoning_probe_enabled = bool(reasoning_probe_enabled)
        self._reasoning_probe_period_s = 1.0 / max(0.05, float(reasoning_probe_hz))
        self._reasoning_probe_timeout_s = max(0.1, float(reasoning_probe_timeout_ms) / 1000.0)
        self._reasoning_probe_max_tokens = max(128, int(reasoning_probe_max_tokens))
        self._remote_enabled = self._provider in {"brev", "cosmos", "openai"} and self._chat_url is not None
        self._request_count = 0
        self._success_count = 0
        self._probe_request_count = 0
        self._probe_success_count = 0
        self._last_stats_log_s = 0.0
        self._request_seq = 0

        self._transcript_lock = threading.Lock()
        self._transcript: deque[dict[str, Any]] = deque(maxlen=max(10, int(transcript_max_items)))
        self._frame_history: deque[tuple[float, int, np.ndarray]] = deque()
        self._last_seen_frame_mono_ns: Optional[int] = None
        self._decision_signature_history: deque[str] = deque(maxlen=max(2, int(decision_repeat_detector_window)))

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._pending: Optional[_OrchestratorRequest] = None
        self._request_inflight = False
        self._last_submit_s = 0.0
        self._last_probe_submit_s = 0.0
        self._probe_lock = threading.Lock()
        self._probe_cond = threading.Condition(self._probe_lock)
        self._pending_probe: Optional[_OrchestratorRequest] = None

        self._latest_decision: Optional[OrchestratorDecision] = None
        self._latest_action: Optional[ActionPlan] = None
        self._latest_action_ts_s: Optional[float] = None
        self._latest_action_primitive: Optional[str] = None
        self._latest_latency_ms: Optional[float] = None
        self._latest_reasoning: Optional[str] = None
        self._memory = MemoryManager(
            MemoryManagerConfig(
                enabled=bool(memory_enabled),
                jsonl_path=str(memory_jsonl_path),
                recent_events=max(1, int(memory_recent_events)),
                digest_items=max(1, int(memory_digest_items)),
                distill_every_n_events=max(1, int(memory_distill_every_n_events)),
            )
        )

        if self._remote_enabled:
            logger.info(
                "orchestrator remote enabled provider=%s url=%s model=%s policy_version=%s",
                self._provider,
                self._chat_url,
                self._model,
                self._policy_version,
            )
        else:
            logger.info("orchestrator remote disabled provider=%s base_url=%s", self._provider, self._chat_url)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._probe_thread: Optional[threading.Thread] = None
        if self._remote_enabled and self._reasoning_probe_enabled:
            self._probe_thread = threading.Thread(target=self._run_probe, daemon=True)
            self._probe_thread.start()

    def plan(self, st: PerceptionState) -> ActionPlan:
        now = time.monotonic()
        frame_age_ms = self._update_frame_history(now)
        active_age_s = None
        if self._latest_action_ts_s is not None:
            active_age_s = max(0.0, now - self._latest_action_ts_s)

        if (now - self._last_submit_s) >= self._orchestrator_period_s:
            frames = self._sample_frame_history()
            if len(frames) >= self._request_min_fresh_frames:
                self._memory.append_event(
                    "context_event",
                    {
                        "frames_available": len(frames),
                        "frame_age_ms": frame_age_ms,
                        "control_active_primitive": self._latest_action_primitive,
                        "control_active_age_s": active_age_s,
                    },
                )
                self._append_transcript(
                    "system",
                    (
                        f"context frames={len(frames)} frame_age_ms={frame_age_ms} "
                        f"control_active={self._latest_action_primitive or '-'} control_age_s={active_age_s}"
                    ),
                )
                with self._lock:
                    can_submit = not (self._inflight_guard_enabled and self._request_inflight)
                    if can_submit:
                        self._pending = _OrchestratorRequest(
                            state=st,
                            frames=frames,
                            frame_age_ms=frame_age_ms,
                            control_active_primitive=self._latest_action_primitive,
                            control_active_age_s=active_age_s,
                        )
                        self._last_submit_s = now
                        self._cond.notify_all()
        if (
            self._reasoning_probe_enabled
            and self._remote_enabled
            and (now - self._last_probe_submit_s) >= self._reasoning_probe_period_s
        ):
            frames = self._sample_frame_history()
            with self._probe_lock:
                self._pending_probe = _OrchestratorRequest(
                    state=st,
                    frames=frames,
                    frame_age_ms=frame_age_ms,
                    control_active_primitive=self._latest_action_primitive,
                    control_active_age_s=active_age_s,
                )
                self._last_probe_submit_s = now
                self._probe_cond.notify_all()

        with self._lock:
            action = self._latest_action
            action_ts = self._latest_action_ts_s

        if self._remote_enabled:
            # Remote-first mode: never synthesize semantic local decisions when remote is configured.
            if action is None or action_ts is None:
                return _neutral_remote_wait_action()
            if (now - action_ts) <= self._response_ttl_s:
                return action
            return action

        return self._fallback.plan(st)

    def snapshot(self) -> tuple[None, list[str], Optional[OrchestratorDecision]]:
        with self._transcript_lock:
            transcript = [entry.get("line", "") for entry in self._transcript]
        return None, transcript, self._latest_decision

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._cond.notify_all()
        with self._probe_lock:
            self._probe_cond.notify_all()
        self._thread.join(timeout=1.0)
        if self._probe_thread is not None:
            self._probe_thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                while not self._stop.is_set() and self._pending is None:
                    self._cond.wait(timeout=0.1)
                if self._stop.is_set():
                    break
                req = self._pending
                self._pending = None
                if req is not None:
                    self._request_inflight = True

            if req is None:
                continue

            try:
                if self._remote_enabled:
                    decision = self._remote_decision(req)
                    if decision is None:
                        self._memory.append_event(
                            "decision_event",
                            {
                                "source": "remote_none",
                                "state": None,
                                "intent": None,
                                "style": None,
                                "primitive": None,
                                "target_zone": None,
                                "confidence": None,
                                "allow_interrupt": None,
                                "urgency": None,
                                "zone_hint": None,
                                "latency_ms": self._latest_latency_ms,
                            },
                        )
                        self._timeline.write(
                            "fallback_event",
                            {
                                "request_id": self._request_seq,
                                "source": "remote_none",
                                "state": None,
                                "zone_hint": None,
                                "latency_ms": self._latest_latency_ms,
                                "policy_version": self._policy_version,
                            },
                        )
                        continue
                else:
                    action = self._fallback.plan(req.state)
                    decision = OrchestratorDecision(
                        target_state="idle",
                        intent="fallback_planner",
                        style=action.style,
                        primitive_hint=action.primitive.value,
                        target_zone=None,
                        allow_interrupt=action.cancel_current,
                        urgency="low",
                        confidence=action.confidence,
                        rationale=action.explanation or "fallback planner",
                        source="local",
                    )
                if self._remote_enabled:
                    action = _decision_to_action(decision)
            except Exception as exc:
                logger.warning("orchestrator planning failed: %s", exc)
                if self._remote_enabled:
                    self._memory.append_event(
                        "decision_event",
                        {
                            "source": "remote_error",
                            "state": None,
                            "intent": None,
                            "style": None,
                            "primitive": None,
                            "target_zone": None,
                            "confidence": None,
                            "allow_interrupt": None,
                            "urgency": None,
                            "zone_hint": None,
                            "latency_ms": self._latest_latency_ms,
                        },
                    )
                    self._timeline.write(
                        "fallback_event",
                        {
                            "request_id": self._request_seq,
                            "source": "remote_error",
                            "state": None,
                            "zone_hint": None,
                            "latency_ms": self._latest_latency_ms,
                            "error": str(exc),
                            "policy_version": self._policy_version,
                        },
                    )
                    continue
                action = self._fallback.plan(req.state)
                decision = OrchestratorDecision(
                    target_state="idle",
                    intent="fallback_planner",
                    style=action.style,
                    primitive_hint=action.primitive.value,
                    target_zone=None,
                    allow_interrupt=action.cancel_current,
                    urgency="low",
                    confidence=action.confidence,
                    rationale=action.explanation or "fallback planner",
                    source="local",
                )
            finally:
                with self._lock:
                    self._request_inflight = False

            self._append_transcript(
                "decision",
                (
                    f"source={decision.source} target_state={decision.target_state} intent={decision.intent} "
                    f"style={decision.style} urgency={decision.urgency} allow_interrupt={decision.allow_interrupt} "
                    f"primitive={decision.primitive_hint or 'breath'} target_zone={decision.target_zone or '-'} "
                    f"confidence={decision.confidence:.2f} frames={len(req.frames)} rationale={decision.rationale}"
                ),
            )
            self._memory.append_event(
                "decision_event",
                {
                    "source": decision.source,
                    "state": decision.target_state,
                    "intent": decision.intent,
                    "style": decision.style,
                    "primitive": decision.primitive_hint,
                    "target_zone": decision.target_zone,
                    "confidence": decision.confidence,
                    "allow_interrupt": decision.allow_interrupt,
                    "urgency": decision.urgency,
                    "zone_hint": None,
                    "latency_ms": self._latest_latency_ms,
                },
            )
            self._timeline.write(
                "decision_event",
                {
                    "request_id": self._request_seq,
                    "source": decision.source,
                    "state": decision.target_state,
                    "intent": decision.intent,
                    "style": decision.style,
                    "primitive": decision.primitive_hint,
                    "target_zone": decision.target_zone,
                    "confidence": decision.confidence,
                    "allow_interrupt": decision.allow_interrupt,
                    "urgency": decision.urgency,
                    "zone_hint": None,
                    "latency_ms": self._latest_latency_ms,
                    "policy_version": self._policy_version,
                },
            )
            self._track_repetition(decision)
            self._latest_decision = decision
            with self._lock:
                self._latest_action = action
                self._latest_action_ts_s = time.monotonic()
                self._latest_action_primitive = action.primitive.value

            now = time.monotonic()
            if now - self._last_stats_log_s >= 10.0:
                mem_stats = self._memory.stats()
                logger.info(
                    "orchestrator stats requests=%d successes=%d probe_requests=%d probe_successes=%d source=%s state=%s intent=%s style=%s memory_recent=%d memory_digest=%d",
                    self._request_count,
                    self._success_count,
                    self._probe_request_count,
                    self._probe_success_count,
                    decision.source,
                    decision.target_state,
                    decision.intent,
                    decision.style,
                    mem_stats["recent_events"],
                    mem_stats["digest_items"],
                )
                self._last_stats_log_s = now

    def _run_probe(self) -> None:
        while not self._stop.is_set():
            with self._probe_lock:
                while not self._stop.is_set() and self._pending_probe is None:
                    self._probe_cond.wait(timeout=0.1)
                if self._stop.is_set():
                    break
                req = self._pending_probe
                self._pending_probe = None

            if req is None or self._chat_url is None:
                continue

            self._probe_request_count += 1
            self._request_seq += 1
            req_id = self._request_seq
            try:
                payload = self._build_reasoning_probe_payload(req)
                t0 = time.monotonic()
                response = _post_json(
                    self._chat_url,
                    payload,
                    timeout_s=self._reasoning_probe_timeout_s,
                    api_key=self._api_key,
                )
                latency_ms = (time.monotonic() - t0) * 1000.0
                reasoning = _extract_reasoning(response)
                if not reasoning:
                    content = _extract_content(response)
                    reasoning = _extract_think_content(content) if content else None
                if not reasoning:
                    logger.debug("orchestrator probe_end id=%d status=no_reasoning latency_ms=%.1f", req_id, latency_ms)
                    continue
                self._probe_success_count += 1
                self._latest_reasoning = reasoning
                self._append_transcript("reasoning", _preview(reasoning, 240))
                logger.debug("orchestrator probe_end id=%d status=ok latency_ms=%.1f", req_id, latency_ms)
                logger.debug("orchestrator reasoning id=%d text=%s", req_id, _preview(reasoning, 600))
                self._memory.append_event(
                    "reasoning_event",
                    {
                        "request_id": req_id,
                        "latency_ms": latency_ms,
                        "reasoning": reasoning,
                    },
                )
                self._timeline.write(
                    "reasoning_event",
                    {
                        "request_id": req_id,
                        "latency_ms": latency_ms,
                        "reasoning": _preview(reasoning, 1200),
                        "source": "probe",
                        "policy_version": self._policy_version,
                    },
                )
            except Exception as exc:
                logger.debug("orchestrator probe_end id=%d status=error detail=%s", req_id, exc)

    def _update_frame_history(self, now_s: float) -> Optional[float]:
        self._prune_frame_history(now_s)
        snap = self._frame_cache.get(max_age_ms=self._max_frame_age_ms)
        if snap is None:
            return None
        frame_age_ms = (time.monotonic_ns() - snap.mono_ns) / 1_000_000.0
        if self._last_seen_frame_mono_ns == snap.mono_ns:
            return frame_age_ms
        self._last_seen_frame_mono_ns = snap.mono_ns
        self._frame_history.append((now_s, snap.mono_ns, np.asarray(snap.frame).copy()))
        self._prune_frame_history(now_s)
        return frame_age_ms

    def _prune_frame_history(self, now_s: float) -> None:
        cutoff = now_s - self._video_window_s
        while self._frame_history and self._frame_history[0][0] < cutoff:
            self._frame_history.popleft()

    def _sample_frame_history(self) -> list[np.ndarray]:
        if not self._frame_history:
            return []
        frames = [entry[2] for entry in self._frame_history]
        if len(frames) <= self._video_max_frames:
            return frames
        if self._video_max_frames == 1:
            return [frames[-1]]
        n = len(frames)
        k = self._video_max_frames
        idxs = [int(i * (n - 1) / (k - 1)) for i in range(k - 1)] + [n - 1]
        return [frames[i] for i in idxs]

    def _remote_decision(self, req: _OrchestratorRequest) -> Optional[OrchestratorDecision]:
        assert self._chat_url is not None
        self._request_seq += 1
        req_id = self._request_seq
        self._request_count += 1
        with self._transcript_lock:
            transcript_len = len(self._transcript)
        mem_stats = self._memory.stats()
        logger.debug(
            "orchestrator req_start id=%d frames=%d transcript=%d mem_recent=%d mem_digest=%d frame_age_ms=%s active_primitive=%s",
            req_id,
            len(req.frames),
            transcript_len,
            mem_stats["recent_events"],
            mem_stats["digest_items"],
            req.frame_age_ms,
            req.control_active_primitive,
        )
        self._timeline.write(
            "request_start",
            {
                "request_id": req_id,
                "frames": len(req.frames),
                "transcript_items": transcript_len,
                "memory_recent": mem_stats["recent_events"],
                "memory_digest": mem_stats["digest_items"],
                "frame_age_ms": req.frame_age_ms,
                "control_active_primitive": req.control_active_primitive,
                "control_active_age_s": req.control_active_age_s,
                "policy_version": self._policy_version,
            },
        )
        payload = self._build_payload(req)
        t0 = time.monotonic()
        response = _post_json(
            self._chat_url,
            payload,
            timeout_s=self._request_timeout_s,
            api_key=self._api_key,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0
        self._latest_latency_ms = latency_ms
        content = _extract_content(response)
        if content is None:
            logger.warning("orchestrator req_end id=%d status=no_content latency_ms=%.1f", req_id, latency_ms)
            self._timeline.write(
                "request_end",
                {
                    "request_id": req_id,
                    "status": "no_content",
                    "latency_ms": latency_ms,
                    "policy_version": self._policy_version,
                },
            )
            return None
        reasoning = _extract_reasoning(response)
        if not reasoning and content:
            reasoning = _extract_think_block(content)
        if reasoning:
            self._latest_reasoning = reasoning
            logger.debug("orchestrator reasoning id=%d text=%s", req_id, _preview(reasoning, 600))
            self._append_transcript("reasoning", _preview(reasoning, 240))
            self._memory.append_event(
                "reasoning_event",
                {
                    "request_id": req_id,
                    "reasoning": reasoning,
                },
            )
            self._timeline.write(
                "reasoning_event",
                {
                    "request_id": req_id,
                    "latency_ms": latency_ms,
                    "reasoning": _preview(reasoning, 1200),
                    "policy_version": self._policy_version,
                },
            )
        decision = _parse_decision_content(content)
        if decision is None:
            logger.warning(
                "orchestrator req_end id=%d status=parse_fail latency_ms=%.1f preview=%s",
                req_id,
                latency_ms,
                _preview(content),
            )
            self._timeline.write(
                "request_end",
                {
                    "request_id": req_id,
                    "status": "parse_fail",
                    "latency_ms": latency_ms,
                    "preview": _preview(content, 400),
                    "policy_version": self._policy_version,
                },
            )
            return None
        if not _is_valid_canonical_decision(decision):
            logger.warning(
                "orchestrator req_end id=%d status=invalid_canonical latency_ms=%.1f intent=%s primitive=%s",
                req_id,
                latency_ms,
                decision.intent,
                decision.primitive_hint,
            )
            self._timeline.write(
                "request_end",
                {
                    "request_id": req_id,
                    "status": "invalid_canonical",
                    "latency_ms": latency_ms,
                    "intent": decision.intent,
                    "primitive_hint": decision.primitive_hint,
                    "policy_version": self._policy_version,
                },
            )
            return None
        self._success_count += 1
        logger.debug(
            "orchestrator req_end id=%d status=ok latency_ms=%.1f state=%s intent=%s style=%s primitive=%s target_zone=%s",
            req_id,
            latency_ms,
            decision.target_state,
            decision.intent,
            decision.style,
            decision.primitive_hint,
            decision.target_zone,
        )
        self._timeline.write(
            "request_end",
            {
                "request_id": req_id,
                "status": "ok",
                "latency_ms": latency_ms,
                "state": decision.target_state,
                "intent": decision.intent,
                "style": decision.style,
                "primitive_hint": decision.primitive_hint,
                "target_zone": decision.target_zone,
                "policy_version": self._policy_version,
            },
        )
        return decision

    def _build_payload(self, req: _OrchestratorRequest) -> dict[str, Any]:
        transcript_tail = self._build_transcript_window()
        memory_ctx = self._memory.context()
        memory_digest = list(memory_ctx["session_memory_digest"])[-self._context_memory_digest_max_items :]
        image_data_urls, resized_shapes, total_jpeg_bytes = _encode_frames_to_data_urls(
            req.frames,
            max_width=self._video_max_width,
            jpeg_quality=self._video_jpeg_quality,
        )
        policy = {
            "version": self._policy_version,
            "identity": self._policy_identity,
            "capabilities": self._policy_capabilities,
            "safety": self._policy_safety,
            "style": self._policy_style,
            "output_contract": self._policy_output_contract,
        }
        context = {
            "policy": policy,
            "vision": {
                "frames_sent": len(image_data_urls),
                "resized_shapes": resized_shapes,
                "jpeg_total_bytes": total_jpeg_bytes,
                "frame_age_ms": req.frame_age_ms,
            },
            "control_state": {
                "active_primitive": req.control_active_primitive,
                "active_age_s": req.control_active_age_s,
                "latest_decision": None
                if self._latest_decision is None
                else {
                    "target_state": self._latest_decision.target_state,
                    "intent": self._latest_decision.intent,
                    "style": self._latest_decision.style,
                    "primitive_hint": self._latest_decision.primitive_hint,
                    "target_zone": self._latest_decision.target_zone,
                    "confidence": self._latest_decision.confidence,
                },
            },
            "perception_state_raw": _perception_state_context(req.state),
            "recent_events": memory_ctx["recent_events"],
            "session_memory_digest": memory_digest,
            "transcript_tail": transcript_tail,
        }
        user_text = (
            f"[policy_version={self._policy_version}]\n"
            f"Identity:\n{self._policy_identity}\n\n"
            f"Capabilities:\n{self._policy_capabilities}\n\n"
            f"Safety:\n{self._policy_safety}\n\n"
            f"Style:\n{self._policy_style}\n\n"
            f"Output contract:\n{self._policy_output_contract}\n\n"
            "Rely primarily on visual evidence from frames and robot context.\n"
            "Answer the request using the following format:\n"
            "<think>\n"
            "Your reasoning.\n"
            "</think>\n"
            "{\"target_state\":\"...\",\"intent\":\"...\",\"style\":\"...\",\"primitive_hint\":\"...\",\"target_zone\":\"...\",\"allow_interrupt\":true,\"urgency\":\"...\",\"confidence\":0.0,\"rationale\":\"...\"}\n\n"
            "Context JSON:\n"
            + json.dumps(context, separators=(",", ":"), ensure_ascii=True)
        )

        system_prompt = "You are a helpful assistant."
        if self._planner_prompt:
            user_text += f"\n\nOperator guidance:\n{self._planner_prompt}"

        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image_url", "image_url": {"url": u}} for u in image_data_urls],
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            "temperature": 0.1,
            "max_tokens": 700,
            "stream": False,
        }

    def _build_reasoning_probe_payload(self, req: _OrchestratorRequest) -> dict[str, Any]:
        image_data_urls, resized_shapes, total_jpeg_bytes = _encode_frames_to_data_urls(
            req.frames,
            max_width=self._video_max_width,
            jpeg_quality=self._video_jpeg_quality,
        )
        probe_context = {
            "vision": {
                "frames_sent": len(image_data_urls),
                "resized_shapes": resized_shapes,
                "jpeg_total_bytes": total_jpeg_bytes,
                "frame_age_ms": req.frame_age_ms,
            },
            "robot_state": {
                "active_primitive": req.control_active_primitive,
                "active_age_s": req.control_active_age_s,
            },
            "perception_state_raw": _perception_state_context(req.state),
        }
        user_text = (
            "Analyze the short frame sequence and context.\n"
            "Answer the question using the following format:\n"
            "<think>\n"
            "Your reasoning.\n"
            "</think>\n"
            "Write your final answer immediately after the </think> tag.\n\n"
            "Context JSON:\n"
            + json.dumps(probe_context, separators=(",", ":"), ensure_ascii=True)
        )
        system_prompt = "You are a helpful assistant."

        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image_url", "image_url": {"url": u}} for u in image_data_urls],
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            "temperature": 0.2,
            "max_tokens": self._reasoning_probe_max_tokens,
            "stream": False,
        }

    def _append_transcript(self, role: str, content: str) -> None:
        category = _normalize_transcript_role(role)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        line = f"{timestamp} {category}: {content}"
        item = {
            "ts_wall_s": time.time(),
            "ts_mono_s": time.monotonic(),
            "role": category,
            "text": str(content),
            "line": line,
        }
        with self._transcript_lock:
            self._transcript.append(item)
        logger.debug("orchestrator transcript %s", line)

    def _build_transcript_window(self) -> list[str]:
        with self._transcript_lock:
            entries = list(self._transcript)
        if not entries:
            return []

        selected: list[dict[str, Any]] = []
        per_role: dict[str, int] = {}
        for item in reversed(entries):
            role = str(item.get("role", "system"))
            count = per_role.get(role, 0)
            if count >= self._context_transcript_per_type_max_items:
                continue
            per_role[role] = count + 1
            selected.append(item)
            if len(selected) >= self._context_transcript_max_items:
                break
        selected.reverse()
        lines = [str(item.get("line", "")).strip() for item in selected if str(item.get("line", "")).strip()]
        if not lines:
            return []

        # Preserve newest lines under character budget.
        while lines and sum(len(line) + 1 for line in lines) > self._context_transcript_max_chars:
            lines.pop(0)
        return lines

    def _track_repetition(self, decision: OrchestratorDecision) -> None:
        signature = "|".join(
            [
                decision.target_state,
                decision.intent,
                decision.style,
                decision.primitive_hint or "-",
                decision.target_zone or "-",
                f"{decision.confidence:.2f}",
                decision.rationale[:120],
            ]
        )
        self._decision_signature_history.append(signature)
        if len(self._decision_signature_history) < self._decision_signature_history.maxlen:
            return

        unique_count = len(set(self._decision_signature_history))
        if unique_count == 1:
            logger.warning(
                "orchestrator repeated_decision_detected window=%d signature=%s",
                self._decision_signature_history.maxlen,
                signature,
            )


def _decision_to_action(decision: OrchestratorDecision) -> ActionPlan:
    hint = (decision.primitive_hint or "").strip().lower()
    if hint not in {"hold", "breath", "glance", "nod", "orient_to_zone"}:
        hint = _default_primitive_for_state(decision.target_state)

    zone = decision.target_zone if decision.target_zone in {"left", "center", "right"} else None

    cancel_current = bool(decision.allow_interrupt or decision.urgency == "high")
    if hint == "nod":
        return ActionPlan(
            primitive=PrimitiveKind.NOD,
            command=NodCommand(amp_rad=0.2, duration_s=0.5, cycles=1, rate_rad_s=1.8),
            confidence=decision.confidence,
            explanation=f"{decision.source}:{decision.rationale}",
            style=decision.style,
            cancel_current=cancel_current,
        )
    if hint == "glance":
        direction = "left"
        if zone == "right":
            direction = "right"
        elif zone == "left":
            direction = "left"
        elif "right" in decision.rationale:
            direction = "right"
        return ActionPlan(
            primitive=PrimitiveKind.GLANCE,
            command=GlanceCommand(direction=direction, amp_rad=0.24, duration_s=0.55, rate_rad_s=1.6),
            confidence=decision.confidence,
            explanation=f"{decision.source}:{decision.rationale}",
            style=decision.style,
            cancel_current=cancel_current,
        )
    if hint == "orient_to_zone":
        zone_value = zone or "center"
        return ActionPlan(
            primitive=PrimitiveKind.ORIENT_TO_ZONE,
            command=OrientToZoneCommand(zone=zone_value, amp_rad=0.2, rate_rad_s=1.4),
            confidence=decision.confidence,
            explanation=f"{decision.source}:{decision.rationale}",
            style=decision.style,
            cancel_current=cancel_current,
        )
    if hint == "hold":
        return ActionPlan(
            primitive=PrimitiveKind.HOLD,
            command=HoldCommand(),
            confidence=decision.confidence,
            explanation=f"{decision.source}:{decision.rationale}",
            style=decision.style,
            cancel_current=cancel_current,
        )
    return ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command=BreathCommand(amp_rad=0.08, period_s=6.5, rate_rad_s=1.0),
        confidence=decision.confidence,
        explanation=f"{decision.source}:{decision.rationale}",
        style=decision.style,
        cancel_current=cancel_current,
    )


def _neutral_remote_wait_action() -> ActionPlan:
    return ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.2,
        explanation="remote_waiting_for_first_decision",
        style="calm",
        cancel_current=False,
    )


def _default_primitive_for_state(state: str) -> str:
    defaults = {
        "idle": "breath",
        "user_detected": "orient_to_zone",
        "engaging": "breath",
        "tracking": "orient_to_zone",
        "acknowledging": "nod",
        "reacquiring": "orient_to_zone",
    }
    return defaults.get(state, "breath")


def _encode_frames_to_data_urls(
    frames: list[np.ndarray],
    *,
    max_width: int,
    jpeg_quality: int,
) -> tuple[list[str], list[list[int]], int]:
    urls: list[str] = []
    resized_shapes: list[list[int]] = []
    total_bytes = 0
    if not frames:
        return urls, resized_shapes, total_bytes

    for frame in frames:
        arr = np.asarray(frame)
        img = Image.fromarray(arr)
        if max_width > 0 and img.width > max_width:
            new_h = int(round((max_width / float(img.width)) * img.height))
            img = img.resize((max_width, max(1, new_h)))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
        jpeg = out.getvalue()
        total_bytes += len(jpeg)
        resized_shapes.append([img.height, img.width, 3])
        urls.append("data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"))
    return urls, resized_shapes, total_bytes


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
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        joined = "\n".join(parts).strip()
        return joined if joined else None
    return None


def _extract_reasoning(response: dict[str, Any]) -> Optional[str]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return None
    message = choice0.get("message")
    if not isinstance(message, dict):
        return None
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            joined = "\n".join(p for p in parts if p).strip()
            if joined:
                return joined
        if isinstance(value, dict):
            try:
                return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
            except Exception:
                continue
    return None


def _extract_think_content(content: str) -> Optional[str]:
    raw = content.strip()
    if not raw:
        return None
    start = raw.find("<think>")
    end = raw.find("</think>")
    if start >= 0 and end > start:
        think = raw[start + len("<think>") : end].strip()
        return think or None
    return raw if raw else None


def _extract_think_block(content: str) -> Optional[str]:
    raw = content.strip()
    if not raw:
        return None
    start = raw.find("<think>")
    end = raw.find("</think>")
    if start >= 0 and end > start:
        think = raw[start + len("<think>") : end].strip()
        return think or None
    return None


def _parse_decision_content(content: str) -> Optional[OrchestratorDecision]:
    cleaned = content.strip()
    data = _parse_json_obj(cleaned)
    if data is None:
        candidate = _extract_first_json_object(cleaned)
        data = None if candidate is None else _parse_json_obj(candidate)
    if data is None or not isinstance(data, dict):
        return None

    explicit_intent_present = "intent" in data
    data = _normalize_decision_payload(data)

    target_state = str(data.get("target_state", "")).strip().lower()
    intent_raw = data.get("intent")
    intent = _clean_text(intent_raw)
    if explicit_intent_present and intent is None:
        return None
    if intent is None:
        intent = _derive_intent(data)
    if intent is None:
        return None
    style = str(data.get("style", "calm")).strip().lower()
    primitive_hint = data.get("primitive_hint")
    primitive_hint_str = None if primitive_hint in (None, "") else str(primitive_hint).strip().lower()
    if primitive_hint_str not in _PRIMITIVE_HINTS:
        primitive_hint_str = None
    target_zone_raw = data.get("target_zone", data.get("zone_hint"))
    target_zone = None if target_zone_raw in (None, "") else str(target_zone_raw).strip().lower()
    if target_zone not in {None, "left", "center", "right"}:
        target_zone = None
    allow_interrupt = bool(data.get("allow_interrupt", False))
    urgency = str(data.get("urgency", "medium")).strip().lower()
    rationale = _derive_rationale(data)
    if rationale is None:
        return None
    try:
        confidence = float(data.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    if style not in _STYLES:
        style = "calm"
    if urgency not in _URGENCY:
        urgency = "medium"
    if target_state not in _TARGET_STATES:
        target_state = _infer_target_state(intent=intent, primitive_hint=primitive_hint_str)
    return OrchestratorDecision(
        target_state=target_state,
        intent=intent,
        style=style,
        primitive_hint=primitive_hint_str,
        target_zone=target_zone,
        allow_interrupt=allow_interrupt,
        urgency=urgency,
        confidence=confidence,
        rationale=rationale,
        source="remote",
    )


def _normalize_decision_payload(data: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = [data]
    for key in ("decision", "prediction", "action", "action_details"):
        nested = data.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)

    for item in candidates:
        for k, v in item.items():
            merged.setdefault(k, v)

    if isinstance(data.get("prediction"), str):
        merged.setdefault("primitive_hint", data["prediction"])
        merged.setdefault("intent", data["prediction"])
    if isinstance(data.get("primitive"), str):
        merged.setdefault("primitive_hint", data["primitive"])
    if isinstance(data.get("action"), str):
        merged.setdefault("primitive_hint", data["action"])
    if isinstance(data.get("zone"), str):
        merged.setdefault("target_zone", data["zone"])
    if isinstance(data.get("state"), str):
        merged.setdefault("target_state", data["state"])
    if isinstance(data.get("inference_confidence"), (int, float, str)):
        merged.setdefault("confidence", data["inference_confidence"])
    if isinstance(data.get("interruptible"), bool):
        merged.setdefault("allow_interrupt", data["interruptible"])
    if isinstance(data.get("should_interrupt"), bool):
        merged.setdefault("allow_interrupt", data["should_interrupt"])
    if isinstance(data.get("priority"), str):
        merged.setdefault("urgency", data["priority"])
    if isinstance(data.get("reason"), str):
        merged.setdefault("rationale", data["reason"])
    if isinstance(data.get("explanation"), str):
        merged.setdefault("rationale", data["explanation"])
    if isinstance(data.get("analysis"), str):
        merged.setdefault("rationale", data["analysis"])
    if isinstance(data.get("inference"), str):
        merged.setdefault("rationale", data["inference"])
    return merged


def _clean_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.lower() in {"none", "null", "n/a", "na"}:
        return None
    return cleaned


def _derive_intent(data: dict[str, Any]) -> Optional[str]:
    for key in ("prediction", "primitive_hint", "primitive", "action"):
        intent = _clean_text(data.get(key))
        if intent is not None:
            return intent
    return None


def _derive_rationale(data: dict[str, Any]) -> Optional[str]:
    for key in ("rationale", "reason", "explanation", "analysis", "inference"):
        rationale = _clean_text(data.get(key))
        if rationale is not None:
            return rationale
    return None


def _parse_json_obj(raw: str) -> Optional[dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _infer_target_state(*, intent: str, primitive_hint: Optional[str]) -> str:
    intent_token = intent.strip().lower()
    hint = (primitive_hint or "").strip().lower()
    if "reacquire" in intent_token:
        return "reacquiring"
    if "track" in intent_token:
        return "tracking"
    if "ack" in intent_token:
        return "acknowledging"
    if "idle" in intent_token:
        return "idle"
    if hint == "nod":
        return "acknowledging"
    if hint in {"orient_to_zone", "glance"}:
        return "tracking"
    if hint in {"hold", "breath"}:
        return "idle"
    return "user_detected"


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


def _perception_state_context(st: PerceptionState) -> dict[str, Any]:
    person = None
    if st.primary_person is not None:
        person = {
            "cx": float(st.primary_person.cx),
            "cy": float(st.primary_person.cy),
            "w": float(st.primary_person.w),
            "h": float(st.primary_person.h),
        }
    return {
        "timestamp_monotonic_s": st.timestamp_monotonic_s,
        "fps": st.fps,
        "latency_ms": st.latency_ms,
        "primary_person": person,
        "primary_person_conf": st.primary_person_conf,
        "pointing_target": None
        if st.pointing_target is None
        else {"x": float(st.pointing_target.x), "y": float(st.pointing_target.y)},
        "pointing_conf": st.pointing_conf,
        "debug": st.debug if isinstance(st.debug, dict) else {},
    }


def _normalize_transcript_role(role: str) -> str:
    token = str(role or "").strip().lower()
    if token in {"obs", "observation"}:
        return "observation"
    if token in {"decision", "action"}:
        return "decision"
    if token in {"reasoning", "think"}:
        return "reasoning"
    return "system"


def _is_valid_canonical_decision(decision: OrchestratorDecision) -> bool:
    if decision.target_state not in _TARGET_STATES:
        return False
    if decision.style not in _STYLES:
        return False
    if decision.urgency not in _URGENCY:
        return False
    if not isinstance(decision.intent, str) or not decision.intent.strip():
        return False
    if decision.primitive_hint is not None and decision.primitive_hint not in _PRIMITIVE_HINTS:
        return False
    if decision.target_zone is not None and decision.target_zone not in {"left", "center", "right"}:
        return False
    if not isinstance(decision.rationale, str) or not decision.rationale.strip():
        return False
    if not (0.0 <= float(decision.confidence) <= 1.0):
        return False
    return True


def _preview(text: str, n: int = 180) -> str:
    s = text.strip().replace("\n", "\\n")
    if len(s) <= n:
        return s
    return s[:n] + "..."
