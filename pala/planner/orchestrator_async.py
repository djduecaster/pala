from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
import io
import json
import logging
import re
import subprocess
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
from .identity import load_identity_text
from .memory_manager import MemoryManager, MemoryManagerConfig
from .protocol import PlannerInterface
from .scene_summarizer import AsyncSceneSummarizer
from .state_models import OrchestratorDecision, SceneSummary
from .timeline import TimelineConfig, TimelineWriter

logger = logging.getLogger(__name__)

_TARGET_STATES = {"idle", "user_detected", "engaging", "tracking", "acknowledging", "reacquiring"}
_STYLES = {"calm", "curious", "focused"}
_URGENCY = {"low", "medium", "high"}
_PRIMITIVE_HINTS = {"hold", "breath", "glance", "nod", "orient_to_zone"}
_ACTION_REQUIRED_FIELDS = {
    "target_state",
    "intent",
    "style",
    "primitive_hint",
    "allow_interrupt",
    "urgency",
    "confidence",
    "rationale",
}


@dataclass
class _OrchestratorRequest:
    state: PerceptionState
    frames: list[np.ndarray]
    frame_age_ms: Optional[float]
    control_active_primitive: Optional[str]
    control_active_age_s: Optional[float]
    latest_summary: Optional[SceneSummary]
    summary_age_ms: Optional[float]


@dataclass
class _FrameFetchRequest:
    reason: str


class AsyncOrchestratorPlanner(PlannerInterface):
    """Remote-first orchestrator with summary-first planning and strict action parsing."""

    owns_semantic_behavior = True

    def __init__(
        self,
        *,
        frame_cache: LatestFrameCache,
        fallback: Optional[PlannerInterface] = None,
        summarizer: Optional[AsyncSceneSummarizer] = None,
        memory_manager: Optional[MemoryManager] = None,
        timeline: Optional[TimelineWriter] = None,
        provider: str = "brev",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "nvidia/cosmos-reason2-2b",
        planner_prompt: Optional[str] = None,
        runtime_mode: str = "unknown",
        policy_version: str = "v1",
        policy_identity: str = (
            "You are PALA, a social desk companion lamp that should feel alive, expressive, and safe."
        ),
        identity_file_path: Optional[str] = "memory/identity.md",
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
        summary_ttl_ms: int = 6000,
        planner_strict_schema: bool = True,
        planner_allow_frame_fetch: bool = True,
        planner_max_tool_calls_per_cycle: int = 1,
        planner_include_images_first_pass: bool = False,
        transcript_max_items: int = 80,
        context_max_transcript_items: int = 0,
        context_transcript_max_items: int = 24,
        context_transcript_per_type_max_items: int = 8,
        context_transcript_max_chars: int = 4000,
        context_memory_digest_max_items: int = 3,
        memory_enabled: bool = True,
        memory_jsonl_path: str = "logs/orchestrator_memory.jsonl",
        memory_recent_events: int = 200,
        memory_digest_items: int = 0,
        memory_distill_every_n_events: int = 0,
        memory_recent_decisions: int = 8,
        memory_recent_summaries: int = 8,
        memory_recent_reasoning: int = 8,
        decision_repeat_detector_window: int = 6,
        timeline_jsonl_path: str = "logs/orchestrator_timeline.jsonl",
        inflight_guard_enabled: bool = True,
        request_min_fresh_frames: int = 1,
        reasoning_probe_enabled: bool = False,
        reasoning_probe_hz: float = 0.1,
        reasoning_probe_timeout_ms: int = 8000,
        reasoning_probe_max_tokens: int = 1024,
        commitment_ttl_ms: int = 12000,
        planner_self_critique_enabled: bool = True,
        planner_self_critique_confidence: float = 0.68,
        planner_self_critique_margin: float = 0.05,
        planner_self_critique_max_calls_per_cycle: int = 1,
    ) -> None:
        self._frame_cache = frame_cache
        self._fallback = fallback or HeuristicPlanner()
        self._summarizer = summarizer
        self._provider = str(provider).strip().lower()
        self._chat_url = _normalize_chat_url(base_url)
        self._api_key = api_key
        self._model = str(model or "nvidia/cosmos-reason2-2b")
        self._planner_prompt = (planner_prompt or "").strip()
        self._runtime_mode = str(runtime_mode or "unknown")
        self._policy_version = str(policy_version or "v1").strip() or "v1"
        self._policy_identity = load_identity_text(identity_file_path, policy_identity)
        self._policy_capabilities = str(policy_capabilities or "").strip()
        self._policy_safety = str(policy_safety or "").strip()
        self._policy_style = str(policy_style or "").strip()
        self._policy_output_contract = str(policy_output_contract or "").strip()
        self._max_frame_age_ms = max(50, int(max_frame_age_ms))
        self._video_window_s = max(0.1, float(video_window_s))
        self._video_max_frames = max(1, int(video_max_frames))
        self._video_max_width = max(64, int(video_max_width))
        self._video_jpeg_quality = max(1, min(100, int(video_jpeg_quality)))
        self._request_timeout_s = max(0.1, float(request_timeout_ms) / 1000.0)
        self._response_ttl_s = max(0.1, float(response_ttl_ms) / 1000.0)
        self._summary_ttl_s = max(0.1, float(summary_ttl_ms) / 1000.0)
        self._orchestrator_period_s = 1.0 / max(0.2, float(orchestrator_hz))
        self._planner_strict_schema = bool(planner_strict_schema)
        self._planner_allow_frame_fetch = bool(planner_allow_frame_fetch)
        self._planner_max_tool_calls_per_cycle = max(0, int(planner_max_tool_calls_per_cycle))
        self._planner_include_images_first_pass = bool(planner_include_images_first_pass)
        legacy_max_items = max(0, int(context_max_transcript_items))
        default_max_items = max(1, int(context_transcript_max_items))
        self._context_transcript_max_items = legacy_max_items if legacy_max_items > 0 else default_max_items
        self._context_transcript_per_type_max_items = max(1, int(context_transcript_per_type_max_items))
        self._context_transcript_max_chars = max(64, int(context_transcript_max_chars))
        self._context_memory_digest_max_items = max(1, int(context_memory_digest_max_items))
        self._memory_recent_decisions = max(1, int(memory_recent_decisions))
        self._memory_recent_summaries = max(1, int(memory_recent_summaries))
        self._memory_recent_reasoning = max(1, int(memory_recent_reasoning))

        self._timeline = timeline or TimelineWriter(
            TimelineConfig(enabled=True, jsonl_path=str(timeline_jsonl_path)),
        )
        self._memory = memory_manager or MemoryManager(
            MemoryManagerConfig(
                enabled=bool(memory_enabled),
                jsonl_path=str(memory_jsonl_path),
                recent_events=max(64, int(memory_recent_events)),
                digest_items=int(memory_digest_items),
                distill_every_n_events=int(memory_distill_every_n_events),
            )
        )
        self._inflight_guard_enabled = bool(inflight_guard_enabled)
        self._request_min_fresh_frames = max(0, int(request_min_fresh_frames))
        self._reasoning_probe_enabled = bool(reasoning_probe_enabled)
        self._reasoning_probe_period_s = 1.0 / max(0.05, float(reasoning_probe_hz))
        self._reasoning_probe_timeout_s = max(0.1, float(reasoning_probe_timeout_ms) / 1000.0)
        self._reasoning_probe_max_tokens = max(128, int(reasoning_probe_max_tokens))
        self._commitment_ttl_s = max(0.5, float(commitment_ttl_ms) / 1000.0)
        self._planner_self_critique_enabled = bool(planner_self_critique_enabled)
        self._planner_self_critique_confidence = max(0.0, min(1.0, float(planner_self_critique_confidence)))
        self._planner_self_critique_margin = max(0.0, float(planner_self_critique_margin))
        self._planner_self_critique_max_calls_per_cycle = max(0, int(planner_self_critique_max_calls_per_cycle))
        self._remote_enabled = self._provider in {"brev", "cosmos", "openai"} and self._chat_url is not None
        self._request_count = 0
        self._success_count = 0
        self._probe_request_count = 0
        self._probe_success_count = 0
        self._last_stats_log_s = 0.0
        self._request_seq = 0
        self._request_seq_lock = threading.Lock()
        self._last_stale_action_log_s = 0.0

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
        self._active_commitment: Optional[dict[str, Any]] = None
        self._active_commitment_expires_mono_s: Optional[float] = None

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
        self._timeline.write(
            "run_start",
            {
                "runtime_mode": self._runtime_mode,
                "provider": self._provider,
                "base_url": self._chat_url,
                "model": self._model,
                "policy_version": self._policy_version,
                "planner_hz": 1.0 / self._orchestrator_period_s,
                "summary_ttl_s": self._summary_ttl_s,
                "video_max_frames": self._video_max_frames,
                "frame_fetch_enabled": self._planner_allow_frame_fetch,
                "include_images_first_pass": self._planner_include_images_first_pass,
                "self_critique_enabled": self._planner_self_critique_enabled,
                "strict_schema": self._planner_strict_schema,
                "git_sha": _resolve_git_sha(),
            },
        )

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._probe_thread: Optional[threading.Thread] = None
        if self._remote_enabled and self._reasoning_probe_enabled:
            self._probe_thread = threading.Thread(target=self._run_probe, daemon=True)
            self._probe_thread.start()

    def plan(self, st: PerceptionState) -> ActionPlan:
        now = time.monotonic()
        if self._summarizer is not None:
            self._summarizer.observe(st)
        frame_age_ms = self._update_frame_history(now)
        active_age_s = None
        if self._latest_action_ts_s is not None:
            active_age_s = max(0.0, now - self._latest_action_ts_s)
        latest_summary, summary_age_ms = self._read_latest_summary(now)

        if (now - self._last_submit_s) >= self._orchestrator_period_s:
            frames = self._sample_frame_history()
            if len(frames) >= self._request_min_fresh_frames:
                with self._lock:
                    can_submit = not (self._inflight_guard_enabled and self._request_inflight)
                    if can_submit:
                        self._pending = _OrchestratorRequest(
                            state=st,
                            frames=frames,
                            frame_age_ms=frame_age_ms,
                            control_active_primitive=self._latest_action_primitive,
                            control_active_age_s=active_age_s,
                            latest_summary=latest_summary,
                            summary_age_ms=summary_age_ms,
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
                    latest_summary=latest_summary,
                    summary_age_ms=summary_age_ms,
                )
                self._last_probe_submit_s = now
                self._probe_cond.notify_all()

        with self._lock:
            action = self._latest_action
            action_ts = self._latest_action_ts_s

        if self._remote_enabled:
            if action is None or action_ts is None:
                return _neutral_remote_wait_action()
            if (now - action_ts) <= self._response_ttl_s:
                return action
            if (now - self._last_stale_action_log_s) >= 2.0:
                logger.warning(
                    "orchestrator action stale age_s=%.2f ttl_s=%.2f; issuing neutral hold",
                    now - action_ts,
                    self._response_ttl_s,
                )
                self._last_stale_action_log_s = now
            return _stale_remote_action()

        return self._fallback.plan(st)

    def snapshot(self) -> tuple[dict[str, Any], list[str], Optional[OrchestratorDecision]]:
        memory_stats = self._memory.stats()
        with self._transcript_lock:
            transcript = [entry.get("line", "") for entry in self._transcript]
        return memory_stats, transcript, self._latest_decision

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._cond.notify_all()
        with self._probe_lock:
            self._probe_cond.notify_all()
        self._thread.join(timeout=1.0)
        if self._probe_thread is not None:
            self._probe_thread.join(timeout=1.0)
        if self._summarizer is not None:
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
                if req is not None:
                    self._request_inflight = True

            if req is None:
                continue

            try:
                if self._remote_enabled:
                    decision = self._remote_decision(req)
                    if decision is None:
                        self._timeline.write(
                            "fallback_event",
                            {
                                "request_id": self._request_seq,
                                "source": "remote_none",
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
                    self._timeline.write(
                        "fallback_event",
                        {
                            "request_id": self._request_seq,
                            "source": "remote_error",
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
                    f"confidence={decision.confidence:.2f} "
                    f"utility_goal={decision.user_utility_goal or '-'} "
                    f"success_criteria={decision.success_criteria or '-'} "
                    f"commit_s={decision.commit_s if decision.commit_s is not None else '-'} "
                    f"rationale={decision.rationale}"
                ),
            )
            decision_payload = {
                "source": decision.source,
                "state": decision.target_state,
                "intent": decision.intent,
                "style": decision.style,
                "primitive": decision.primitive_hint,
                "target_zone": decision.target_zone,
                "confidence": decision.confidence,
                "allow_interrupt": decision.allow_interrupt,
                "urgency": decision.urgency,
                "user_utility_goal": decision.user_utility_goal,
                "why_now": decision.why_now,
                "success_criteria": decision.success_criteria,
                "commit_s": decision.commit_s,
                "latency_ms": self._latest_latency_ms,
            }
            self._memory.append_event("decision_event", decision_payload)
            self._timeline.write(
                "decision_event",
                {
                    "request_id": self._request_seq,
                    **decision_payload,
                    "policy_version": self._policy_version,
                },
            )
            self._track_repetition(decision)
            self._latest_decision = decision
            self._update_active_commitment(decision, time.monotonic())
            with self._lock:
                self._latest_action = action
                self._latest_action_ts_s = time.monotonic()
                self._latest_action_primitive = action.primitive.value

            now = time.monotonic()
            if now - self._last_stats_log_s >= 10.0:
                logger.info(
                    "orchestrator stats requests=%d successes=%d probe_requests=%d probe_successes=%d source=%s state=%s intent=%s style=%s",
                    self._request_count,
                    self._success_count,
                    self._probe_request_count,
                    self._probe_success_count,
                    decision.source,
                    decision.target_state,
                    decision.intent,
                    decision.style,
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
            req_id = self._next_request_id()
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
                self._memory.append_event(
                    "reasoning_event",
                    {
                        "request_id": req_id,
                        "latency_ms": latency_ms,
                        "reasoning": _preview(reasoning, 1200),
                        "source": "probe",
                    },
                )
                logger.debug("orchestrator probe_end id=%d status=ok latency_ms=%.1f", req_id, latency_ms)
            except Exception as exc:
                logger.debug("orchestrator probe_end id=%d status=error detail=%s", req_id, exc)

    def _read_latest_summary(self, now_s: float) -> tuple[Optional[SceneSummary], Optional[float]]:
        if self._summarizer is None:
            return None, None
        summary = self._summarizer.latest_summary()
        if summary is None:
            return None, None
        age_ms = max(0.0, (now_s - summary.ts_mono_s) * 1000.0)
        return summary, age_ms

    def _build_summary_window(self, now_s: float) -> tuple[list[dict[str, Any]], bool]:
        if self._summarizer is None:
            return [], False
        out: list[dict[str, Any]] = []
        stale = False
        for summary in self._summarizer.recent_summaries(self._memory_recent_summaries):
            age_ms = max(0.0, (now_s - summary.ts_mono_s) * 1000.0)
            if age_ms > (self._summary_ttl_s * 1000.0):
                stale = True
                continue
            payload = summary.to_payload()
            payload["age_ms"] = age_ms
            out.append(payload)
        return out, stale

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
        req_id = self._next_request_id()
        self._request_count += 1
        summary_state = "none"
        if req.latest_summary is not None:
            summary_state = req.latest_summary.scene_state
        logger.debug(
            "orchestrator req_start id=%d frames=%d summary=%s summary_age_ms=%s active_primitive=%s",
            req_id,
            len(req.frames),
            summary_state,
            req.summary_age_ms,
            req.control_active_primitive,
        )
        self._timeline.write(
            "request_start",
            {
                "request_id": req_id,
                "frames_available": len(req.frames),
                "summary_state": summary_state,
                "summary_age_ms": req.summary_age_ms,
                "control_active_primitive": req.control_active_primitive,
                "control_active_age_s": req.control_active_age_s,
                "policy_version": self._policy_version,
            },
        )

        t0 = time.monotonic()
        payload = self._build_payload(
            req,
            include_images=self._planner_include_images_first_pass,
            frame_fetch_reason=("first_pass" if self._planner_include_images_first_pass else None),
        )
        response = _post_json(
            self._chat_url,
            payload,
            timeout_s=self._request_timeout_s,
            api_key=self._api_key,
        )
        content = _extract_content(response)
        if content is None:
            latency_ms = (time.monotonic() - t0) * 1000.0
            self._latest_latency_ms = latency_ms
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
        self._capture_reasoning(response, req_id, latency_ms=(time.monotonic() - t0) * 1000.0)

        final_content = content
        tool_calls = 0
        while self._planner_allow_frame_fetch and tool_calls < self._planner_max_tool_calls_per_cycle:
            tool_req = _parse_frame_fetch_request(final_content)
            if tool_req is None:
                break
            if not req.frames:
                logger.warning("orchestrator frame_fetch requested but no frames available")
                break
            tool_calls += 1
            self._timeline.write(
                "frame_fetch_event",
                {
                    "request_id": req_id,
                    "tool_calls": tool_calls,
                    "reason": tool_req.reason,
                    "frames_sent": len(req.frames),
                    "policy_version": self._policy_version,
                },
            )
            payload = self._build_payload(req, include_images=True, frame_fetch_reason=tool_req.reason)
            response = _post_json(
                self._chat_url,
                payload,
                timeout_s=self._request_timeout_s,
                api_key=self._api_key,
            )
            content = _extract_content(response)
            if content is None:
                break
            self._capture_reasoning(response, req_id, latency_ms=(time.monotonic() - t0) * 1000.0)
            final_content = content

        latency_ms = (time.monotonic() - t0) * 1000.0
        self._latest_latency_ms = latency_ms
        decision = _parse_decision_content(final_content)
        if decision is None:
            logger.warning(
                "orchestrator req_end id=%d status=parse_fail latency_ms=%.1f preview=%s",
                req_id,
                latency_ms,
                _preview(final_content),
            )
            self._timeline.write(
                "request_end",
                {
                    "request_id": req_id,
                    "status": "parse_fail",
                    "latency_ms": latency_ms,
                    "preview": _preview(final_content, 400),
                    "policy_version": self._policy_version,
                },
            )
            return None
        if self._planner_strict_schema and not _is_valid_canonical_decision(decision):
            self._timeline.write(
                "request_end",
                {
                    "request_id": req_id,
                    "status": "invalid_canonical",
                    "latency_ms": latency_ms,
                    "policy_version": self._policy_version,
                },
            )
            return None

        replan_calls = 0
        while replan_calls < self._planner_self_critique_max_calls_per_cycle:
            replan_directive = self._build_self_critique_directive(req=req, decision=decision)
            if replan_directive is None:
                break
            replan_calls += 1
            self._timeline.write(
                "self_critique_event",
                {
                    "request_id": req_id,
                    "attempt": replan_calls,
                    "directive": replan_directive,
                    "policy_version": self._policy_version,
                },
            )
            payload = self._build_payload(
                req,
                include_images=False,
                frame_fetch_reason="self_critique",
                replan_directive=replan_directive,
            )
            response = _post_json(
                self._chat_url,
                payload,
                timeout_s=self._request_timeout_s,
                api_key=self._api_key,
            )
            content = _extract_content(response)
            if content is None:
                break
            self._capture_reasoning(response, req_id, latency_ms=(time.monotonic() - t0) * 1000.0)
            candidate = _parse_decision_content(content)
            if candidate is None:
                continue
            if self._planner_strict_schema and not _is_valid_canonical_decision(candidate):
                continue
            current_score = self._score_decision_for_request(req=req, decision=decision, discourage_repeat=False)
            candidate_score = self._score_decision_for_request(req=req, decision=candidate, discourage_repeat=True)
            if candidate_score >= (current_score + self._planner_self_critique_margin):
                decision = candidate
                final_content = content
            else:
                break

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
                "user_utility_goal": decision.user_utility_goal,
                "success_criteria": decision.success_criteria,
                "commit_s": decision.commit_s,
                "tool_calls": tool_calls,
                "self_critique_calls": replan_calls,
                "policy_version": self._policy_version,
            },
        )
        return decision

    def _build_self_critique_directive(
        self,
        *,
        req: _OrchestratorRequest,
        decision: OrchestratorDecision,
    ) -> Optional[str]:
        if not self._planner_self_critique_enabled:
            return None
        zone_hint = _zone_hint_from_state(req.state)
        signature_repeat = False
        if self._latest_decision is not None:
            signature_repeat = _decision_signature(self._latest_decision) == _decision_signature(decision)
        missing_utility = req.state.primary_person is not None and not (decision.user_utility_goal or "").strip()
        zone_conflict = (
            zone_hint is not None
            and req.state.primary_person is not None
            and decision.target_zone is not None
            and zone_hint != decision.target_zone
        )
        low_conf = decision.confidence < self._planner_self_critique_confidence
        if not (signature_repeat or zone_conflict or low_conf or missing_utility):
            return None
        reasons: list[str] = []
        if signature_repeat:
            reasons.append("repeated_decision")
        if zone_conflict:
            reasons.append(f"zone_conflict current={zone_hint} planned={decision.target_zone}")
        if low_conf:
            reasons.append(f"low_confidence={decision.confidence:.2f}")
        if missing_utility:
            reasons.append("missing_user_utility_goal")
        return (
            "Generate a better alternative action with clearer intent and timing. "
            "Do not repeat the same primitive+zone unless strictly necessary for safety. "
            "Include user_utility_goal, why_now, success_criteria, and commit_s when person is present. "
            f"Observed issues: {', '.join(reasons)}."
        )

    def _score_decision_for_request(
        self,
        *,
        req: _OrchestratorRequest,
        decision: OrchestratorDecision,
        discourage_repeat: bool,
    ) -> float:
        score = 0.0
        score += 0.55 * max(0.0, min(1.0, decision.confidence))
        if decision.urgency == "high":
            score += 0.25
        elif decision.urgency == "medium":
            score += 0.12
        zone_hint = _zone_hint_from_state(req.state)
        if zone_hint is not None and decision.target_zone == zone_hint:
            score += 0.22
        if req.state.primary_person is not None and decision.primitive_hint in {"orient_to_zone", "glance", "nod"}:
            score += 0.16
        if req.state.primary_person is not None and (decision.user_utility_goal or "").strip():
            score += 0.18
        if req.state.primary_person is not None and (decision.success_criteria or "").strip():
            score += 0.12
        if decision.commit_s is not None:
            score += 0.06
        if discourage_repeat and self._latest_decision is not None:
            if _decision_signature(self._latest_decision) == _decision_signature(decision):
                score -= 0.35
        return score

    def _capture_reasoning(self, response: dict[str, Any], req_id: int, *, latency_ms: float) -> None:
        reasoning = _extract_reasoning(response)
        if not reasoning:
            content = _extract_content(response)
            if content:
                reasoning = _extract_think_block(content)
        if not reasoning:
            return
        self._latest_reasoning = reasoning
        self._latest_latency_ms = latency_ms
        preview = _preview(reasoning, 1200)
        self._append_transcript("reasoning", _preview(reasoning, 240))
        self._memory.append_event(
            "reasoning_event",
            {
                "request_id": req_id,
                "latency_ms": latency_ms,
                "reasoning": preview,
                "source": "remote",
            },
        )
        self._timeline.write(
            "reasoning_event",
            {
                "request_id": req_id,
                "latency_ms": latency_ms,
                "reasoning": preview,
                "policy_version": self._policy_version,
            },
        )

    def _build_payload(
        self,
        req: _OrchestratorRequest,
        *,
        include_images: bool,
        frame_fetch_reason: Optional[str],
        replan_directive: Optional[str] = None,
    ) -> dict[str, Any]:
        now = time.monotonic()
        summary_window, summary_stale = self._build_summary_window(now)
        transcript_tail = self._build_transcript_window()
        decision_memory = self._memory.recent_payloads("decision_event", self._memory_recent_decisions)
        reasoning_memory = self._memory.recent_payloads("reasoning_event", self._memory_recent_reasoning)
        latest_summary_payload = None
        if (
            req.latest_summary is not None
            and req.summary_age_ms is not None
            and req.summary_age_ms <= (self._summary_ttl_s * 1000.0)
        ):
            latest_summary_payload = req.latest_summary.to_payload()
            latest_summary_payload["age_ms"] = req.summary_age_ms

        context = {
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
                    "user_utility_goal": self._latest_decision.user_utility_goal,
                    "success_criteria": self._latest_decision.success_criteria,
                    "commit_s": self._latest_decision.commit_s,
                },
            },
            "summary_memory": {
                "latest_summary": latest_summary_payload,
                "recent_summaries": summary_window,
                "summary_stale": summary_stale,
            },
            "memory": {
                "active_commitment": self._current_active_commitment(),
                "recent_decisions": decision_memory,
                "recent_reasoning": reasoning_memory,
                "transcript_tail": transcript_tail,
            },
            "frame_meta": {
                "frames_available": len(req.frames),
                "images_included": include_images,
                "frame_age_ms": req.frame_age_ms,
                "frame_fetch_reason": frame_fetch_reason,
            },
        }
        action_schema = {
            "target_state": "idle|user_detected|engaging|tracking|acknowledging|reacquiring",
            "intent": "string",
            "style": "calm|curious|focused",
            "primitive_hint": "hold|breath|glance|nod|orient_to_zone",
            "target_zone": "left|center|right|null",
            "allow_interrupt": "boolean",
            "urgency": "low|medium|high",
            "confidence": "number 0..1",
            "rationale": "short string",
            "act_now": "boolean (optional)",
            "user_utility_goal": "optional short string",
            "why_now": "optional short string",
            "success_criteria": "optional short string",
            "commit_s": "optional number 0.5..20",
        }
        user_text = (
            f"[policy_version={self._policy_version}]\n"
            f"Identity:\n{self._policy_identity}\n\n"
            f"Capabilities:\n{self._policy_capabilities}\n\n"
            f"Safety:\n{self._policy_safety}\n\n"
            f"Style:\n{self._policy_style}\n\n"
            "You are planning physical behavior for an embodied desk lamp.\n"
            "Use summary memory first. Request frame_fetch only when uncertainty is high or state changed too quickly.\n"
            "Do not keep holding only because previous action was hold.\n\n"
            "Decision process:\n"
            "1) Safety gate.\n"
            "2) Detect meaningful change from summary + memory.\n"
            "3) Determine user-helpful opportunity.\n"
            "4) Pick one allowed primitive.\n"
            "5) Decide if interrupt is required.\n\n"
            "Always specify a concrete user_utility_goal and success_criteria when person is present.\n"
            "If more visual detail is required, you may return exactly:\n"
            '{"tool_call":{"name":"frame_fetch","reason":"short reason"}}\n\n'
            "Otherwise return ONLY one strict JSON object matching this schema (no markdown, no extra keys):\n"
            + json.dumps(action_schema, separators=(",", ":"), ensure_ascii=True)
            + "\n\n"
            "Important: do not output <think> tags in this response. JSON only.\n\n"
            "Decision Context JSON:\n"
            + json.dumps(context, separators=(",", ":"), ensure_ascii=True)
        )
        if self._planner_prompt:
            user_text += f"\n\nOperator guidance:\n{self._planner_prompt}"
        if replan_directive:
            user_text += f"\n\nReplan directive:\n{replan_directive}"

        image_items: list[dict[str, Any]] = []
        if include_images and req.frames:
            image_data_urls, _, _ = _encode_frames_to_data_urls(
                req.frames,
                max_width=self._video_max_width,
                jpeg_quality=self._video_jpeg_quality,
            )
            image_items = [{"type": "image_url", "image_url": {"url": u}} for u in image_data_urls]

        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        *image_items,
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 700,
            "stream": False,
        }

    def _next_request_id(self) -> int:
        with self._request_seq_lock:
            self._request_seq += 1
            return self._request_seq

    def _build_reasoning_probe_payload(self, req: _OrchestratorRequest) -> dict[str, Any]:
        image_data_urls, _, _ = _encode_frames_to_data_urls(
            req.frames,
            max_width=self._video_max_width,
            jpeg_quality=self._video_jpeg_quality,
        )
        transcript_tail = self._build_transcript_window()
        probe_context = {
            "frame_meta": {
                "frames_sent": len(image_data_urls),
                "frame_age_ms": req.frame_age_ms,
            },
            "robot_state": {
                "active_primitive": req.control_active_primitive,
                "active_age_s": req.control_active_age_s,
            },
            "memory": {
                "working_memory": transcript_tail,
                "active_commitment": self._current_active_commitment(),
            },
            "summary_state": None if req.latest_summary is None else req.latest_summary.to_payload(),
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

        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
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
        memory_lines = self._memory.recent_lines(
            event_types=("decision_event", "reasoning_event"),
            limit=self._context_transcript_max_items,
            max_chars=self._context_transcript_max_chars,
        )
        if memory_lines:
            return memory_lines
        with self._transcript_lock:
            entries = list(self._transcript)
        if not entries:
            return []
        allowed_roles = {"decision", "reasoning"}
        selected: list[dict[str, Any]] = []
        per_role: dict[str, int] = {}
        for item in reversed(entries):
            role = str(item.get("role", "system"))
            if role not in allowed_roles:
                continue
            count = per_role.get(role, 0)
            if count >= self._context_transcript_per_type_max_items:
                continue
            per_role[role] = count + 1
            selected.append(item)
            if len(selected) >= self._context_transcript_max_items:
                break
        selected.reverse()
        lines = [str(item.get("line", "")).strip() for item in selected if str(item.get("line", "")).strip()]
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

    def _current_active_commitment(self) -> Optional[dict[str, Any]]:
        expires = self._active_commitment_expires_mono_s
        if self._active_commitment is None or expires is None:
            return None
        remaining = expires - time.monotonic()
        if remaining <= 0.0:
            self._active_commitment = None
            self._active_commitment_expires_mono_s = None
            return None
        out = dict(self._active_commitment)
        out["expires_in_ms"] = int(max(0.0, remaining) * 1000.0)
        return out

    def _update_active_commitment(self, decision: OrchestratorDecision, now_mono_s: float) -> None:
        self._active_commitment = {
            "target_state": decision.target_state,
            "intent": decision.intent,
            "style": decision.style,
            "primitive_hint": decision.primitive_hint,
            "target_zone": decision.target_zone,
            "confidence": decision.confidence,
            "user_utility_goal": decision.user_utility_goal,
            "success_criteria": decision.success_criteria,
            "commit_s": decision.commit_s,
        }
        commit_s = self._commitment_ttl_s if decision.commit_s is None else max(0.5, min(20.0, float(decision.commit_s)))
        self._active_commitment_expires_mono_s = now_mono_s + commit_s


def _decision_to_action(decision: OrchestratorDecision) -> ActionPlan:
    hint = (decision.primitive_hint or "").strip().lower()
    if hint not in _PRIMITIVE_HINTS:
        hint = _default_primitive_for_state(decision.target_state)

    zone = decision.target_zone if decision.target_zone in {"left", "center", "right"} else None
    cancel_current = bool(decision.allow_interrupt or decision.urgency == "high")
    explanation = _decision_explanation(decision, hint=hint, zone=zone)
    if hint == "nod":
        return ActionPlan(
            primitive=PrimitiveKind.NOD,
            command=NodCommand(amp_rad=0.2, duration_s=0.5, cycles=1, rate_rad_s=1.8),
            confidence=decision.confidence,
            explanation=explanation,
            style=decision.style,
            cancel_current=cancel_current,
        )
    if hint == "glance":
        direction = "left" if zone != "right" else "right"
        return ActionPlan(
            primitive=PrimitiveKind.GLANCE,
            command=GlanceCommand(direction=direction, amp_rad=0.24, duration_s=0.55, rate_rad_s=1.6),
            confidence=decision.confidence,
            explanation=explanation,
            style=decision.style,
            cancel_current=cancel_current,
        )
    if hint == "orient_to_zone":
        return ActionPlan(
            primitive=PrimitiveKind.ORIENT_TO_ZONE,
            command=OrientToZoneCommand(zone=zone or "center", amp_rad=0.2, rate_rad_s=1.4),
            confidence=decision.confidence,
            explanation=explanation,
            style=decision.style,
            cancel_current=cancel_current,
        )
    if hint == "hold":
        return ActionPlan(
            primitive=PrimitiveKind.HOLD,
            command=HoldCommand(),
            confidence=decision.confidence,
            explanation=explanation,
            style=decision.style,
            cancel_current=cancel_current,
        )
    return ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command=BreathCommand(amp_rad=0.08, period_s=6.5, rate_rad_s=1.0),
        confidence=decision.confidence,
        explanation=explanation,
        style=decision.style,
        cancel_current=cancel_current,
    )


def _decision_explanation(decision: OrchestratorDecision, *, hint: str, zone: Optional[str]) -> str:
    rationale = " ".join((decision.rationale or "").strip().split())
    utility_goal = (decision.user_utility_goal or "").strip() or "-"
    why_now = (decision.why_now or "").strip() or "-"
    success_criteria = (decision.success_criteria or "").strip() or "-"
    commit_s = "-" if decision.commit_s is None else f"{float(decision.commit_s):.2f}"
    return (
        f"source={decision.source} "
        f"target_state={decision.target_state} "
        f"intent={decision.intent} "
        f"primitive={hint} "
        f"zone={zone or 'none'} "
        f"urgency={decision.urgency} "
        f"allow_interrupt={int(bool(decision.allow_interrupt))} "
        f"confidence={decision.confidence:.2f} "
        f"user_utility_goal={utility_goal} "
        f"why_now={why_now} "
        f"success_criteria={success_criteria} "
        f"commit_s={commit_s} "
        f"rationale={rationale}"
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


def _stale_remote_action() -> ActionPlan:
    return ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.2,
        explanation="remote_action_stale_hold",
        style="calm",
        cancel_current=True,
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


def _is_valid_canonical_decision(decision: OrchestratorDecision) -> bool:
    if decision.target_state not in _TARGET_STATES:
        return False
    if decision.style not in _STYLES:
        return False
    if decision.urgency not in _URGENCY:
        return False
    if decision.primitive_hint not in _PRIMITIVE_HINTS:
        return False
    if decision.target_zone not in {None, "left", "center", "right"}:
        return False
    return True


def _parse_frame_fetch_request(content: str) -> Optional[_FrameFetchRequest]:
    data = _parse_json_content(content)
    if data is None:
        return None
    tool_call = data.get("tool_call")
    if isinstance(tool_call, dict):
        name = _clean_text(tool_call.get("name"))
        if name and name.lower() == "frame_fetch":
            reason = _clean_text(tool_call.get("reason")) or "visual uncertainty"
            return _FrameFetchRequest(reason=reason)
    tool = _clean_text(data.get("tool"))
    if tool and tool.lower() == "frame_fetch":
        reason = _clean_text(data.get("reason")) or "visual uncertainty"
        return _FrameFetchRequest(reason=reason)
    return None


def _parse_decision_content(content: str) -> Optional[OrchestratorDecision]:
    data = _parse_json_content(content)
    if data is None:
        return None

    candidates = _decision_candidate_maps(data)
    target_state_raw = _first_clean_text(candidates, "target_state", "state")
    intent_raw = _first_clean_text(candidates, "intent", "action_intent", "goal", "prediction")
    style_raw = _first_clean_text(candidates, "style", "mood", "tone")
    primitive_raw = _first_clean_text(candidates, "primitive_hint", "primitive", "action", "prediction")
    urgency_raw = _first_clean_text(candidates, "urgency", "priority")
    rationale_raw = _first_clean_text(candidates, "rationale", "inference", "analysis", "reason", "explanation")
    zone_raw = _first_clean_text(candidates, "target_zone", "zone", "zone_hint", "direction")
    utility_goal_raw = _first_clean_text(candidates, "user_utility_goal", "utility_goal", "goal")
    why_now_raw = _first_clean_text(candidates, "why_now", "justification", "why")
    success_criteria_raw = _first_clean_text(
        candidates,
        "success_criteria",
        "success_check",
        "next_check",
        "expected_outcome",
    )
    commit_raw = _first_value(candidates, "commit_s", "commit_seconds", "horizon_s")

    primitive_hint = _normalize_primitive_hint(primitive_raw, intent=intent_raw, rationale=rationale_raw)
    target_state = _normalize_target_state(target_state_raw, primitive_hint=primitive_hint, intent=intent_raw)
    style = _normalize_style(style_raw, primitive_hint=primitive_hint)
    urgency = _normalize_urgency(urgency_raw, primitive_hint=primitive_hint)
    intent = _normalize_intent(intent_raw, primitive_hint=primitive_hint, target_state=target_state)
    if intent.strip().lower() in {"none", "null", "n/a", "na"}:
        intent = _default_intent_for_state(target_state)

    if target_state not in _TARGET_STATES:
        return None
    if style not in _STYLES:
        return None
    if primitive_hint not in _PRIMITIVE_HINTS:
        return None
    if urgency not in _URGENCY:
        return None

    allow_interrupt_value = _coerce_bool_value(
        _first_value(candidates, "allow_interrupt", "cancel_current", "interrupt"),
        default=primitive_hint in {"glance", "nod", "orient_to_zone"},
    )
    if allow_interrupt_value is None:
        return None
    confidence = _coerce_confidence(
        _first_value(candidates, "confidence", "inference_confidence", "score", "probability"),
        default=(0.72 if primitive_hint in {"glance", "nod", "orient_to_zone"} else 0.6),
    )

    target_zone = _normalize_target_zone(zone_raw, rationale=rationale_raw)
    commit_s = _normalize_commit_s(commit_raw)

    rationale = rationale_raw or (
        f"normalized decision target_state={target_state} primitive={primitive_hint} style={style} urgency={urgency}"
    )

    act_now = _coerce_bool_value(_first_value(candidates, "act_now", "execute_now"), default=True)
    if act_now is None:
        return None
    if not act_now:
        primitive_hint = "hold"
        allow_interrupt_value = False
        target_state = "idle"

    return OrchestratorDecision(
        target_state=target_state,
        intent=intent,
        style=style,
        primitive_hint=primitive_hint,
        target_zone=target_zone,
        allow_interrupt=allow_interrupt_value,
        urgency=urgency,
        confidence=confidence,
        rationale=rationale,
        source="remote",
        user_utility_goal=utility_goal_raw,
        why_now=why_now_raw,
        success_criteria=success_criteria_raw,
        commit_s=commit_s,
    )


def _decision_candidate_maps(data: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [data]
    for key in ("selected_action", "prediction", "action_details", "decision", "action", "output", "result"):
        value = data.get(key)
        if isinstance(value, dict):
            candidates.append(value)
            for nested_key in ("selected_action", "prediction", "action_details", "decision", "action"):
                nested = value.get(nested_key)
                if isinstance(nested, dict):
                    candidates.append(nested)
    return candidates


def _first_value(candidates: list[dict[str, Any]], *keys: str) -> Any:
    for item in candidates:
        for key in keys:
            if key in item:
                return item.get(key)
    return None


def _first_clean_text(candidates: list[dict[str, Any]], *keys: str) -> Optional[str]:
    value = _first_value(candidates, *keys)
    if isinstance(value, str):
        return _clean_text(value)
    return None


def _coerce_confidence(value: Any, *, default: float) -> float:
    if value is None:
        return max(0.0, min(1.0, float(default)))
    try:
        out = float(value)
    except Exception:
        out = default
    return max(0.0, min(1.0, out))


def _normalize_target_state(
    raw: Optional[str],
    *,
    primitive_hint: str,
    intent: Optional[str],
) -> str:
    if raw:
        token = raw.strip().lower()
        mapping = {
            "idle": "idle",
            "waiting": "idle",
            "user_detected": "user_detected",
            "detected": "user_detected",
            "engage": "engaging",
            "engaging": "engaging",
            "assist": "engaging",
            "ack": "acknowledging",
            "acknowledging": "acknowledging",
            "track": "tracking",
            "tracking": "tracking",
            "reacquire": "reacquiring",
            "reacquiring": "reacquiring",
        }
        normalized = mapping.get(token)
        if normalized in _TARGET_STATES:
            return normalized
    text = f"{intent or ''} {primitive_hint}".lower()
    if _contains_any(text, "reacquire", "search", "lost"):
        return "reacquiring"
    if _contains_any(text, "ack", "greet", "welcome", "nod"):
        return "acknowledging"
    if _contains_any(text, "assist", "help", "task", "engage"):
        return "engaging"
    if primitive_hint in {"orient_to_zone", "glance"}:
        return "tracking"
    if primitive_hint in {"hold", "breath"}:
        return "idle"
    return "tracking"


def _normalize_primitive_hint(raw: Optional[str], *, intent: Optional[str], rationale: Optional[str]) -> str:
    text = " ".join(
        part for part in (raw or "", intent or "", rationale or "") if part
    ).strip().lower()
    if not text:
        return "orient_to_zone"
    if text in _PRIMITIVE_HINTS:
        return text
    mapping = {
        "reach_for_object": "orient_to_zone",
        "look_at_user": "orient_to_zone",
        "track_transition": "orient_to_zone",
        "track_user": "orient_to_zone",
        "assist_with_task": "orient_to_zone",
        "search": "glance",
        "scan": "glance",
        "greet": "nod",
        "acknowledge": "nod",
    }
    if text in mapping:
        return mapping[text]
    if _contains_any(text, "nod", "ack", "greet", "welcome", "hello"):
        return "nod"
    if _contains_any(text, "glance", "scan", "search", "reacquire"):
        return "glance"
    if _contains_any(text, "hold", "wait", "pause", "still"):
        return "hold"
    if _contains_any(text, "breath", "idle", "calm"):
        return "breath"
    if _contains_any(text, "orient", "track", "follow", "turn", "reach", "assist", "look", "focus"):
        return "orient_to_zone"
    return "orient_to_zone"


def _normalize_style(raw: Optional[str], *, primitive_hint: str) -> str:
    if raw:
        token = raw.strip().lower()
        if token in _STYLES:
            return token
        if token in {"attentive", "intentional", "working"}:
            return "focused"
        if token in {"friendly", "playful", "expressive"}:
            return "curious"
        if token in {"gentle", "neutral", "still"}:
            return "calm"
    if primitive_hint in {"hold", "breath"}:
        return "calm"
    if primitive_hint in {"glance", "nod"}:
        return "curious"
    return "focused"


def _normalize_urgency(raw: Optional[str], *, primitive_hint: str) -> str:
    if raw:
        token = raw.strip().lower()
        if token in _URGENCY:
            return token
        if token in {"immediate", "critical", "urgent", "now"}:
            return "high"
        if token in {"soon", "prompt"}:
            return "medium"
        if token in {"gentle", "background", "later"}:
            return "low"
    if primitive_hint in {"hold", "breath"}:
        return "low"
    if primitive_hint in {"nod", "glance"}:
        return "medium"
    return "medium"


def _normalize_target_zone(raw: Optional[str], *, rationale: Optional[str]) -> Optional[str]:
    if raw:
        token = raw.strip().lower()
        if token in {"left", "center", "right"}:
            return token
        if token in {"l", "west"}:
            return "left"
        if token in {"r", "east"}:
            return "right"
        if token in {"middle", "centre"}:
            return "center"
    text = (rationale or "").strip().lower()
    for token in ("left", "center", "right"):
        if token in text:
            return token
    return None


def _normalize_commit_s(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return max(0.5, min(20.0, value))


def _normalize_intent(raw: Optional[str], *, primitive_hint: str, target_state: str) -> str:
    cleaned = _clean_text(raw) if raw is not None else None
    if cleaned is not None:
        token = cleaned.strip().lower()
        if token not in {"none", "null", "n/a", "na"}:
            return cleaned
    return _default_intent_for_state(target_state if target_state in _TARGET_STATES else _state_for_primitive(primitive_hint))


def _default_intent_for_state(state: str) -> str:
    defaults = {
        "idle": "idle_presence",
        "user_detected": "user_detected",
        "engaging": "assist_with_task",
        "tracking": "track_transition",
        "acknowledging": "acknowledge_presence",
        "reacquiring": "reacquire_user",
    }
    return defaults.get(state, "track_transition")


def _state_for_primitive(primitive_hint: str) -> str:
    mapping = {
        "hold": "idle",
        "breath": "idle",
        "nod": "acknowledging",
        "glance": "reacquiring",
        "orient_to_zone": "tracking",
    }
    return mapping.get(primitive_hint, "tracking")


def _zone_hint_from_state(st: PerceptionState) -> Optional[str]:
    if isinstance(st.debug, dict):
        raw = st.debug.get("zone_hint")
        if isinstance(raw, str):
            token = raw.strip().lower()
            if token in {"left", "center", "right"}:
                return token
    if st.primary_person is None:
        return None
    cx = float(st.primary_person.cx)
    if cx < 0.33:
        return "left"
    if cx < 0.66:
        return "center"
    return "right"


def _decision_signature(decision: OrchestratorDecision) -> str:
    return "|".join(
        [
            decision.target_state,
            decision.intent,
            decision.style,
            decision.primitive_hint or "-",
            decision.target_zone or "-",
            f"{decision.confidence:.2f}",
        ],
    )


def _contains_any(text: str, *tokens: str) -> bool:
    lowered = text.lower()
    for token in tokens:
        pattern = rf"\b{re.escape(token.lower())}\b"
        if re.search(pattern, lowered):
            return True
    return False


def _parse_json_content(content: str) -> Optional[dict[str, Any]]:
    cleaned = _strip_markdown_fences(content.strip())
    data = _parse_json_obj(cleaned)
    if data is not None:
        return data
    candidate = _extract_first_json_object(cleaned)
    if candidate is None:
        return None
    return _parse_json_obj(candidate)


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


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    if lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_obj(raw: str) -> Optional[dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_bool_value(value: Any, *, default: bool) -> Optional[bool]:
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
        if token in {"false", "0", "no", "n", "off", "", "none", "null"}:
            return False
    return None


def _clean_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.lower() in {"none", "null", "n/a", "na"}:
        return None
    return cleaned


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


def _normalize_transcript_role(role: str) -> str:
    token = str(role or "").strip().lower()
    if token in {"decision", "reasoning"}:
        return token
    return "system"


def _preview(raw: str, max_chars: int = 200) -> str:
    text = raw.strip().replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _resolve_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None
    return out if out else None
