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
from .state_models import ObservationPacket, InteractionBelief, OrchestratorDecision, SceneSummary

logger = logging.getLogger(__name__)

_TARGET_STATES = {"idle", "user_detected", "engaging", "tracking", "acknowledging", "reacquiring"}
_STYLES = {"calm", "curious", "focused"}
_URGENCY = {"low", "medium", "high"}


@dataclass
class _OrchestratorRequest:
    observation: ObservationPacket
    belief: InteractionBelief
    frames: list[np.ndarray]


class AsyncOrchestratorPlanner(PlannerInterface):
    """Async orchestrator with transcript context and latest-only remote requests."""

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
        memory_enabled: bool = True,
        memory_jsonl_path: str = "logs/orchestrator_memory.jsonl",
        memory_recent_events: int = 10,
        memory_digest_items: int = 3,
        memory_distill_every_n_events: int = 20,
        decision_repeat_detector_window: int = 6,
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
        self._max_frame_age_ms = max(50, int(max_frame_age_ms))
        self._video_window_s = max(0.1, float(video_window_s))
        self._video_max_frames = max(1, int(video_max_frames))
        self._video_max_width = max(64, int(video_max_width))
        self._video_jpeg_quality = max(1, min(100, int(video_jpeg_quality)))
        self._reacquire_timeout_s = 3.0
        self._request_timeout_s = max(0.05, float(request_timeout_ms) / 1000.0)
        self._response_ttl_s = max(0.1, float(response_ttl_ms) / 1000.0)
        self._orchestrator_period_s = 1.0 / max(0.2, float(orchestrator_hz))
        self._context_max_transcript_items = max(0, int(context_max_transcript_items))
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
        self._transcript: deque[str] = deque(maxlen=max(10, int(transcript_max_items)))
        self._zone_history: deque[str] = deque(maxlen=6)
        self._frame_history: deque[tuple[float, int, np.ndarray]] = deque()
        self._zone_transition_times: deque[float] = deque()
        self._last_seen_frame_mono_ns: Optional[int] = None
        self._last_person_seen_s: Optional[float] = None
        self._last_person_zone: Optional[str] = None
        self._last_zone_hint: Optional[str] = None
        self._last_zone_change_s: float = time.monotonic()
        self._belief_state: str = "idle"
        self._belief_since_s: float = time.monotonic()
        self._decision_signature_history: deque[str] = deque(maxlen=max(2, int(decision_repeat_detector_window)))
        self._last_decision_zone: Optional[str] = None

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._pending: Optional[_OrchestratorRequest] = None
        self._last_submit_s = 0.0
        self._last_probe_submit_s = 0.0
        self._probe_lock = threading.Lock()
        self._probe_cond = threading.Condition(self._probe_lock)
        self._pending_probe: Optional[_OrchestratorRequest] = None

        self._latest_decision: Optional[OrchestratorDecision] = None
        self._latest_action: Optional[ActionPlan] = None
        self._latest_action_ts_s: Optional[float] = None
        self._latest_action_primitive: Optional[str] = None
        self._latest_summary: Optional[SceneSummary] = None
        self._latest_observation: Optional[ObservationPacket] = None
        self._latest_belief: Optional[InteractionBelief] = None
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
                "orchestrator remote enabled provider=%s url=%s model=%s",
                self._provider,
                self._chat_url,
                self._model,
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
        summary = self._summarize_state(st, frame_age_ms)
        observation = self._build_observation(summary, now)
        belief = self._update_belief(observation, now)
        self._latest_summary = summary
        self._latest_observation = observation
        self._latest_belief = belief

        if (now - self._last_submit_s) >= self._orchestrator_period_s:
            frames = self._sample_frame_history()
            self._memory.append_event(
                "observation_event",
                {
                    "state": belief.state,
                    "zone_hint": observation.zone_hint,
                    "person_present": observation.person_present,
                    "activity_hint": observation.activity_hint,
                    "uncertainty_flags": observation.uncertainty_flags,
                    "zone_transitions_recent": observation.zone_transitions_recent,
                    "control_active_primitive": observation.control_active_primitive,
                },
            )
            with self._lock:
                self._pending = _OrchestratorRequest(
                    observation=observation,
                    belief=belief,
                    frames=frames,
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
                    observation=observation,
                    belief=belief,
                    frames=frames,
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

        decision = _local_decision(observation, belief)
        logger.debug(
            "orchestrator fallback source=local target_state=%s intent=%s style=%s zone=%s",
            decision.target_state,
            decision.intent,
            decision.style,
            decision.target_zone,
        )
        return _decision_to_action(decision, observation)

    def snapshot(self) -> tuple[Optional[SceneSummary], list[str], Optional[OrchestratorDecision]]:
        with self._transcript_lock:
            transcript = list(self._transcript)
        return self._latest_summary, transcript, self._latest_decision

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
                                "state": req.belief.state,
                                "intent": None,
                                "style": None,
                                "primitive": None,
                                "target_zone": None,
                                "confidence": None,
                                "allow_interrupt": None,
                                "urgency": None,
                                "zone_hint": req.observation.zone_hint,
                                "latency_ms": self._latest_latency_ms,
                            },
                        )
                        continue
                else:
                    decision = _local_decision(req.observation, req.belief)
                action = _decision_to_action(decision, req.observation)
            except Exception as exc:
                logger.warning("orchestrator planning failed: %s", exc)
                if self._remote_enabled:
                    self._memory.append_event(
                        "decision_event",
                        {
                            "source": "remote_error",
                            "state": req.belief.state,
                            "intent": None,
                            "style": None,
                            "primitive": None,
                            "target_zone": None,
                            "confidence": None,
                            "allow_interrupt": None,
                            "urgency": None,
                            "zone_hint": req.observation.zone_hint,
                            "latency_ms": self._latest_latency_ms,
                        },
                    )
                    continue
                decision = _local_decision(req.observation, req.belief)
                action = _decision_to_action(decision, req.observation)

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
                    "zone_hint": req.observation.zone_hint,
                    "latency_ms": self._latest_latency_ms,
                },
            )
            self._track_repetition(decision, req.observation)
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

    def _summarize_state(self, st: PerceptionState, frame_age_ms: Optional[float]) -> SceneSummary:
        zone = None
        if isinstance(st.debug, dict):
            raw_zone = st.debug.get("zone_hint")
            if isinstance(raw_zone, str) and raw_zone:
                zone = raw_zone
        if zone is not None:
            now = time.monotonic()
            if self._last_zone_hint is not None and self._last_zone_hint != zone:
                self._zone_transition_times.append(now)
                self._last_zone_change_s = now
            elif self._last_zone_hint is None:
                self._last_zone_change_s = now
            self._last_zone_hint = zone
            self._zone_history.append(zone)

        person_present = st.primary_person is not None
        if person_present:
            self._last_person_seen_s = time.monotonic()
            if zone is not None:
                self._last_person_zone = zone
        uncertainty_flags: list[str] = []
        conf = st.primary_person_conf
        if person_present and conf is not None and conf < 0.5:
            uncertainty_flags.append("low_person_conf")

        if frame_age_ms is not None and frame_age_ms > 250.0:
            uncertainty_flags.append("stale_frame")

        activity_hint = self._infer_activity_hint(person_present, zone)
        return SceneSummary(
            timestamp_monotonic_s=time.monotonic(),
            person_present=person_present,
            zone_hint=zone,
            primary_person_conf=conf,
            activity_hint=activity_hint,
            uncertainty_flags=uncertainty_flags,
            frame_age_ms=frame_age_ms,
        )

    def _infer_activity_hint(self, person_present: bool, zone: Optional[str]) -> Optional[str]:
        if not person_present:
            return "away"
        if len(self._zone_history) < 3:
            return "engaged"
        recent = list(self._zone_history)[-3:]
        if len(set(recent)) >= 2:
            return "transitioning"
        if zone == "center":
            return "focused_work"
        return "engaged"

    def _build_observation(self, summary: SceneSummary, now_s: float) -> ObservationPacket:
        transition_cutoff = now_s - 3.0
        while self._zone_transition_times and self._zone_transition_times[0] < transition_cutoff:
            self._zone_transition_times.popleft()
        zone_stable_s = max(0.0, now_s - self._last_zone_change_s) if summary.zone_hint is not None else 0.0
        active_age_s = None
        if self._latest_action_ts_s is not None:
            active_age_s = max(0.0, now_s - self._latest_action_ts_s)
        return ObservationPacket(
            timestamp_monotonic_s=now_s,
            person_present=summary.person_present,
            zone_hint=summary.zone_hint,
            primary_person_conf=summary.primary_person_conf,
            frame_age_ms=summary.frame_age_ms,
            activity_hint=summary.activity_hint,
            uncertainty_flags=list(summary.uncertainty_flags),
            zone_stable_s=zone_stable_s,
            zone_transitions_recent=len(self._zone_transition_times),
            control_active_primitive=self._latest_action_primitive,
            control_active_age_s=active_age_s,
        )

    def _update_belief(self, observation: ObservationPacket, now_s: float) -> InteractionBelief:
        recent_absence = (
            not observation.person_present
            and self._last_person_seen_s is not None
            and (now_s - self._last_person_seen_s) <= self._reacquire_timeout_s
        )
        if not observation.person_present:
            if recent_absence:
                next_state = "reacquiring"
                confidence = 0.55
                reason = "person recently visible"
            else:
                next_state = "idle"
                confidence = 0.7
                reason = "person absent"
        elif observation.zone_transitions_recent >= 2:
            next_state = "tracking"
            confidence = 0.7
            reason = "recent zone transitions"
        elif observation.zone_hint == "center" and observation.zone_stable_s >= 1.2:
            next_state = "engaging"
            confidence = 0.74
            reason = "stable centered engagement"
        else:
            next_state = "user_detected"
            confidence = 0.64
            reason = "person visible"

        if next_state != self._belief_state:
            self._belief_state = next_state
            self._belief_since_s = now_s

        return InteractionBelief(
            timestamp_monotonic_s=now_s,
            state=self._belief_state,
            confidence=confidence,
            last_seen_zone=self._last_person_zone,
            person_last_seen_s=self._last_person_seen_s,
            reason=reason,
            uncertainty_flags=list(observation.uncertainty_flags),
        )

    def _remote_decision(self, req: _OrchestratorRequest) -> Optional[OrchestratorDecision]:
        assert self._chat_url is not None
        self._request_seq += 1
        req_id = self._request_seq
        self._request_count += 1
        with self._transcript_lock:
            transcript_len = len(self._transcript)
        mem_stats = self._memory.stats()
        logger.debug(
            "orchestrator req_start id=%d frames=%d transcript=%d mem_recent=%d mem_digest=%d zone=%s person=%s belief=%s",
            req_id,
            len(req.frames),
            transcript_len,
            mem_stats["recent_events"],
            mem_stats["digest_items"],
            req.observation.zone_hint,
            req.observation.person_present,
            req.belief.state,
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
            return None
        reasoning = _extract_reasoning(response)
        if reasoning:
            self._latest_reasoning = reasoning
            logger.debug("orchestrator reasoning id=%d text=%s", req_id, _preview(reasoning, 600))
            self._memory.append_event(
                "reasoning_event",
                {
                    "request_id": req_id,
                    "reasoning": reasoning,
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
        return decision

    def _build_payload(self, req: _OrchestratorRequest) -> dict[str, Any]:
        with self._transcript_lock:
            transcript_tail = (
                list(self._transcript)[-self._context_max_transcript_items :]
                if self._context_max_transcript_items > 0
                else []
            )
        memory_ctx = self._memory.context()
        image_data_urls, resized_shapes, total_jpeg_bytes = _encode_frames_to_data_urls(
            req.frames,
            max_width=self._video_max_width,
            jpeg_quality=self._video_jpeg_quality,
        )
        context = {
            "observation": {
                "person_present": req.observation.person_present,
                "zone_hint": req.observation.zone_hint,
                "primary_person_conf": req.observation.primary_person_conf,
                "frame_age_ms": req.observation.frame_age_ms,
                "activity_hint": req.observation.activity_hint,
                "uncertainty_flags": req.observation.uncertainty_flags,
                "zone_stable_s": req.observation.zone_stable_s,
                "zone_transitions_recent": req.observation.zone_transitions_recent,
                "control_active_primitive": req.observation.control_active_primitive,
                "control_active_age_s": req.observation.control_active_age_s,
            },
            "belief": {
                "state": req.belief.state,
                "confidence": req.belief.confidence,
                "last_seen_zone": req.belief.last_seen_zone,
                "person_last_seen_s": req.belief.person_last_seen_s,
                "reason": req.belief.reason,
                "uncertainty_flags": req.belief.uncertainty_flags,
            },
            "recent_events": memory_ctx["recent_events"],
            "session_memory_digest": memory_ctx["session_memory_digest"],
            "transcript_tail": transcript_tail,
            "frames_sent": len(image_data_urls),
            "resized_shapes": resized_shapes,
            "jpeg_total_bytes": total_jpeg_bytes,
        }
        user_text = "Use the provided context and frame sequence to decide next action.\n" + json.dumps(
            context,
            separators=(",", ":"),
            ensure_ascii=True,
        )

        system_prompt = (
            "You are a policy orchestrator for a social desk robot lamp. "
            "You may receive multiple frames in chronological order. "
            "Return JSON only with keys: target_state, intent, style, primitive_hint, target_zone, allow_interrupt, urgency, confidence, rationale. "
            "target_state must be one of ['idle','user_detected','engaging','tracking','acknowledging','reacquiring']. "
            "style must be one of ['calm','curious','focused']. "
            "primitive_hint should be one of ['hold','breath','glance','nod','orient_to_zone'] or null. "
            "target_zone should be one of ['left','center','right'] or null. "
            "urgency must be one of ['low','medium','high']. "
            "allow_interrupt is boolean. confidence in [0,1]. Keep rationale concise. "
            "If person_present=true and zone_hint is left/right with low uncertainty, do not default to center. "
            "Include observation_zone_used and policy_reason in the rationale text."
        )
        if self._planner_prompt:
            system_prompt += f" Operator guidance: {self._planner_prompt}"

        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        *[{"type": "image_url", "image_url": {"url": u}} for u in image_data_urls],
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 180,
            "stream": False,
        }

    def _build_reasoning_probe_payload(self, req: _OrchestratorRequest) -> dict[str, Any]:
        image_data_urls, resized_shapes, total_jpeg_bytes = _encode_frames_to_data_urls(
            req.frames,
            max_width=self._video_max_width,
            jpeg_quality=self._video_jpeg_quality,
        )
        probe_context = {
            "observation": {
                "person_present": req.observation.person_present,
                "zone_hint": req.observation.zone_hint,
                "primary_person_conf": req.observation.primary_person_conf,
                "activity_hint": req.observation.activity_hint,
                "uncertainty_flags": req.observation.uncertainty_flags,
                "zone_stable_s": req.observation.zone_stable_s,
                "zone_transitions_recent": req.observation.zone_transitions_recent,
            },
            "belief": {
                "state": req.belief.state,
                "confidence": req.belief.confidence,
                "reason": req.belief.reason,
            },
            "frames_sent": len(image_data_urls),
            "resized_shapes": resized_shapes,
            "jpeg_total_bytes": total_jpeg_bytes,
        }
        user_text = "Analyze the short frame sequence and context.\n" + json.dumps(
            probe_context,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        system_prompt = (
            "You are analyzing a social robot lamp scene over time. "
            "Answer in this exact format:\n"
            "<think>\n"
            "your reasoning\n"
            "</think>\n\n"
            "<answer>\n"
            "short concise conclusion\n"
            "</answer>"
        )

        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        *[{"type": "image_url", "image_url": {"url": u}} for u in image_data_urls],
                    ],
                },
            ],
            "temperature": 0.2,
            "max_tokens": self._reasoning_probe_max_tokens,
            "stream": False,
        }

    def _append_transcript(self, role: str, content: str) -> None:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        with self._transcript_lock:
            line = f"{timestamp} {role}: {content}"
            self._transcript.append(line)
        logger.debug("orchestrator transcript %s", line)

    def _track_repetition(self, decision: OrchestratorDecision, observation: ObservationPacket) -> None:
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
            self._last_decision_zone = observation.zone_hint
            return

        unique_count = len(set(self._decision_signature_history))
        zone_changed = (
            self._last_decision_zone is not None
            and observation.zone_hint is not None
            and self._last_decision_zone != observation.zone_hint
        )
        if unique_count == 1 and zone_changed:
            logger.warning(
                "orchestrator repeated_decision_detected window=%d zone_prev=%s zone_now=%s",
                self._decision_signature_history.maxlen,
                self._last_decision_zone,
                observation.zone_hint,
            )
        self._last_decision_zone = observation.zone_hint


def _local_decision(observation: ObservationPacket, belief: InteractionBelief) -> OrchestratorDecision:
    if belief.state == "reacquiring":
        zone = belief.last_seen_zone if belief.last_seen_zone in {"left", "center", "right"} else "center"
        return OrchestratorDecision(
            target_state="reacquiring",
            intent="reacquire_attention",
            style="focused",
            primitive_hint="orient_to_zone",
            target_zone=zone,
            allow_interrupt=True,
            urgency="medium",
            confidence=0.56,
            rationale="person recently visible, attempt brief reacquire",
            source="local",
        )
    if belief.state == "idle":
        return OrchestratorDecision(
            target_state="idle",
            intent="idle_presence",
            style="calm",
            primitive_hint="breath",
            target_zone=None,
            allow_interrupt=False,
            urgency="low",
            confidence=0.62,
            rationale="no person present",
            source="local",
        )
    if belief.state == "tracking":
        return OrchestratorDecision(
            target_state="tracking",
            intent="track_transition",
            style="curious",
            primitive_hint="orient_to_zone",
            target_zone=observation.zone_hint,
            allow_interrupt=True,
            urgency="medium",
            confidence=0.72,
            rationale="recent zone transitions indicate movement",
            source="local",
        )
    if belief.state == "engaging":
        return OrchestratorDecision(
            target_state="engaging",
            intent="engaged_focus",
            style="focused",
            primitive_hint="nod",
            target_zone=observation.zone_hint,
            allow_interrupt=False,
            urgency="low",
            confidence=0.72,
            rationale="stable centered engagement",
            source="local",
        )
    return OrchestratorDecision(
        target_state="user_detected",
        intent="maintain_presence",
        style="curious",
        primitive_hint="orient_to_zone",
        target_zone=observation.zone_hint,
        allow_interrupt=True,
        urgency="medium",
        confidence=0.66,
        rationale="person present, orient gently",
        source="local",
    )


def _decision_to_action(decision: OrchestratorDecision, observation: ObservationPacket) -> ActionPlan:
    hint = (decision.primitive_hint or "").strip().lower()
    if hint not in {"hold", "breath", "glance", "nod", "orient_to_zone"}:
        hint = _default_primitive_for_state(decision.target_state)

    zone = decision.target_zone if decision.target_zone in {"left", "center", "right"} else None
    zone_hint = observation.zone_hint if observation.zone_hint in {"left", "center", "right"} else None

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
        zone_value = zone or zone_hint or "center"
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


def _preview(text: str, n: int = 180) -> str:
    s = text.strip().replace("\n", "\\n")
    if len(s) <= n:
        return s
    return s[:n] + "..."
