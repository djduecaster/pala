from __future__ import annotations

import base64
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import io
import logging
import time
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
from PIL import Image

from ..types import ActionPlan, HoldCommand, PerceptionState, PrimitiveKind
from ..utils import maybe_logger
from .action_compiler import ActionCompiler
from .arbiter import Arbiter, ArbiterConfig
from .context_builder import ContextBuilder
from .env_summarizer import EnvSummarizer
from .frame_window import RollingFrameWindow
from .governor import Governor
from .health_manager import HealthManager
from .idle_engine import IdleEngine, IdleEngineConfig
from .intent_proposer import IntentProposer
from .prompts import build_env_user_text, build_messages, build_planner_user_text
from .remote_api import RemoteCallResult, extract_message_content, normalize_chat_url, post_chat_json
from .schemas import env_response_format, intent_response_format
from .trace_bus import TraceBus
from .types import ProposalCandidate, ProposerResponse
from .world_state_store import DecisionSnapshot, EnvironmentSnapshot, WorldStateStore

logger = logging.getLogger(__name__)


@dataclass
class BehaviorPolicyConfig:
    fallback_style: str = "calm"
    fallback_confidence: float = 0.1
    persist_every_step: bool = False

    remote_enabled: bool = False
    base_url: Optional[str] = None
    remote_provider: str = "auto"
    api_key: Optional[str] = None
    model: str = "nvidia/cosmos-reason2-2b"

    request_timeout_ms: int = 20000
    error_backoff_s: float = 1.5
    client_error_backoff_s: float = 5.0

    env_hz: float = 0.25
    planner_hz: float = 0.6
    planner_event_delta_threshold: float = 0.65
    planner_event_cooldown_s: float = 0.7

    max_frame_age_ms: int = 500
    frame_window_s: float = 6.0
    env_max_frames: int = 4
    planner_max_frames: int = 1
    frame_max_width: int = 320
    frame_jpeg_quality: int = 60
    request_min_fresh_frames: int = 1
    planner_include_latest_frame: bool = True

    env_max_tokens: int = 600
    planner_max_tokens: int = 480

    proposer_max_age_s: float = 10.0
    planner_max_proposals: int = 2
    planner_use_env_context: bool = True

    arbiter_min_dwell_s: float = 1.2
    arbiter_base_margin: float = 0.05
    arbiter_takeover_no_signal_streak: int = 2
    arbiter_takeover_no_commit_s: float = 2.8

    idle_after_s: float = 0.0
    idle_glance_after_s: float = 0.8

    policy_identity: str = "You are PALA, a social desk companion lamp."
    policy_capabilities: str = "Use available primitives for expressive lamp motion."
    policy_safety: str = "Prioritize safe, smooth, non-aggressive motion."
    policy_style: str = "Use calm by default."
    planner_prompt: str = ""

    env_log_path: Optional[str] = "logs/behavior_env.jsonl"
    planner_log_path: Optional[str] = "logs/behavior_planner.jsonl"
    reasoning_log_path: Optional[str] = "logs/behavior_reasoning.jsonl"
    trace_log_path: Optional[str] = "logs/behavior_trace.jsonl"


@dataclass
class _InFlightCall:
    request_id: int
    started_mono_s: float
    payload: Mapping[str, Any]
    future: Future[RemoteCallResult]


class BehaviorPolicy:
    """Remote-first behavior policy with deterministic arbitration and strict JSON contracts."""

    owns_semantic_behavior = True

    def __init__(
        self,
        *,
        world_state: Optional[WorldStateStore] = None,
        config: Optional[BehaviorPolicyConfig] = None,
        clock: Optional[Callable[[], float]] = None,
        frame_cache: Optional[Any] = None,
        dwell_s: float = 2.0,
        cooldown_s: float = 1.0,
        max_hold_s: float = 2.0,
    ):
        _ = (dwell_s, cooldown_s, max_hold_s)
        self._cfg = config or BehaviorPolicyConfig()
        self._world_state = world_state or WorldStateStore()
        self._clock = clock or time.monotonic
        self._frame_cache = frame_cache
        self._frame_window = RollingFrameWindow(window_s=self._cfg.frame_window_s)
        self._last_ingested_frame_ns: Optional[int] = None

        self._context_builder = ContextBuilder()
        self._env_summarizer = EnvSummarizer()
        self._intent_proposer = IntentProposer()
        self._governor = Governor()
        self._arbiter = Arbiter(
            ArbiterConfig(
                min_dwell_s=self._cfg.arbiter_min_dwell_s,
                base_margin=self._cfg.arbiter_base_margin,
                idle_after_s=self._cfg.idle_after_s,
                takeover_no_signal_streak=max(1, int(self._cfg.arbiter_takeover_no_signal_streak)),
                takeover_no_commit_s=max(0.2, float(self._cfg.arbiter_takeover_no_commit_s)),
            )
        )
        self._idle_engine = IdleEngine(
            IdleEngineConfig(
                idle_after_s=self._cfg.idle_after_s,
                glance_after_s=self._cfg.idle_glance_after_s,
            )
        )
        self._compiler = ActionCompiler()
        self._health = HealthManager()

        self._chat_url = normalize_chat_url(self._cfg.base_url or "", provider=self._cfg.remote_provider)
        self._remote_enabled = bool(self._cfg.remote_enabled and self._chat_url)
        self._executor: Optional[ThreadPoolExecutor] = None
        if self._remote_enabled:
            # Keep planner responsive if env summarization stalls.
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="behavior_remote")

        self._env_inflight: Optional[_InFlightCall] = None
        self._planner_inflight: Optional[_InFlightCall] = None
        self._last_env_submit_s = 0.0
        self._last_planner_submit_s = 0.0
        self._last_planner_event_submit_s = 0.0
        self._next_env_allowed_s = 0.0
        self._next_planner_allowed_s = 0.0
        self._pending_planner_event = False
        self._env_request_seq = 0
        self._planner_request_seq = 0

        self._latest_remote_proposals: Optional[ProposerResponse] = None
        self._latest_remote_wall_s: float = 0.0

        self._current_action: ActionPlan = self._new_hold_action("startup_hold")
        self._last_action_commit_s = self._clock()
        self._last_commit_intent = "idle_presence"
        self._last_commit_utility = 0.2
        self._recent_commit_times: deque[float] = deque(maxlen=32)
        self._last_valid_zone_hint: Optional[str] = None
        self._idle_tick = 0

        self._env_log = maybe_logger(self._cfg.env_log_path) if self._cfg.env_log_path else None
        self._planner_log = maybe_logger(self._cfg.planner_log_path) if self._cfg.planner_log_path else None
        self._reasoning_log = maybe_logger(self._cfg.reasoning_log_path) if self._cfg.reasoning_log_path else None
        self._trace = TraceBus(maybe_logger(self._cfg.trace_log_path) if self._cfg.trace_log_path else None)

    @property
    def world_state(self) -> WorldStateStore:
        return self._world_state

    def set_control_state(self, control_state: Any) -> None:
        self._world_state.set_control_state(control_state)

    def shutdown(self) -> None:
        for logger_obj in (self._env_log, self._planner_log, self._reasoning_log):
            if logger_obj is None:
                continue
            try:
                logger_obj.close()
            except Exception:  # noqa: BLE001
                continue
        self._trace.close()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def step(self, st: Optional[PerceptionState]) -> ActionPlan:
        now = self._clock()
        self._idle_tick += 1

        self._ingest_latest_frame()
        self._drain_env_inflight(st=st, now=now)
        self._drain_planner_inflight(st=st, now=now)

        if self._remote_enabled:
            self._maybe_schedule_env(st=st, now=now)
            self._maybe_schedule_planner(st=st, now=now)

        chosen_source = "none"
        arb_reason = "keep_current"
        committed = False
        commit_error = None

        no_commit_s = max(0.0, now - self._last_action_commit_s)
        self._update_perception_health(st=st)
        snap = self._world_state.snapshot()

        candidates: list[ProposalCandidate] = []
        remote = self._latest_remote_candidates()
        candidates.extend(remote)
        zone_hint = self._zone_hint(st=st, snapshot=snap)
        idle_candidates = [
            ProposalCandidate(proposal=item, source="idle_engine")
            for item in self._idle_engine.propose(
                no_commit_s=no_commit_s,
                zone_hint=zone_hint,
                tick_index=self._idle_tick,
            )
        ]
        candidates.extend(idle_candidates)

        governed = self._governor.evaluate(candidates)
        arbiter = self._arbiter.select(
            candidates=governed,
            current_action=self._current_action,
            current_utility=self._last_commit_utility,
            action_age_s=max(0.0, now - self._last_action_commit_s),
            no_commit_s=no_commit_s,
            last_intent=self._last_commit_intent,
            recent_switches=self._recent_switch_count(now),
            planner_open_breaker=self._health.planner_open_breaker(),
            planner_no_signal_streak=self._health.planner_no_signal_streak(),
            perception_degraded=self._health.perception_degraded(),
        )

        if arbiter.decision == "commit" and arbiter.chosen is not None:
            proposal = arbiter.chosen.candidate.proposal
            compiled = self._compiler.compile(proposal)
            if compiled.action is None:
                commit_error = compiled.error or "compile_failed"
            else:
                chosen_source = arbiter.chosen.candidate.source
                self._set_current_action(
                    compiled.action,
                    rationale=proposal.rationale_short,
                    intent=proposal.intent,
                    utility=arbiter.best_utility,
                    now=now,
                )
                committed = True

        arb_reason = arbiter.reason if commit_error is None else commit_error

        if self._cfg.persist_every_step:
            self._world_state.persist()

        self._trace.emit(
            {
                "ts_wall_s": time.time(),
                "current_action": {
                    "primitive": self._current_action.primitive.value,
                    "style": self._current_action.style,
                    "confidence": self._current_action.confidence,
                },
                "health": {
                    "planner": self._health.planner.as_dict(),
                    "env": self._health.env.as_dict(),
                    "perception": self._health.perception.as_dict(),
                },
                "decision": {
                    "committed": committed,
                    "reason": arb_reason,
                    "source": chosen_source,
                    "best_utility": arbiter.best_utility,
                },
                "signals": {
                    "no_commit_s": no_commit_s,
                    "zone_hint": zone_hint,
                    "env_delta": _as_float((snap.get("latest_env_snapshot") or {}).get("delta_score"), default=0.0),
                    "detector_alive": _debug_get(st, "detector_alive"),
                    "source_alive": _debug_get(st, "source_alive"),
                },
                "candidate_count": len(governed),
            }
        )

        return self._current_action

    def _ingest_latest_frame(self) -> None:
        if self._frame_cache is None:
            return
        try:
            snap = self._frame_cache.get(max_age_ms=self._cfg.max_frame_age_ms)
        except Exception:  # noqa: BLE001
            return
        if snap is None:
            self._frame_window.prune()
            return
        mono_ns = int(snap.mono_ns)
        if self._last_ingested_frame_ns is not None and mono_ns == self._last_ingested_frame_ns:
            self._frame_window.prune(now_ns=mono_ns)
            return
        self._last_ingested_frame_ns = mono_ns
        self._frame_window.add_frame(np.asarray(snap.frame), mono_ns=mono_ns)

    def _latest_remote_candidates(self) -> list[ProposalCandidate]:
        response = self._latest_remote_proposals
        if response is None:
            return []
        age = time.time() - self._latest_remote_wall_s
        if age > max(0.2, float(self._cfg.proposer_max_age_s)):
            return []
        limited = response.proposals[: max(1, int(self._cfg.planner_max_proposals))]
        return [ProposalCandidate(proposal=item, source="remote") for item in limited]

    def _maybe_schedule_env(self, *, st: Optional[PerceptionState], now: float) -> None:
        if self._executor is None:
            return
        if float(self._cfg.env_hz) <= 0.0:
            return
        if now < self._next_env_allowed_s:
            return
        period_s = 1.0 / max(0.05, float(self._cfg.env_hz))
        if (now - self._last_env_submit_s) < period_s:
            return

        # Planner is the primary low-latency control path; avoid concurrent env calls on the same endpoint.
        if self._planner_inflight is not None:
            self._env_summarizer.mark_pending({"queued": True, "ts_mono_s": now})
            return
        if self._env_inflight is not None:
            self._env_summarizer.mark_pending({"queued": True, "ts_mono_s": now})
            return

        payload = self._build_env_payload(st=st)
        if payload is None:
            return
        accepted = self._env_summarizer.submit_or_replace(payload)
        if not accepted:
            return

        self._env_request_seq += 1
        req_id = self._env_request_seq
        self._last_env_submit_s = now

        future = self._executor.submit(
            post_chat_json,
            url=self._chat_url,
            payload=payload["body"],
            timeout_s=self._request_timeout_s(),
            api_key=self._cfg.api_key,
            provider=self._cfg.remote_provider,
        )
        self._env_inflight = _InFlightCall(req_id, now, payload, future)
        self._write_log(
            self._env_log,
            {
                "ts_wall_s": time.time(),
                "request_id": req_id,
                "phase": "env_processor",
                "status": "req_start",
                "frames": payload["frames"],
                "response_format": "json_schema",
            },
        )

    def _maybe_schedule_planner(self, *, st: Optional[PerceptionState], now: float) -> None:
        if self._executor is None:
            return
        if now < self._next_planner_allowed_s:
            return

        planner_hz = self._health.planner_effective_hz(self._cfg.planner_hz)
        period_s = 1.0 / max(0.05, planner_hz)
        due_periodic = (now - self._last_planner_submit_s) >= period_s
        due_event = self._pending_planner_event and (
            (now - self._last_planner_event_submit_s) >= max(0.05, self._cfg.planner_event_cooldown_s)
        )
        if not due_periodic and not due_event:
            return
        if self._planner_inflight is not None:
            self._intent_proposer.mark_pending({"queued": True, "ts_mono_s": now})
            return

        payload = self._build_planner_payload(st=st, now=now)
        if payload is None:
            return
        accepted = self._intent_proposer.submit_or_replace(payload)
        if not accepted:
            return

        self._planner_request_seq += 1
        req_id = self._planner_request_seq
        self._last_planner_submit_s = now
        if due_event:
            self._last_planner_event_submit_s = now
            self._pending_planner_event = False

        future = self._executor.submit(
            post_chat_json,
            url=self._chat_url,
            payload=payload["body"],
            timeout_s=self._request_timeout_s(),
            api_key=self._cfg.api_key,
            provider=self._cfg.remote_provider,
        )
        self._planner_inflight = _InFlightCall(req_id, now, payload, future)
        self._write_log(
            self._planner_log,
            {
                "ts_wall_s": time.time(),
                "request_id": req_id,
                "phase": "planner",
                "status": "req_start",
                "frames": payload["frames"],
                "env_context": bool(self._cfg.planner_use_env_context),
                "response_format": "json_schema",
            },
        )

    def _drain_env_inflight(self, *, st: Optional[PerceptionState], now: float) -> None:
        call = self._env_inflight
        if call is None:
            return

        if not call.future.done():
            if not self._watchdog_expired(call=call, now=now):
                return
            self._env_inflight = None
            self._cancel_future(call.future)
            result = self._synthetic_transport_error(
                now=now,
                started_mono_s=call.started_mono_s,
                error="watchdog_timeout",
            )
        else:
            self._env_inflight = None
            result = self._safe_future_result(call=call, now=now)

        status = "transport_error"
        error = result.error
        summary = None
        delta_score = None
        zone_hint = None
        person_present = None
        content_text = None
        reasoning_text = None
        finish_reason = None
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        parse_stage = "none"

        if result.ok and result.response_json is not None:
            finish_reason, prompt_tokens, completion_tokens, total_tokens = _response_meta(result.response_json)
            content_text, reasoning_text = extract_message_content(result.response_json)
            if content_text is None:
                status = "empty_content"
                error = "missing_message_content"
                self._env_summarizer.complete_request("")
                parse_stage = self._env_summarizer.last_parse_stage
            else:
                parsed = self._env_summarizer.complete_request(content_text)
                parse_stage = self._env_summarizer.last_parse_stage
                if parsed is None:
                    status = "parse_fail"
                    detail = self._env_summarizer.last_parse_error or "unknown"
                    error = f"env_json_parse_failed:{detail}"
                else:
                    status = "ok"
                    error = None
                    summary = parsed.summary.summary_short
                    delta_score = parsed.summary.delta_score
                    zone_hint = parsed.summary.features.get("zone_hint")
                    person_present = bool(parsed.summary.features.get("person_present", False))
                    self._world_state.update_environment(
                        EnvironmentSnapshot(
                            scene=parsed.summary.scene,
                            events=parsed.summary.events,
                            hypotheses=parsed.summary.hypotheses,
                            summary=parsed.summary.summary_short,
                            delta_score=parsed.summary.delta_score,
                            features=dict(parsed.summary.features),
                        )
                    )
                    if parsed.summary.delta_score >= self._cfg.planner_event_delta_threshold:
                        self._pending_planner_event = True
        else:
            self._env_summarizer.complete_request("")
            parse_stage = self._env_summarizer.last_parse_stage

        if status == "ok":
            self._next_env_allowed_s = 0.0
        else:
            self._next_env_allowed_s = max(self._next_env_allowed_s, now + self._compute_failure_backoff_s(result))

        self._health.on_env_result(status=status, latency_ms=result.latency_ms)

        self._write_log(
            self._env_log,
            {
                "ts_wall_s": time.time(),
                "request_id": call.request_id,
                "phase": "env_processor",
                "status": status,
                "latency_ms": round(result.latency_ms, 1),
                "error": error,
                "response_preview": None if content_text is None else self._preview_text(content_text),
                "parse_stage": parse_stage,
                "delta_score": delta_score,
                "summary": summary,
                "zone_hint": zone_hint,
                "person_present": person_present,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )
        if status == "ok":
            logger.info(
                "env req=%s ok latency_ms=%.1f delta=%.2f person=%s zone=%s summary=%s",
                call.request_id,
                round(result.latency_ms, 1),
                float(delta_score or 0.0),
                person_present,
                zone_hint,
                self._preview_text(summary or "", max_chars=140),
            )
        else:
            logger.info(
                "env req=%s status=%s latency_ms=%.1f error=%s",
                call.request_id,
                status,
                round(result.latency_ms, 1),
                error,
            )

        if reasoning_text:
            self._write_log(
                self._reasoning_log,
                {
                    "ts_wall_s": time.time(),
                    "request_id": call.request_id,
                    "component": "env_processor",
                    "latency_ms": round(result.latency_ms, 1),
                    "reasoning": reasoning_text,
                },
            )

        pending = self._env_summarizer.take_latest_pending()
        if pending is not None:
            self._last_env_submit_s = now - (1.0 / max(0.05, float(self._cfg.env_hz)))
            self._maybe_schedule_env(st=st, now=now)

    def _drain_planner_inflight(self, *, st: Optional[PerceptionState], now: float) -> None:
        call = self._planner_inflight
        if call is None:
            return

        if not call.future.done():
            if not self._watchdog_expired(call=call, now=now):
                return
            self._planner_inflight = None
            self._cancel_future(call.future)
            result = self._synthetic_transport_error(
                now=now,
                started_mono_s=call.started_mono_s,
                error="watchdog_timeout",
            )
        else:
            self._planner_inflight = None
            result = self._safe_future_result(call=call, now=now)

        status = "transport_error"
        error = result.error
        content_text = None
        reasoning_text = None
        parsed_response: Optional[ProposerResponse] = None
        finish_reason = None
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        parse_stage = "none"

        if result.ok and result.response_json is not None:
            finish_reason, prompt_tokens, completion_tokens, total_tokens = _response_meta(result.response_json)
            content_text, reasoning_text = extract_message_content(result.response_json)
            if content_text is None:
                status = "empty_content"
                error = "missing_message_content"
                self._intent_proposer.complete_request("")
                parse_stage = self._intent_proposer.last_parse_stage
            else:
                parsed = self._intent_proposer.complete_request(content_text)
                parse_stage = self._intent_proposer.last_parse_stage
                if parsed is None:
                    status = "parse_fail"
                    detail = self._intent_proposer.last_parse_error or "unknown"
                    error = f"planner_json_parse_failed:{detail}"
                else:
                    status = "ok"
                    error = None
                    parsed_response = parsed.response
                    parsed_response = ProposerResponse(
                        schema_version=parsed_response.schema_version,
                        proposals=parsed_response.proposals[: max(1, int(self._cfg.planner_max_proposals))],
                        notes_short=parsed_response.notes_short,
                    )
                    self._latest_remote_proposals = parsed_response
                    self._latest_remote_wall_s = time.time()
        else:
            self._intent_proposer.complete_request("")
            parse_stage = self._intent_proposer.last_parse_stage

        if status == "ok":
            self._next_planner_allowed_s = 0.0
        else:
            self._next_planner_allowed_s = max(
                self._next_planner_allowed_s,
                now + self._compute_failure_backoff_s(result),
            )

        self._health.on_planner_result(status=status, latency_ms=result.latency_ms, response=parsed_response)

        top = None
        if parsed_response and parsed_response.proposals:
            item = parsed_response.proposals[0]
            top = {
                "intent": item.intent,
                "primitive": item.primitive,
                "score": item.score,
                "confidence": item.confidence,
            }

        self._write_log(
            self._planner_log,
            {
                "ts_wall_s": time.time(),
                "request_id": call.request_id,
                "phase": "planner",
                "status": status,
                "latency_ms": round(result.latency_ms, 1),
                "error": error,
                "response_preview": None if content_text is None else self._preview_text(content_text),
                "parse_stage": parse_stage,
                "proposal_count": 0 if parsed_response is None else len(parsed_response.proposals),
                "top_proposal": top,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )
        if status == "ok":
            primitive = None if top is None else top.get("primitive")
            intent = None if top is None else top.get("intent")
            score = None if top is None else top.get("score")
            confidence = None if top is None else top.get("confidence")
            logger.info(
                "planner req=%s ok latency_ms=%.1f proposals=%s top=%s/%s score=%s conf=%s response=%s",
                call.request_id,
                round(result.latency_ms, 1),
                0 if parsed_response is None else len(parsed_response.proposals),
                intent,
                primitive,
                None if score is None else round(float(score), 3),
                None if confidence is None else round(float(confidence), 3),
                self._preview_text(content_text or "", max_chars=180),
            )
        else:
            logger.info(
                "planner req=%s status=%s latency_ms=%.1f error=%s response=%s",
                call.request_id,
                status,
                round(result.latency_ms, 1),
                error,
                None if content_text is None else self._preview_text(content_text, max_chars=180),
            )

        if reasoning_text:
            self._write_log(
                self._reasoning_log,
                {
                    "ts_wall_s": time.time(),
                    "request_id": call.request_id,
                    "component": "planner",
                    "latency_ms": round(result.latency_ms, 1),
                    "reasoning": reasoning_text,
                },
            )

        pending = self._intent_proposer.take_latest_pending()
        if pending is not None:
            planner_hz = self._health.planner_effective_hz(self._cfg.planner_hz)
            self._last_planner_submit_s = now - (1.0 / max(0.05, planner_hz))
            self._maybe_schedule_planner(st=st, now=now)

    def _build_env_payload(self, *, st: Optional[PerceptionState]) -> Optional[Dict[str, Any]]:
        frames = self._frame_window.sample(max_frames=self._cfg.env_max_frames)
        if len(frames) < max(1, int(self._cfg.request_min_fresh_frames)):
            return None

        newest_ns = max(item.mono_ns for item in frames)
        frame_timeline = [
            {
                "ordinal": idx + 1,
                "age_s": round(max(0.0, (newest_ns - item.mono_ns) / 1_000_000_000.0), 3),
            }
            for idx, item in enumerate(frames)
        ]

        image_data_urls = [
            _encode_frame_data_url(
                frame=item.frame,
                max_width=self._cfg.frame_max_width,
                jpeg_quality=self._cfg.frame_jpeg_quality,
            )
            for item in frames
        ]

        context = self._context_builder.build_env_context(
            world_snapshot=self._world_state.snapshot(),
            current_action=self._current_action,
            frame_timeline=frame_timeline,
        )
        user_text = build_env_user_text(context=context, policy_identity=self._cfg.policy_identity)

        body: Dict[str, Any] = {
            "model": self._cfg.model,
            "messages": build_messages(user_text=user_text, image_data_urls=image_data_urls),
            "temperature": 0.0,
            "top_p": 0.3,
            "presence_penalty": 0.0,
            "max_tokens": int(self._cfg.env_max_tokens),
            "stream": False,
            "response_format": env_response_format(provider=self._cfg.remote_provider),
        }

        return {"body": body, "frames": len(frames)}

    def _build_planner_payload(self, *, st: Optional[PerceptionState], now: float) -> Optional[Dict[str, Any]]:
        image_data_urls = []
        if self._cfg.planner_include_latest_frame:
            latest = self._frame_window.latest()
            if latest is not None:
                image_data_urls.append(
                    _encode_frame_data_url(
                        frame=latest.frame,
                        max_width=self._cfg.frame_max_width,
                        jpeg_quality=self._cfg.frame_jpeg_quality,
                    )
                )

        snap = self._world_state.snapshot()
        if not bool(self._cfg.planner_use_env_context):
            snap = dict(snap)
            snap["latest_env_snapshot"] = {}
            snap["event_tail"] = []
        no_commit_s = max(0.0, now - self._last_action_commit_s)
        context = self._context_builder.build_planner_context(
            st=st,
            world_snapshot=snap,
            current_action=self._current_action,
            planner_health=self._health.planner.as_dict(),
            now_mono_s=now,
            last_commit_mono_s=self._last_action_commit_s,
            no_commit_s=no_commit_s,
        )
        user_text = build_planner_user_text(
            context=context,
            policy_identity=self._cfg.policy_identity,
            policy_capabilities=self._cfg.policy_capabilities,
            policy_safety=self._cfg.policy_safety,
            policy_style=self._cfg.policy_style,
            planner_prompt=self._cfg.planner_prompt,
            max_proposals=max(1, int(self._cfg.planner_max_proposals)),
        )

        body: Dict[str, Any] = {
            "model": self._cfg.model,
            "messages": build_messages(
                user_text=user_text,
                image_data_urls=image_data_urls[: max(0, int(self._cfg.planner_max_frames))],
            ),
            "temperature": 0.0,
            "top_p": 0.3,
            "presence_penalty": 0.0,
            "max_tokens": int(self._cfg.planner_max_tokens),
            "stream": False,
            "response_format": intent_response_format(provider=self._cfg.remote_provider),
        }

        return {"body": body, "frames": len(image_data_urls)}

    def _set_current_action(
        self,
        action: ActionPlan,
        *,
        rationale: str,
        intent: str,
        utility: float,
        now: float,
    ) -> None:
        self._current_action = action
        self._last_action_commit_s = now
        self._last_commit_intent = intent
        self._last_commit_utility = max(0.0, min(1.5, float(utility)))
        self._recent_commit_times.append(now)

        self._world_state.append_decision(
            DecisionSnapshot(
                primitive=action.primitive.value,
                style=action.style,
                confidence=action.confidence,
                rationale_short=(rationale or action.explanation or "").strip()[:220],
            )
        )

    def _recent_switch_count(self, now: float) -> int:
        window_s = 8.0
        while self._recent_commit_times and (now - self._recent_commit_times[0]) > window_s:
            self._recent_commit_times.popleft()
        return len(self._recent_commit_times)

    def _zone_hint(self, *, st: Optional[PerceptionState], snapshot: Mapping[str, Any]) -> str:
        if st is not None and st.debug:
            zone = str(st.debug.get("zone_hint", "")).strip().lower()
            if zone in {"left", "center", "right"}:
                self._last_valid_zone_hint = zone
                return zone
        latest_env = snapshot.get("latest_env_snapshot") or {}
        features = latest_env.get("features") or {}
        zone = str(features.get("zone_hint", "")).strip().lower()
        if zone in {"left", "center", "right"}:
            self._last_valid_zone_hint = zone
            return zone
        inferred = _infer_zone_hint_from_text(
            latest_env.get("scene"),
            latest_env.get("events"),
            latest_env.get("summary"),
        )
        if inferred in {"left", "center", "right"}:
            self._last_valid_zone_hint = inferred
            return inferred
        if self._last_valid_zone_hint in {"left", "center", "right"}:
            return self._last_valid_zone_hint
        return "unknown"

    def _update_perception_health(self, *, st: Optional[PerceptionState]) -> None:
        if st is None or not isinstance(st.debug, Mapping):
            return
        detector_raw = st.debug.get("detector_alive")
        source_raw = st.debug.get("source_alive")
        stale = bool(st.debug.get("stale_frame", False))
        detector_alive = detector_raw if isinstance(detector_raw, bool) else None
        source_alive = source_raw if isinstance(source_raw, bool) else None
        if detector_alive is None and source_alive is None and not stale:
            return
        self._health.on_perception_result(
            detector_alive=detector_alive,
            source_alive=source_alive,
            stale_frame=stale,
        )

    def _same_action_signature(self, a: ActionPlan, b: ActionPlan) -> bool:
        return a.primitive == b.primitive and a.command == b.command and a.style == b.style

    def _new_hold_action(self, reason: str) -> ActionPlan:
        return ActionPlan(
            primitive=PrimitiveKind.HOLD,
            command=HoldCommand(),
            confidence=self._cfg.fallback_confidence,
            style=self._cfg.fallback_style,
            cancel_current=False,
            explanation=reason,
        )

    def _compute_failure_backoff_s(self, result: RemoteCallResult) -> float:
        base = max(0.0, float(self._cfg.error_backoff_s))
        if 400 <= int(result.status_code) < 500:
            return max(base, float(self._cfg.client_error_backoff_s))
        return base

    def _request_timeout_s(self) -> float:
        return max(0.25, self._cfg.request_timeout_ms / 1000.0)

    def _watchdog_timeout_s(self) -> float:
        # Guard against futures that never transition to done().
        return self._request_timeout_s() + 0.75

    def _watchdog_expired(self, *, call: _InFlightCall, now: float) -> bool:
        return (now - call.started_mono_s) > self._watchdog_timeout_s()

    @staticmethod
    def _cancel_future(future: Future[RemoteCallResult]) -> None:
        try:
            future.cancel()
        except Exception:  # noqa: BLE001
            return

    def _safe_future_result(self, *, call: _InFlightCall, now: float) -> RemoteCallResult:
        try:
            result = call.future.result()
        except Exception as exc:  # noqa: BLE001
            return self._synthetic_transport_error(
                now=now,
                started_mono_s=call.started_mono_s,
                error=f"future_exception:{type(exc).__name__}:{exc}",
            )
        if isinstance(result, RemoteCallResult):
            return result
        return self._synthetic_transport_error(
            now=now,
            started_mono_s=call.started_mono_s,
            error=f"future_invalid_result:{type(result).__name__}",
        )

    @staticmethod
    def _synthetic_transport_error(*, now: float, started_mono_s: float, error: str) -> RemoteCallResult:
        return RemoteCallResult(
            ok=False,
            status_code=0,
            latency_ms=max(0.0, now - started_mono_s) * 1000.0,
            response_json=None,
            error=error,
        )

    @staticmethod
    def _write_log(log_obj: Any, payload: Dict[str, Any]) -> None:
        if log_obj is None:
            return
        try:
            log_obj.write(payload)
        except Exception:  # noqa: BLE001
            return

    @staticmethod
    def _preview_text(text: str, *, max_chars: int = 280) -> str:
        token = " ".join(str(text).split()).strip()
        if len(token) <= max_chars:
            return token
        return token[: max_chars - 3] + "..."


def _encode_frame_data_url(*, frame: np.ndarray, max_width: int, jpeg_quality: int) -> str:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    img = Image.fromarray(arr, mode="RGB")

    max_width = max(16, int(max_width))
    if img.width > max_width:
        new_h = int(round((max_width / float(img.width)) * img.height))
        img = img.resize((max_width, max(1, new_h)))

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=max(30, min(95, int(jpeg_quality))), optimize=True)
    b64 = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _debug_get(st: Optional[PerceptionState], key: str) -> Any:
    if st is None or not isinstance(st.debug, Mapping):
        return None
    return st.debug.get(key)


def _infer_zone_hint_from_text(*parts: Any) -> str:
    token = " ".join(" ".join(str(part or "").split()) for part in parts if part).lower()
    if not token:
        return "unknown"

    padded = f" {token} "
    markers = {
        "left": ("to my left", "on my left", " left side ", " left "),
        "right": ("to my right", "on my right", " right side ", " right "),
        "center": ("in front of me", "ahead of me", " center ", " middle "),
    }

    best_zone = "unknown"
    best_idx: Optional[int] = None
    for zone, variants in markers.items():
        for marker in variants:
            idx = padded.find(marker)
            if idx < 0:
                continue
            if best_idx is None or idx < best_idx:
                best_idx = idx
                best_zone = zone
    return best_zone


def _response_meta(response_json: Mapping[str, Any]) -> tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    try:
        choices = response_json.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                raw_reason = first.get("finish_reason")
                if isinstance(raw_reason, str) and raw_reason.strip():
                    finish_reason = raw_reason.strip()
    except Exception:  # noqa: BLE001
        pass

    try:
        usage = response_json.get("usage")
        if isinstance(usage, Mapping):
            prompt_raw = usage.get("prompt_tokens")
            completion_raw = usage.get("completion_tokens")
            total_raw = usage.get("total_tokens")
            if isinstance(prompt_raw, int):
                prompt_tokens = prompt_raw
            if isinstance(completion_raw, int):
                completion_tokens = completion_raw
            if isinstance(total_raw, int):
                total_tokens = total_raw
    except Exception:  # noqa: BLE001
        pass

    return finish_reason, prompt_tokens, completion_tokens, total_tokens
