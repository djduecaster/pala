from __future__ import annotations

import base64
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import logging
import re
import time
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
from PIL import Image

from ..types import ActionPlan, HoldCommand, PerceptionState, PrimitiveKind
from ..utils import maybe_logger
from .action_translator import ActionTranslator
from .env_processor import CosmosEnvProcessor, EnvProcessorConfig, EnvProcessorParseResult
from .frame_window import RollingFrameWindow
from .planner_client import CosmosPlannerClient, PlannerClientConfig, PlannerDecision
from .prompts import build_env_user_text, build_messages, build_planner_user_text
from .remote_api import RemoteCallResult, extract_message_content, normalize_chat_url, post_chat_json
from .world_state_store import DecisionSnapshot, WorldStateStore

logger = logging.getLogger(__name__)


@dataclass
class BehaviorPolicyConfig:
    fallback_style: str = "calm"
    fallback_confidence: float = 0.1
    persist_every_step: bool = False
    remote_enabled: bool = False
    provider: str = "brev"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "nvidia/cosmos-reason2-2b"
    request_timeout_ms: int = 6000
    error_backoff_s: float = 1.5
    client_error_backoff_s: float = 5.0
    env_hz: float = 1.0
    planner_hz: float = 0.5
    planner_event_delta_threshold: float = 0.65
    planner_event_cooldown_s: float = 0.7
    max_frame_age_ms: int = 500
    frame_window_s: float = 6.0
    env_max_frames: int = 6
    planner_max_frames: int = 1
    frame_max_width: int = 320
    frame_jpeg_quality: int = 60
    request_min_fresh_frames: int = 1
    planner_include_latest_frame: bool = True
    env_max_tokens: int = 900
    planner_max_tokens: int = 900
    policy_identity: str = "You are PALA, a social desk companion lamp."
    policy_capabilities: str = "Use available primitives for expressive lamp motion."
    policy_safety: str = "Prioritize safe, smooth, non-aggressive motion."
    policy_style: str = "Use calm by default."
    planner_prompt: str = ""
    repeated_breath_guard_count: int = 4
    stale_breath_hold_after_s: float = 12.0
    env_log_path: Optional[str] = "logs/behavior_env.jsonl"
    planner_log_path: Optional[str] = "logs/behavior_planner.jsonl"
    reasoning_log_path: Optional[str] = "logs/behavior_reasoning.jsonl"


@dataclass
class _InFlightCall:
    request_id: int
    started_mono_s: float
    payload: Mapping[str, Any]
    future: Future[RemoteCallResult]


class BehaviorPolicy:
    """Remote-first behavior policy with async env/planner workers."""

    owns_semantic_behavior = True

    def __init__(
        self,
        *,
        planner: Any = None,
        world_state: Optional[WorldStateStore] = None,
        config: Optional[BehaviorPolicyConfig] = None,
        clock: Optional[Callable[[], float]] = None,
        frame_cache: Optional[Any] = None,
        # accepted for runtime call-site compatibility
        dwell_s: float = 2.0,
        cooldown_s: float = 1.0,
        max_hold_s: float = 2.0,
    ):
        _ = (dwell_s, cooldown_s, max_hold_s)
        self._planner_fallback = planner
        self._world_state = world_state or WorldStateStore()
        self._cfg = config or BehaviorPolicyConfig()
        self._clock = clock or time.monotonic
        self._frame_cache = frame_cache
        self._frame_window = RollingFrameWindow(window_s=self._cfg.frame_window_s)
        self._last_ingested_frame_ns: Optional[int] = None

        self._env_processor = CosmosEnvProcessor(
            EnvProcessorConfig(
                max_inflight=1,
                event_delta_threshold=self._cfg.planner_event_delta_threshold,
            )
        )
        self._planner_client = CosmosPlannerClient(PlannerClientConfig(max_inflight=1))
        self._translator = ActionTranslator()

        self._chat_url = normalize_chat_url(self._cfg.base_url or "")
        self._remote_enabled = bool(self._cfg.remote_enabled and self._chat_url)
        self._executor: Optional[ThreadPoolExecutor] = None
        if self._remote_enabled:
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
        self._repeated_breath_decision_count = 0
        self._last_action_commit_s = self._clock()

        self._env_log = maybe_logger(self._cfg.env_log_path) if self._cfg.env_log_path else None
        self._planner_log = maybe_logger(self._cfg.planner_log_path) if self._cfg.planner_log_path else None
        self._reasoning_log = maybe_logger(self._cfg.reasoning_log_path) if self._cfg.reasoning_log_path else None

        self._current_action: ActionPlan = self._new_hold_action("startup_hold")

    @property
    def world_state(self) -> WorldStateStore:
        return self._world_state

    def set_control_state(self, control_state: Any) -> None:
        self._world_state.set_control_state(control_state)

    def step(self, st: Optional[PerceptionState]) -> ActionPlan:
        now = self._clock()
        self._ingest_latest_frame()

        self._drain_env_inflight(now=now)
        self._drain_planner_inflight(now=now)

        if self._remote_enabled:
            self._maybe_schedule_env(st=st, now=now)
            self._maybe_schedule_planner(st=st, now=now)
        else:
            fallback = self._plan_with_fallback(st)
            if fallback is not None:
                self._set_current_action(fallback, rationale=fallback.explanation or "")

        if self._cfg.persist_every_step:
            self._world_state.persist()
        self._maybe_decay_stale_breath(now=now)
        return self._current_action

    def shutdown(self) -> None:
        for logger_obj in (self._env_log, self._planner_log, self._reasoning_log):
            if logger_obj is not None:
                try:
                    logger_obj.close()
                except Exception:  # noqa: BLE001 - shutdown best-effort
                    continue
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _ingest_latest_frame(self) -> None:
        if self._frame_cache is None:
            return
        try:
            snap = self._frame_cache.get(max_age_ms=self._cfg.max_frame_age_ms)
        except Exception:  # noqa: BLE001 - defensive
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

    def _maybe_schedule_env(self, *, st: Optional[PerceptionState], now: float) -> None:
        if self._executor is None:
            return
        if now < self._next_env_allowed_s:
            return
        if self._env_inflight is not None:
            # Keep only "latest pending" intent; payload is rebuilt when worker frees.
            self._env_processor.submit_or_replace({"queued": True, "ts_mono_s": now})
            return
        period_s = 1.0 / max(0.05, float(self._cfg.env_hz))
        if (now - self._last_env_submit_s) < period_s:
            return
        payload = self._build_env_payload(st=st)
        if payload is None:
            return
        accepted = self._env_processor.submit_or_replace(payload)
        if not accepted:
            return
        self._env_request_seq += 1
        req_id = self._env_request_seq
        self._last_env_submit_s = now
        future = self._executor.submit(
            post_chat_json,
            url=self._chat_url,
            payload=payload["body"],
            timeout_s=max(0.25, self._cfg.request_timeout_ms / 1000.0),
            api_key=self._cfg.api_key,
        )
        self._env_inflight = _InFlightCall(request_id=req_id, started_mono_s=now, payload=payload, future=future)
        self._write_log(
            self._env_log,
            {
                "ts_wall_s": time.time(),
                "request_id": req_id,
                "phase": "env_processor",
                "status": "req_start",
                "frames": payload["frames"],
            },
        )

    def _maybe_schedule_planner(self, *, st: Optional[PerceptionState], now: float) -> None:
        if self._executor is None:
            return
        if now < self._next_planner_allowed_s:
            return
        if self._planner_inflight is not None:
            # Keep only "latest pending" intent; payload is rebuilt when worker frees.
            self._planner_client.submit_or_replace({"queued": True, "ts_mono_s": now})
            return
        period_s = 1.0 / max(0.05, float(self._cfg.planner_hz))
        due_periodic = (now - self._last_planner_submit_s) >= period_s
        due_event = self._pending_planner_event and (
            (now - self._last_planner_event_submit_s) >= max(0.05, self._cfg.planner_event_cooldown_s)
        )
        if not due_periodic and not due_event:
            return
        payload = self._build_planner_payload(st=st)
        if payload is None:
            return
        accepted = self._planner_client.submit_or_replace(payload)
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
            timeout_s=max(0.25, self._cfg.request_timeout_ms / 1000.0),
            api_key=self._cfg.api_key,
        )
        self._planner_inflight = _InFlightCall(request_id=req_id, started_mono_s=now, payload=payload, future=future)
        self._write_log(
            self._planner_log,
            {
                "ts_wall_s": time.time(),
                "request_id": req_id,
                "phase": "planner",
                "status": "req_start",
                "frames": payload["frames"],
            },
        )

    def _drain_env_inflight(self, *, now: float) -> None:
        call = self._env_inflight
        if call is None or not call.future.done():
            return
        self._env_inflight = None
        result = call.future.result()
        content_text = None
        reasoning_text = None
        parse_result: Optional[EnvProcessorParseResult] = None
        status = "transport_error"
        error = result.error
        raw_delta: Optional[float] = None

        if result.ok and result.response_json is not None:
            content_text, reasoning_text = extract_message_content(result.response_json)
            if content_text is None:
                status = "empty_content"
                error = "missing_message_content"
                parse_result = self._env_processor.complete_request("")
            else:
                parse_result = self._env_processor.complete_request(content_text)
                if parse_result is None:
                    status = "parse_fail"
                    error = "env_tag_parse_failed"
                else:
                    status = "ok"
                    error = None
                    raw_delta = parse_result.snapshot.delta_score
                    parse_result.snapshot.delta_score = self._normalize_env_delta(parse_result.snapshot)
                    self._world_state.update_environment(parse_result.snapshot)
                    if parse_result.snapshot.delta_score >= self._cfg.planner_event_delta_threshold:
                        self._pending_planner_event = True
        else:
            self._env_processor.complete_request("")
        if status == "ok":
            self._next_env_allowed_s = 0.0
        else:
            self._next_env_allowed_s = max(
                self._next_env_allowed_s,
                now + self._compute_failure_backoff_s(result),
            )

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
                "delta_score_raw": raw_delta,
                "delta_score": None if parse_result is None else parse_result.snapshot.delta_score,
                "summary": None if parse_result is None else parse_result.snapshot.summary,
            },
        )
        reasoning_log_text = reasoning_text
        if reasoning_log_text is None and parse_result is not None:
            reasoning_log_text = parse_result.reasoning_text
        if reasoning_log_text:
            self._write_log(
                self._reasoning_log,
                {
                    "ts_wall_s": time.time(),
                    "request_id": call.request_id,
                    "component": "env_processor",
                    "latency_ms": round(result.latency_ms, 1),
                    "reasoning": reasoning_log_text,
                },
            )

        pending = self._env_processor.take_latest_pending()
        if pending is not None:
            self._last_env_submit_s = now - (1.0 / max(0.05, float(self._cfg.env_hz)))
            self._maybe_schedule_env(st=None, now=now)

    def _drain_planner_inflight(self, *, now: float) -> None:
        call = self._planner_inflight
        if call is None or not call.future.done():
            return
        self._planner_inflight = None
        result = call.future.result()
        content_text = None
        reasoning_text = None
        decision: Optional[PlannerDecision] = None
        status = "transport_error"
        error = result.error
        translation_error: Optional[str] = None

        if result.ok and result.response_json is not None:
            content_text, reasoning_text = extract_message_content(result.response_json)
            if content_text is None:
                status = "empty_content"
                error = "missing_message_content"
                decision = self._planner_client.complete_request("")
            else:
                decision = self._planner_client.complete_request(content_text)
                if decision is None:
                    status = "parse_fail"
                    error = "planner_tag_parse_failed"
                else:
                    decision = self._apply_repetition_guard(decision)
                    decision = self._repair_planner_decision(decision)
                    status = "ok"
                    error = None
                    translated = self._translator.translate(decision)
                    translation_error = translated.error
                    if translated.action is not None:
                        if self._should_commit_action(translated.action):
                            self._set_current_action(translated.action, rationale=decision.rationale_short)
                        else:
                            self._world_state.append_decision(
                                DecisionSnapshot(
                                    primitive=decision.primitive or "none",
                                    style=decision.style,
                                    confidence=decision.confidence,
                                    rationale_short=(decision.rationale_short or "").strip()[:220],
                                )
                            )
                    else:
                        self._world_state.append_decision(
                            DecisionSnapshot(
                                primitive=decision.primitive or "none",
                                style=decision.style,
                                confidence=decision.confidence,
                                rationale_short=(decision.rationale_short or "").strip()[:220],
                            )
                        )
        else:
            self._planner_client.complete_request("")
        if status == "ok":
            self._next_planner_allowed_s = 0.0
        else:
            self._next_planner_allowed_s = max(
                self._next_planner_allowed_s,
                now + self._compute_failure_backoff_s(result),
            )

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
                "decision_json": None
                if decision is None
                else {
                    "act_now": decision.act_now,
                    "primitive": decision.primitive,
                    "command": decision.command,
                    "style": decision.style,
                    "confidence": decision.confidence,
                },
                "rationale_short": None if decision is None else decision.rationale_short,
                "translation_error": translation_error,
            },
        )
        reasoning_log_text = reasoning_text
        if reasoning_log_text is None and decision is not None:
            reasoning_log_text = decision.reasoning_text
        if reasoning_log_text:
            self._write_log(
                self._reasoning_log,
                {
                    "ts_wall_s": time.time(),
                    "request_id": call.request_id,
                    "component": "planner",
                    "latency_ms": round(result.latency_ms, 1),
                    "reasoning": reasoning_log_text,
                },
            )

        pending = self._planner_client.take_latest_pending()
        if pending is not None:
            self._last_planner_submit_s = now - (1.0 / max(0.05, float(self._cfg.planner_hz)))
            self._maybe_schedule_planner(st=None, now=now)

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
        context = self._build_env_context(st=st, frame_timeline=frame_timeline)
        user_text = build_env_user_text(context=context, policy_identity=self._cfg.policy_identity)
        body = {
            "model": self._cfg.model,
            "messages": build_messages(user_text=user_text, image_data_urls=image_data_urls),
            "temperature": 0.0,
            "max_tokens": int(self._cfg.env_max_tokens),
            "stream": False,
        }
        return {"body": body, "frames": len(frames)}

    def _build_planner_payload(self, *, st: Optional[PerceptionState]) -> Optional[Dict[str, Any]]:
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
        context = self._build_planner_context(st=st)
        user_text = build_planner_user_text(
            context=context,
            policy_identity=self._cfg.policy_identity,
            policy_capabilities=self._cfg.policy_capabilities,
            policy_safety=self._cfg.policy_safety,
            policy_style=self._cfg.policy_style,
            planner_prompt=self._cfg.planner_prompt,
        )
        body = {
            "model": self._cfg.model,
            "messages": build_messages(user_text=user_text, image_data_urls=image_data_urls[: self._cfg.planner_max_frames]),
            "temperature": 0.0,
            "max_tokens": int(self._cfg.planner_max_tokens),
            "stream": False,
        }
        return {"body": body, "frames": len(image_data_urls)}

    def _build_env_context(
        self,
        *,
        st: Optional[PerceptionState],
        frame_timeline: Optional[list[dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        _ = st
        snap = self._world_state.snapshot()
        event_tail = snap.get("event_tail", [])[-2:]
        compact_events = [
            {
                "t": self._format_ts_seconds(item.get("timestamp_wall_s")),
                "summary": self._preview_text(str(item.get("summary", "")), max_chars=240),
            }
            for item in event_tail
        ]
        latest = snap.get("latest_env_snapshot") or {}
        return {
            "control_state": snap.get("control_state_latest"),
            "latest_env_summary": latest.get("summary"),
            "recent_env_events": compact_events,
            "frame_timeline": frame_timeline or [],
        }

    def _build_planner_context(self, *, st: Optional[PerceptionState]) -> Dict[str, Any]:
        _ = st
        snap = self._world_state.snapshot()
        event_tail = snap.get("event_tail", [])[-6:]
        compact_events = [
            {
                "t": self._format_ts_seconds(item.get("timestamp_wall_s")),
                "summary": self._preview_text(str(item.get("summary", "")), max_chars=300),
            }
            for item in event_tail
        ]
        decision_tail = snap.get("decision_tail", [])[-5:]
        compact_decisions = [
            {
                "t": self._format_ts_seconds(item.get("timestamp_wall_s")),
                "primitive": item.get("primitive"),
                "style": item.get("style"),
                "confidence": item.get("confidence"),
                "rationale_short": self._preview_text(str(item.get("rationale_short", "")), max_chars=220),
            }
            for item in decision_tail
        ]
        return {
            "identity_core": snap.get("identity_core"),
            "latest_env_snapshot": snap.get("latest_env_snapshot"),
            "event_tail": compact_events,
            "decision_tail": compact_decisions,
            "control_state": snap.get("control_state_latest"),
            "session_digest": snap.get("session_digest"),
            "current_action": {
                "primitive": self._current_action.primitive.value,
                "style": self._current_action.style,
                "confidence": self._current_action.confidence,
            },
        }

    def _set_current_action(self, action: ActionPlan, *, rationale: str) -> None:
        self._current_action = action
        self._last_action_commit_s = self._clock()
        self._world_state.append_decision(
            DecisionSnapshot(
                primitive=action.primitive.value,
                style=action.style,
                confidence=action.confidence,
                rationale_short=(rationale or action.explanation or "").strip()[:220],
            )
        )

    def _apply_repetition_guard(self, decision: PlannerDecision) -> PlannerDecision:
        if decision.primitive == "breath" and decision.act_now and self._current_action.primitive == PrimitiveKind.BREATH:
            self._repeated_breath_decision_count += 1
        elif decision.primitive == "breath" and decision.act_now:
            self._repeated_breath_decision_count = 1
        else:
            self._repeated_breath_decision_count = 0

        if self._repeated_breath_decision_count < max(1, int(self._cfg.repeated_breath_guard_count)):
            return decision

        return PlannerDecision(
            act_now=False,
            primitive=None,
            command={},
            style=decision.style,
            confidence=min(0.4, decision.confidence),
            rationale_short="local guard: suppress repeated breath decision",
            reasoning_text=decision.reasoning_text,
            raw_text=decision.raw_text,
        )

    def _repair_planner_decision(self, decision: PlannerDecision) -> PlannerDecision:
        if decision.primitive != "orient_to_zone":
            return decision
        zone = decision.command.get("zone")
        if isinstance(zone, str) and zone.strip().lower() in {"left", "center", "right"}:
            return decision

        resolved = self._infer_zone_from_context(decision)
        if resolved is None:
            # Fail-safe default when orient request omits explicit zone.
            resolved = "center"

        command = dict(decision.command)
        command["zone"] = resolved
        return PlannerDecision(
            act_now=decision.act_now,
            primitive=decision.primitive,
            command=command,
            style=decision.style,
            confidence=decision.confidence,
            rationale_short=decision.rationale_short,
            reasoning_text=decision.reasoning_text,
            raw_text=decision.raw_text,
        )

    def _infer_zone_from_context(self, decision: PlannerDecision) -> Optional[str]:
        # Prefer explicit textual hints from planner rationale/raw response.
        zone = self._infer_zone_from_text(
            decision.rationale_short,
            decision.raw_text,
        )
        if zone is not None:
            return zone

        snap = self._world_state.snapshot()
        latest = snap.get("latest_env_snapshot") or {}
        zone = self._infer_zone_from_text(
            latest.get("events"),
            latest.get("hypotheses"),
            latest.get("summary"),
            latest.get("scene"),
        )
        if zone is not None:
            return zone

        active_cmd = getattr(self._current_action, "command", None)
        if self._current_action.primitive == PrimitiveKind.ORIENT_TO_ZONE:
            active_zone = getattr(active_cmd, "zone", None)
            if isinstance(active_zone, str) and active_zone in {"left", "center", "right"}:
                return active_zone
        return None

    def _maybe_decay_stale_breath(self, *, now: float) -> None:
        if self._current_action.primitive != PrimitiveKind.BREATH:
            return
        if (now - self._last_action_commit_s) < max(1.0, float(self._cfg.stale_breath_hold_after_s)):
            return
        snap = self._world_state.snapshot()
        latest = snap.get("latest_env_snapshot") or {}
        try:
            delta = float(latest.get("delta_score", 0.0))
        except (TypeError, ValueError):
            delta = 0.0
        if delta > 0.35:
            return
        hold = self._new_hold_action("local guard: stale breath -> hold")
        self._set_current_action(hold, rationale=hold.explanation or "")
        self._repeated_breath_decision_count = 0

    def _normalize_env_delta(self, snapshot: Any) -> float:
        try:
            delta = float(snapshot.delta_score)
        except (TypeError, ValueError):
            delta = 0.2
        text = " ".join(
            [
                str(getattr(snapshot, "scene", "") or ""),
                str(getattr(snapshot, "events", "") or ""),
                str(getattr(snapshot, "summary", "") or ""),
            ]
        ).lower()
        static_markers = (
            "none applicable",
            "no observable changes",
            "does not show any specific actions",
            "static scene",
            "unchanged",
            "no movement",
            "no significant change",
        )
        if any(marker in text for marker in static_markers):
            return 0.2

        snap = self._world_state.snapshot()
        prev = snap.get("latest_env_snapshot") or {}
        prev_summary = self._preview_text(str(prev.get("summary", "") or ""), max_chars=300).lower()
        curr_summary = self._preview_text(str(getattr(snapshot, "summary", "") or ""), max_chars=300).lower()
        if prev_summary and curr_summary:
            if prev_summary == curr_summary:
                delta = min(delta, 0.15)
            else:
                overlap = self._token_overlap(prev_summary, curr_summary)
                if overlap >= 0.9:
                    delta = min(delta, 0.25)
                elif overlap >= 0.8:
                    delta = min(delta, 0.35)
                elif overlap >= 0.65:
                    delta = min(delta, 0.5)
        return max(0.0, min(1.0, delta))

    def _should_commit_action(self, action: ActionPlan) -> bool:
        same_action = (
            action.primitive == self._current_action.primitive
            and action.command == self._current_action.command
            and action.style == self._current_action.style
        )
        if not same_action:
            return True
        if action.primitive not in {PrimitiveKind.HOME, PrimitiveKind.HOLD}:
            return False
        snap = self._world_state.snapshot()
        latest = snap.get("latest_env_snapshot") or {}
        try:
            delta = float(latest.get("delta_score", 0.0))
        except (TypeError, ValueError):
            delta = 0.0
        return delta >= max(0.75, float(self._cfg.planner_event_delta_threshold))

    def _plan_with_fallback(self, st: Optional[PerceptionState]) -> Optional[ActionPlan]:
        if self._planner_fallback is None or not hasattr(self._planner_fallback, "plan"):
            return self._new_hold_action("fallback hold")
        try:
            action = self._planner_fallback.plan(st)
        except Exception as exc:  # noqa: BLE001 - fail closed in behavior loop
            logger.warning("behavior fallback planner failed error=%s", exc)
            return self._new_hold_action("fallback hold")
        if isinstance(action, ActionPlan):
            return action
        return self._new_hold_action("fallback hold")

    def _new_hold_action(self, reason: str) -> ActionPlan:
        return ActionPlan(
            primitive=PrimitiveKind.HOLD,
            command=HoldCommand(),
            confidence=self._cfg.fallback_confidence,
            style=self._cfg.fallback_style,
            cancel_current=False,
            explanation=reason,
        )

    @staticmethod
    def _write_log(log_obj: Any, payload: Dict[str, Any]) -> None:
        if log_obj is None:
            return
        try:
            log_obj.write(payload)
        except Exception:  # noqa: BLE001 - logging must never break runtime
            return

    def _compute_failure_backoff_s(self, result: RemoteCallResult) -> float:
        base = max(0.0, float(self._cfg.error_backoff_s))
        if 400 <= int(result.status_code) < 500:
            return max(base, float(self._cfg.client_error_backoff_s))
        return base

    @staticmethod
    def _preview_text(text: str, *, max_chars: int = 260) -> str:
        token = " ".join(str(text).split()).strip()
        if len(token) <= max_chars:
            return token
        return token[: max_chars - 3] + "..."

    @staticmethod
    def _format_ts_seconds(ts_wall_s: Any) -> Optional[str]:
        try:
            ts = float(ts_wall_s)
        except (TypeError, ValueError):
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        aset = {tok for tok in a.split() if len(tok) > 2}
        bset = {tok for tok in b.split() if len(tok) > 2}
        if not aset or not bset:
            return 0.0
        inter = len(aset & bset)
        union = len(aset | bset)
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _infer_zone_from_text(*segments: Any) -> Optional[str]:
        text = " ".join(str(seg or "") for seg in segments).lower()
        if not text:
            return None
        if re.search(r"\bleft\b|\bleft-side\b|\bleftward\b", text):
            return "left"
        if re.search(r"\bright\b|\bright-side\b|\brightward\b", text):
            return "right"
        if re.search(r"\bcenter\b|\bcentre\b|\bmiddle\b|\bforward\b|\bfront\b", text):
            return "center"
        return None


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
