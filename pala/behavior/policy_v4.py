from __future__ import annotations

import base64
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import io
import time
from typing import Any, Callable, Mapping, Optional

import numpy as np
from PIL import Image

from ..types import ActionPlan, PerceptionState
from ..utils import maybe_logger
from .action_guard import ActionGuard, ActionGuardConfig, GuardContext, GuardResult
from .decision_schema_v4 import BehaviorDecisionParser, BehaviorDecisionParseResult
from .mode_fsm_v4 import MacroMode, ModeFsmV4, ModeFsmV4Config, ModeSignalsV4, ModeTransitionV4
from .model_clients import (
    BaseModelClient,
    ModelRequest,
    ModelResponse,
    build_model_client,
    extract_message_content,
    normalize_chat_url,
)
from .prompts import behavior_v4_mode_guidance, build_behavior_v4_user_text, build_messages
from .skills_v4 import (
    allowed_primitives_for,
    allowed_skills_for_mode,
    default_action_payload_for_mode,
    default_skill_for_mode,
    skill_spec_v4,
)
from .trace_bus import TraceBus
from .decision_schema_v4 import behavior_decision_response_format


@dataclass
class BehaviorPolicyV4Config:
    mode_fsm: ModeFsmV4Config = field(default_factory=ModeFsmV4Config)
    action_guard: ActionGuardConfig = field(default_factory=ActionGuardConfig)
    startup_wake_enabled: bool = True
    startup_wake_left_s: float = 0.35
    startup_wake_right_s: float = 0.35
    startup_wake_loop_s: float = 0.45
    startup_wake_settle_s: float = 0.70
    startup_wake_rate_rad_s: float = 1.8
    startup_wake_yaw_rad: float = 0.16
    startup_min_s: float = 1.8
    startup_person_conf_fast_exit: float = 0.60
    remote_enabled: bool = False
    base_url: Optional[str] = None
    remote_provider: str = "auto"
    api_key: Optional[str] = None
    model: str = "nvidia/cosmos-reason2-2b"
    request_timeout_ms: int = 20000
    error_backoff_s: float = 1.5
    client_error_backoff_s: float = 5.0
    planner_hz: float = 0.5
    max_frame_age_ms: int = 500
    frame_max_width: int = 320
    frame_jpeg_quality: int = 55
    planner_include_latest_frame: bool = True
    planner_max_tokens: int = 1000
    planner_temperature: float = 0.0
    planner_top_p: float = 1.0
    policy_identity: str = "You are PALA, a social desk companion lamp."
    policy_capabilities: str = "Use safe expressive motion primitives."
    policy_safety: str = "Prioritize smooth, safe, non-aggressive motion."
    policy_style: str = "Default to calm style."
    planner_prompt: str = ""
    planner_log_path: Optional[str] = "logs/behavior_planner.jsonl"
    reasoning_log_path: Optional[str] = "logs/behavior_reasoning.jsonl"
    trace_log_path: Optional[str] = "logs/behavior_trace.jsonl"


@dataclass
class _InFlightCall:
    request_id: int
    started_mono_s: float
    future: Future[ModelResponse]


class BehaviorPolicyV4:
    """
    Behavior V4 scaffold:
    - deterministic macro-mode FSM
    - single decision parser entrypoint
    - strict ActionGuard for action validity and fallback
    """

    owns_semantic_behavior = True

    def __init__(
        self,
        *,
        config: Optional[BehaviorPolicyV4Config] = None,
        clock: Optional[Callable[[], float]] = None,
        frame_cache: Optional[Any] = None,
    ) -> None:
        self._cfg = config or BehaviorPolicyV4Config()
        self._clock = clock or time.monotonic
        self._frame_cache = frame_cache

        now = self._clock()
        self._mode_fsm = ModeFsmV4(config=self._cfg.mode_fsm)
        self._mode_fsm.reset(now_mono_s=now)
        self._last_mode_transition = ModeTransitionV4(
            previous_mode=MacroMode.BOOT_AWAKEN,
            next_mode=MacroMode.BOOT_AWAKEN,
            reason="startup",
            transitioned=False,
            dwell_s=0.0,
        )
        self._active_skill = default_skill_for_mode(MacroMode.BOOT_AWAKEN)
        self._active_skill_started_s = now
        self._active_mood = "calm"
        self._decision_parser = BehaviorDecisionParser()
        self._action_guard = ActionGuard(config=self._cfg.action_guard)

        startup_payload = default_action_payload_for_mode(MacroMode.BOOT_AWAKEN, reason="startup_hold")
        action = self._compile_action(startup_payload)
        self._current_action = action
        self._last_commit_s = now
        self._startup_started_s = now
        self._boot_step_started_s = now
        self._boot_step_index = 0
        self._boot_done = not bool(self._cfg.startup_wake_enabled)
        self._last_st: Optional[PerceptionState] = None
        self._last_guard_result: Optional[GuardResult] = None
        self._last_model_latency_ms: float = 0.0
        self._last_model_stage: str = "none"
        self._last_model_error: Optional[str] = None
        self._last_mode_signals = ModeSignalsV4(startup_complete=self._boot_done)

        chat_url = normalize_chat_url(self._cfg.base_url or "", provider=self._cfg.remote_provider)
        self._remote_enabled = bool(self._cfg.remote_enabled and chat_url)
        self._model_client: Optional[BaseModelClient] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        if self._remote_enabled:
            self._model_client = build_model_client(
                provider=str(self._cfg.remote_provider or "auto"),
                base_url=self._cfg.base_url or "",
                api_key=self._cfg.api_key,
            )
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="behavior_v4_remote")

        self._planner_inflight: Optional[_InFlightCall] = None
        self._planner_pending = False
        self._planner_request_seq = 0
        self._last_planner_submit_s = 0.0
        self._next_planner_allowed_s = 0.0

        self._planner_log = maybe_logger(self._cfg.planner_log_path) if self._cfg.planner_log_path else None
        self._reasoning_log = maybe_logger(self._cfg.reasoning_log_path) if self._cfg.reasoning_log_path else None
        self._trace = TraceBus(maybe_logger(self._cfg.trace_log_path) if self._cfg.trace_log_path else None)

    @property
    def current_mode(self) -> MacroMode:
        return self._mode_fsm.snapshot.mode

    @property
    def current_action(self) -> ActionPlan:
        return self._current_action

    @property
    def last_mode_transition(self) -> ModeTransitionV4:
        return self._last_mode_transition

    @property
    def last_parse_error(self) -> Optional[str]:
        return self._decision_parser.last_parse_error

    @property
    def last_parse_stage(self) -> str:
        return self._decision_parser.last_parse_stage

    @property
    def world_state(self) -> None:
        # V4 single-call path does not maintain env world-state snapshots.
        return None

    def set_control_state(self, _control_state: Any) -> None:
        # Reserved for future use in V4; currently behavior decisions are perception/model-driven.
        return None

    def shutdown(self) -> None:
        for logger_obj in (self._planner_log, self._reasoning_log):
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
        self._last_st = st
        self._drain_planner_inflight(now=now)
        person_conf = self._coerce_conf(st.primary_person_conf) if st is not None else 0.0
        self._update_boot_status(now=now, person_conf=person_conf)

        signals = self._signals_from_perception(st=st, now=now, person_conf=person_conf)
        self._last_mode_signals = signals
        transition = self._mode_fsm.update(now_mono_s=now, signals=signals)
        self._last_mode_transition = transition
        if transition.transitioned:
            self._set_active_skill(default_skill_for_mode(transition.next_mode), now_mono_s=now)
            fallback = self._action_guard.fallback_action(
                mode=transition.next_mode,
                reason=f"mode_enter:{transition.reason}",
            )
            self._commit(action=fallback, now_mono_s=now)

        if self.current_mode == MacroMode.BOOT_AWAKEN and not self._boot_done:
            self._run_boot_sequence(now=now)
            self._emit_trace(now=now)
            return self._current_action

        self._enforce_skill_timeout(now_mono_s=now)
        self._maybe_schedule_planner(st=st, now=now)
        self._emit_trace(now=now)
        return self._current_action

    def apply_model_output(self, raw_text: str, *, model_age_s: float = 0.0) -> GuardResult:
        now = self._clock()
        parsed = self._decision_parser.parse(raw_text)
        if parsed is None:
            fallback = self._action_guard.fallback_action(
                mode=self.current_mode,
                reason=f"parse_error:{self.last_parse_error or 'unknown'}",
            )
            self._commit(action=fallback, now_mono_s=now)
            result = GuardResult(
                accepted=False,
                used_fallback=True,
                reason=f"parse_error:{self.last_parse_error or 'unknown'}",
                action=fallback,
                skill=self._active_skill,
            )
            self._last_guard_result = result
            return result

        return self._apply_parsed_decision(parsed=parsed, model_age_s=model_age_s, now_mono_s=now)

    def _apply_parsed_decision(
        self,
        *,
        parsed: BehaviorDecisionParseResult,
        model_age_s: float,
        now_mono_s: float,
    ) -> GuardResult:
        decision = parsed.decision
        self._apply_requested_mode_transition(mode_transition=decision.mode_transition, now_mono_s=now_mono_s)
        action_age_s = max(0.0, now_mono_s - self._last_commit_s)
        context = GuardContext(
            mode=self.current_mode,
            active_skill=self._active_skill,
            current_action=self._current_action,
            action_age_s=action_age_s,
            model_age_s=max(0.0, float(model_age_s)),
        )
        result = self._action_guard.evaluate(decision=decision, context=context, now_mono_s=now_mono_s)
        self._set_active_skill(result.skill, now_mono_s=now_mono_s)
        self._commit(action=result.action, now_mono_s=now_mono_s)
        if result.accepted:
            self._active_mood = decision.mood
        if result.accepted:
            self._action_guard.mark_committed(action=result.action, now_mono_s=now_mono_s)
        self._last_guard_result = result
        return result

    def _signals_from_perception(
        self,
        *,
        st: Optional[PerceptionState],
        now: float,
        person_conf: float,
    ) -> ModeSignalsV4:
        del now  # reserved for future signal smoothing.
        del person_conf  # person confidence is sourced from debug packet when available.
        debug = st.debug if (st is not None and isinstance(st.debug, Mapping)) else {}

        startup_complete = self._boot_done
        debug_person_conf = self._coerce_conf(debug.get("person_conf"))
        person_present = self._coerce_bool(debug.get("person_present"), default=debug_person_conf > 0.15)

        return ModeSignalsV4(
            person_present=person_present,
            person_conf=debug_person_conf,
            startup_complete=startup_complete,
            search_requested=self._coerce_bool(debug.get("search_requested"), default=False),
            search_complete=self._coerce_bool(debug.get("search_complete"), default=False),
            assist_complete=self._coerce_bool(debug.get("assist_complete"), default=False),
            user_ack=self._coerce_bool(debug.get("user_ack"), default=False),
            task_active=self._coerce_bool(debug.get("task_active"), default=False),
            home_requested=self._coerce_bool(debug.get("home_requested"), default=False),
            home_completed=self._coerce_bool(debug.get("home_completed"), default=False),
            cancel_requested=self._coerce_bool(debug.get("cancel_requested"), default=False),
            health_degraded=self._coerce_bool(debug.get("health_degraded"), default=False),
        )

    def _maybe_schedule_planner(self, *, st: Optional[PerceptionState], now: float) -> None:
        if not self._remote_enabled or self._executor is None or self._model_client is None:
            return
        if self.current_mode == MacroMode.BOOT_AWAKEN and not self._boot_done:
            return
        if now < self._next_planner_allowed_s:
            return
        planner_hz = max(0.05, float(self._cfg.planner_hz))
        period_s = 1.0 / planner_hz
        due = self._planner_pending or ((now - self._last_planner_submit_s) >= period_s)
        if not due:
            return
        if self._planner_inflight is not None:
            self._planner_pending = True
            return

        request, frames = self._build_model_request(st=st, now=now)
        if request is None:
            return

        self._planner_request_seq += 1
        req_id = self._planner_request_seq
        self._last_planner_submit_s = now
        self._planner_pending = False
        future = self._executor.submit(self._model_client.chat, request)
        self._planner_inflight = _InFlightCall(req_id, now, future)
        self._write_log(
            self._planner_log,
            {
                "ts_wall_s": time.time(),
                "request_id": req_id,
                "phase": "planner_v4",
                "status": "req_start",
                "mode": self.current_mode.value,
                "skill": self._active_skill,
                "frames": frames,
                "response_format": "json_schema",
            },
        )

    def _build_model_request(self, *, st: Optional[PerceptionState], now: float) -> tuple[Optional[ModelRequest], int]:
        image_urls: list[str] = []
        if self._cfg.planner_include_latest_frame:
            url = self._latest_frame_data_url()
            if url is not None:
                image_urls.append(url)

        action_age_s = max(0.0, now - self._last_commit_s)
        allowed_skills = sorted(allowed_skills_for_mode(self.current_mode))
        allowed_primitives = sorted(allowed_primitives_for(self.current_mode, self._active_skill))
        context = {
            "mode": self.current_mode.value,
            "active_skill": self._active_skill,
            "active_mood": self._active_mood,
            "mode_reason": self._mode_fsm.snapshot.reason,
            "mode_contract": {
                "allowed_skills": allowed_skills,
                "allowed_primitives_for_active_skill": allowed_primitives,
            },
            "current_action": {
                "primitive": self._current_action.primitive.value,
                "style": self._current_action.style,
                "confidence": float(self._current_action.confidence),
            },
            "timing": {
                "action_age_s": round(action_age_s, 3),
            },
            "signals": {
                "person_present": bool(self._last_mode_signals.person_present),
                "person_conf": float(self._last_mode_signals.person_conf),
                "search_requested": bool(self._last_mode_signals.search_requested),
                "search_complete": bool(self._last_mode_signals.search_complete),
                "assist_complete": bool(self._last_mode_signals.assist_complete),
                "task_active": bool(self._last_mode_signals.task_active),
                "home_requested": bool(self._last_mode_signals.home_requested),
            },
        }
        user_text = build_behavior_v4_user_text(
            context=context,
            policy_identity=self._cfg.policy_identity,
            policy_capabilities=self._cfg.policy_capabilities,
            policy_safety=self._cfg.policy_safety,
            policy_style=self._cfg.policy_style,
            planner_prompt=self._cfg.planner_prompt,
            mode_guidance=behavior_v4_mode_guidance(self.current_mode.value, self._active_skill),
        )
        messages = build_messages(user_text=user_text, image_data_urls=image_urls)
        request = ModelRequest(
            model=self._cfg.model,
            messages=messages,
            response_format=behavior_decision_response_format(),
            timeout_s=max(0.1, float(self._cfg.request_timeout_ms) / 1000.0),
            max_tokens=max(64, int(self._cfg.planner_max_tokens)),
            temperature=float(self._cfg.planner_temperature),
            top_p=float(self._cfg.planner_top_p),
            stream=False,
            extra_body=None,
        )
        return request, len(image_urls)

    def _update_boot_status(self, *, now: float, person_conf: float) -> None:
        if self._boot_done:
            return
        elapsed = max(0.0, now - self._startup_started_s)
        if elapsed >= max(0.0, float(self._cfg.startup_min_s)):
            self._boot_done = True
            return
        if person_conf >= max(0.0, float(self._cfg.startup_person_conf_fast_exit)):
            self._boot_done = True

    def _run_boot_sequence(self, *, now: float) -> None:
        steps = self._boot_steps()
        if not steps:
            self._boot_done = True
            return

        while self._boot_step_index < len(steps):
            duration_s, payload = steps[self._boot_step_index]
            duration_s = max(0.0, float(duration_s))
            elapsed = max(0.0, now - self._boot_step_started_s)
            if elapsed < duration_s:
                self._commit_payload_if_changed(payload=payload, now_mono_s=now)
                return
            self._boot_step_index += 1
            self._boot_step_started_s = now

        self._boot_done = True

    def _boot_steps(self) -> list[tuple[float, dict]]:
        rate = max(0.2, float(self._cfg.startup_wake_rate_rad_s))
        amp = max(0.05, min(0.7, float(self._cfg.startup_wake_yaw_rad)))
        steps = [
            (
                float(self._cfg.startup_wake_left_s),
                {
                    "primitive": "orient_to_zone",
                    "command": {"zone": "left", "amp_rad": amp, "rate_rad_s": rate},
                    "style": "curious",
                    "confidence": 0.45,
                    "cancel_current": False,
                    "explanation": "startup_wake_left",
                },
            ),
            (
                float(self._cfg.startup_wake_right_s),
                {
                    "primitive": "orient_to_zone",
                    "command": {"zone": "right", "amp_rad": amp, "rate_rad_s": rate},
                    "style": "curious",
                    "confidence": 0.45,
                    "cancel_current": False,
                    "explanation": "startup_wake_right",
                },
            ),
            (
                float(self._cfg.startup_wake_loop_s),
                {
                    "primitive": "nod",
                    "command": {"amp_rad": 0.20, "duration_s": 0.45, "cycles": 1, "rate_rad_s": rate},
                    "style": "curious",
                    "confidence": 0.40,
                    "cancel_current": False,
                    "explanation": "startup_wake_loop",
                },
            ),
            (
                float(self._cfg.startup_wake_settle_s),
                {
                    "primitive": "breath",
                    "command": {"amp_rad": 0.07, "period_s": 6.0, "rate_rad_s": 0.9},
                    "style": "calm",
                    "confidence": 0.35,
                    "cancel_current": False,
                    "explanation": "startup_observe_settle",
                },
            ),
        ]
        return steps

    def _commit_payload_if_changed(self, *, payload: dict, now_mono_s: float) -> None:
        action = self._action_from_payload(payload)
        if action is None:
            return
        if self._action_signature(self._current_action) == self._action_signature(action):
            return
        self._commit(action=action, now_mono_s=now_mono_s)

    def _apply_requested_mode_transition(self, *, mode_transition: str, now_mono_s: float) -> None:
        target = self._mode_from_transition(mode_transition)
        if target is None or target == self.current_mode:
            return
        dwell = max(0.0, now_mono_s - self._mode_fsm.snapshot.entered_mono_s)
        min_dwell = max(0.0, float(self._cfg.mode_fsm.min_mode_dwell_s))
        if target != MacroMode.RECOVER_RESET and dwell < min_dwell:
            return
        transition = self._mode_fsm.force_mode(
            now_mono_s=now_mono_s,
            next_mode=target,
            reason=f"model_transition:{target.value}",
        )
        self._last_mode_transition = transition
        self._active_skill = default_skill_for_mode(target)
        self._active_skill_started_s = now_mono_s

    def _set_active_skill(self, skill: str, *, now_mono_s: float) -> None:
        token = str(skill or "").strip().lower()
        if token == self._active_skill:
            return
        self._active_skill = token
        self._active_skill_started_s = now_mono_s

    def _enforce_skill_timeout(self, *, now_mono_s: float) -> None:
        spec = skill_spec_v4(self._active_skill)
        if spec is None:
            return
        elapsed = max(0.0, now_mono_s - self._active_skill_started_s)
        if elapsed <= max(0.1, float(spec.max_dwell_s)):
            return

        timeout_skill = str(spec.timeout_fallback or "").strip().lower()
        timeout_spec = skill_spec_v4(timeout_skill) if timeout_skill else None
        if timeout_spec is not None and timeout_spec.mode != self.current_mode:
            transition = self._mode_fsm.force_mode(
                now_mono_s=now_mono_s,
                next_mode=timeout_spec.mode,
                reason=f"skill_timeout:{spec.name}",
            )
            self._last_mode_transition = transition

        if timeout_spec is not None:
            self._set_active_skill(timeout_spec.name, now_mono_s=now_mono_s)
        else:
            self._set_active_skill(default_skill_for_mode(self.current_mode), now_mono_s=now_mono_s)

        fallback = self._action_guard.fallback_action(
            mode=self.current_mode,
            reason=f"skill_timeout:{spec.name}",
        )
        self._commit(action=fallback, now_mono_s=now_mono_s)

    @staticmethod
    def _mode_from_transition(value: str) -> Optional[MacroMode]:
        token = str(value or "").strip().lower()
        if token == "stay":
            return None
        if not token.startswith("to_"):
            return None
        mode_token = token[3:]
        if mode_token == MacroMode.BOOT_AWAKEN.value:
            return None
        try:
            return MacroMode(mode_token)
        except ValueError:
            return None

    @staticmethod
    def _action_signature(action: ActionPlan) -> tuple[str, str]:
        command = getattr(action, "command", None)
        command_repr = repr(command)
        return action.primitive.value, command_repr

    def _latest_frame_data_url(self) -> Optional[str]:
        if self._frame_cache is None:
            return None
        try:
            packet = self._frame_cache.get(max_age_ms=int(self._cfg.max_frame_age_ms))
        except Exception:  # noqa: BLE001
            return None
        if packet is None:
            return None
        try:
            return _encode_frame_data_url(
                frame=packet.frame,
                max_width=max(16, int(self._cfg.frame_max_width)),
                jpeg_quality=max(30, min(95, int(self._cfg.frame_jpeg_quality))),
            )
        except Exception:  # noqa: BLE001
            return None

    def _drain_planner_inflight(self, *, now: float) -> None:
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
        content_text: Optional[str] = None
        finish_reason = None
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        parse_stage = "none"
        guard_reason: Optional[str] = None
        committed = False

        if result.ok and result.response_json is not None:
            finish_reason, prompt_tokens, completion_tokens, total_tokens = _response_meta(result.response_json)
            content_text, reasoning_text = extract_message_content(result.response_json)
            if content_text is None:
                status = "empty_content"
                error = "missing_message_content"
                self._last_model_error = error
            else:
                parsed = self._decision_parser.parse(content_text)
                parse_stage = self._decision_parser.last_parse_stage
                if parsed is None:
                    status = "parse_fail"
                    error = f"decision_parse_failed:{self._decision_parser.last_parse_error or 'unknown'}"
                    self._last_model_error = error
                    self._last_model_stage = parse_stage
                else:
                    model_age_s = max(0.0, (result.latency_ms / 1000.0))
                    guard = self._apply_parsed_decision(parsed=parsed, model_age_s=model_age_s, now_mono_s=now)
                    status = "ok"
                    error = None
                    parse_stage = parsed.parse_stage
                    guard_reason = guard.reason
                    committed = True
                    self._last_model_error = None
                    self._last_model_stage = parse_stage

            if reasoning_text:
                self._write_log(
                    self._reasoning_log,
                    {
                        "ts_wall_s": time.time(),
                        "request_id": call.request_id,
                        "component": "planner_v4",
                        "latency_ms": round(result.latency_ms, 1),
                        "reasoning": reasoning_text,
                    },
                )
        else:
            self._last_model_error = error
            self._last_model_stage = "none"

        self._last_model_latency_ms = float(result.latency_ms)
        if status == "ok":
            self._next_planner_allowed_s = 0.0
        else:
            self._next_planner_allowed_s = max(self._next_planner_allowed_s, now + self._compute_failure_backoff_s(result))
            # Keep trying latest frames after errors.
            self._planner_pending = True

        self._write_log(
            self._planner_log,
            {
                "ts_wall_s": time.time(),
                "request_id": call.request_id,
                "phase": "planner_v4",
                "status": status,
                "latency_ms": round(result.latency_ms, 1),
                "error": error,
                "response_preview": None if content_text is None else self._preview_text(content_text),
                "parse_stage": parse_stage,
                "mode": self.current_mode.value,
                "skill": self._active_skill,
                "guard_reason": guard_reason,
                "committed": committed,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )

    def _watchdog_expired(self, *, call: _InFlightCall, now: float) -> bool:
        timeout_s = max(0.1, float(self._cfg.request_timeout_ms) / 1000.0)
        watchdog_s = timeout_s + 2.0
        return (now - call.started_mono_s) > watchdog_s

    @staticmethod
    def _cancel_future(future: Future[Any]) -> None:
        try:
            future.cancel()
        except Exception:  # noqa: BLE001
            return

    def _safe_future_result(self, *, call: _InFlightCall, now: float) -> ModelResponse:
        try:
            result = call.future.result()
        except Exception as exc:  # noqa: BLE001
            return self._synthetic_transport_error(
                now=now,
                started_mono_s=call.started_mono_s,
                error=f"future_exception:{type(exc).__name__}:{exc}",
            )
        if isinstance(result, ModelResponse):
            return result
        return self._synthetic_transport_error(
            now=now,
            started_mono_s=call.started_mono_s,
            error=f"future_invalid_result:{type(result).__name__}",
        )

    @staticmethod
    def _synthetic_transport_error(*, now: float, started_mono_s: float, error: str) -> ModelResponse:
        return ModelResponse(
            ok=False,
            status_code=0,
            latency_ms=max(0.0, now - started_mono_s) * 1000.0,
            response_json=None,
            error=error,
        )

    def _compute_failure_backoff_s(self, result: ModelResponse) -> float:
        status = int(result.status_code)
        if 400 <= status < 500:
            return max(0.1, float(self._cfg.client_error_backoff_s))
        return max(0.1, float(self._cfg.error_backoff_s))

    @staticmethod
    def _write_log(log_obj: Any, payload: Mapping[str, Any]) -> None:
        if log_obj is None:
            return
        try:
            log_obj.write(dict(payload))
        except Exception:  # noqa: BLE001
            return

    def _emit_trace(self, *, now: float) -> None:
        guard = self._last_guard_result
        self._trace.emit(
            {
                "ts_wall_s": time.time(),
                "ts_mono_s": now,
                "mode": self.current_mode.value,
                "mode_transition": {
                    "transitioned": self._last_mode_transition.transitioned,
                    "from": self._last_mode_transition.previous_mode.value,
                    "to": self._last_mode_transition.next_mode.value,
                    "reason": self._last_mode_transition.reason,
                },
                "active_skill": self._active_skill,
                "active_mood": self._active_mood,
                "current_action": {
                    "primitive": self._current_action.primitive.value,
                    "style": self._current_action.style,
                    "confidence": float(self._current_action.confidence),
                },
                "signals": {
                    "person_present": bool(self._last_mode_signals.person_present),
                    "person_conf": float(self._last_mode_signals.person_conf),
                    "search_requested": bool(self._last_mode_signals.search_requested),
                    "search_complete": bool(self._last_mode_signals.search_complete),
                    "assist_complete": bool(self._last_mode_signals.assist_complete),
                    "task_active": bool(self._last_mode_signals.task_active),
                    "home_requested": bool(self._last_mode_signals.home_requested),
                },
                "planner": {
                    "enabled": self._remote_enabled,
                    "inflight": self._planner_inflight is not None,
                    "pending": self._planner_pending,
                    "last_latency_ms": round(self._last_model_latency_ms, 1),
                    "last_parse_stage": self._last_model_stage,
                    "last_error": self._last_model_error,
                    "next_allowed_in_s": round(max(0.0, self._next_planner_allowed_s - now), 3),
                },
                "boot": {
                    "enabled": bool(self._cfg.startup_wake_enabled),
                    "done": self._boot_done,
                    "step_index": int(self._boot_step_index),
                },
                "guard": None
                if guard is None
                else {
                    "accepted": guard.accepted,
                    "fallback": guard.used_fallback,
                    "reason": guard.reason,
                    "skill": guard.skill,
                    "primitive": guard.action.primitive.value,
                },
            }
        )

    def _commit(self, *, action: ActionPlan, now_mono_s: float) -> None:
        self._current_action = action
        self._last_commit_s = now_mono_s

    def _compile_action(self, payload: dict) -> ActionPlan:
        action = self._action_from_payload(payload)
        if action is not None:
            return action
        hold = self._action_from_payload(
            {
                "primitive": "hold",
                "command": {},
                "style": "calm",
                "confidence": 0.1,
                "cancel_current": False,
                "explanation": "fallback_startup_hold",
            }
        )
        assert hold is not None
        return hold

    @staticmethod
    def _action_from_payload(payload: dict) -> Optional[ActionPlan]:
        from ..types import action_plan_from_dict

        action = action_plan_from_dict(payload)
        if action is not None:
            action.cancel_current = False
        return action

    @staticmethod
    def _coerce_conf(value: Optional[float]) -> float:
        try:
            conf = float(value if value is not None else 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0.0:
            return 0.0
        if conf > 1.0:
            return 1.0
        return conf

    @staticmethod
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
