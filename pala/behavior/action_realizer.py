from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Optional

from ..types import (
    ActionPlan,
    BreathCommand,
    GazeToCommand,
    GlanceCommand,
    HoldCommand,
    NodCommand,
    OrientToZoneCommand,
    PerceptionState,
    PrimitiveKind,
)
from .intents import BehaviorIntent
from .models import SceneObservation
from .scene_memory import SceneMemorySnapshot


@dataclass
class ActionRealizerConfig:
    enable_acknowledge: bool = True
    enable_reacquire: bool = True
    gaze_yaw_max_rad: float = 0.42
    gaze_pitch_max_rad: float = 0.28
    gaze_update_min_s: float = 0.35
    gaze_min_delta_rad: float = 0.05
    gaze_smoothing_alpha: float = 0.45
    expressive_period_s: float = 10.0
    proactive_motion_s: float = 2.2
    reaffirm_glance_amp_rad: float = 0.14
    remote_passthrough_conf: float = 0.8
    remote_repeat_holdoff_s: float = 3.5


@dataclass(frozen=True)
class _Step:
    label: str
    duration_s: float


class ActionRealizer:
    """Turns high-level intents into executable primitive actions with temporal continuity."""

    def __init__(
        self,
        cfg: Optional[ActionRealizerConfig] = None,
        *,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._cfg = cfg or ActionRealizerConfig()
        self._clock = clock or time.monotonic
        self._last_gaze_emit_s = -1_000.0
        self._last_gaze_target: Optional[tuple[float, float]] = None
        self._last_expressive_s = -1_000.0
        self._last_motion_emit_s = -1_000.0
        self._reaffirm_toggle = 0
        self._last_remote_signature: Optional[str] = None
        self._last_remote_emit_s = -1_000.0
        self._active_script_mode: Optional[str] = None
        self._active_script: list[_Step] = []
        self._step_idx = 0
        self._step_started_s = 0.0

    def realize(
        self,
        *,
        intent: BehaviorIntent,
        proposed: ActionPlan,
        st: Optional[PerceptionState],
        obs: SceneObservation,
        memory: SceneMemorySnapshot,
    ) -> ActionPlan:
        now = self._clock()

        if intent.mode in {"engage", "reacquire"}:
            if intent.mode == "reacquire" and not self._cfg.enable_reacquire:
                # Fall through to continuous modes.
                pass
            else:
                step = self._script_step(mode=intent.mode, now_s=now)
                return self._realize_step(
                    step=step,
                    now_s=now,
                    intent=intent,
                    st=st,
                    obs=obs,
                    memory=memory,
                    proposed=proposed,
                )

        # Reset scripts when switching to continuous modes.
        self._active_script_mode = None
        self._active_script = []
        self._step_idx = 0

        remote = self._maybe_remote_passthrough(now_s=now, proposed=proposed, intent=intent, obs=obs)
        if remote is not None:
            return self._finalize(remote, now_s=now)

        if intent.mode == "assist":
            if not intent.act_now:
                reaffirm = self._maybe_reaffirm_motion(now_s=now, intent=intent, obs=obs, memory=memory)
                if reaffirm is not None:
                    return self._finalize(reaffirm, now_s=now)
                return self._finalize(ActionPlan(
                    primitive=PrimitiveKind.HOLD,
                    command=HoldCommand(),
                    confidence=max(0.5, intent.confidence),
                    explanation=f"{intent.reason};realizer_assist_hold",
                    style="focused",
                    cancel_current=False,
                ), now_s=now)
            expressive = self._maybe_expressive_nod(now_s=now, intent=intent)
            if expressive is not None:
                return self._finalize(expressive, now_s=now)
            gaze = self._maybe_gaze(st=st, style="focused", conf=max(intent.confidence, 0.62), reason="realizer_assist_gaze")
            if gaze is not None:
                return self._finalize(gaze, now_s=now)
            reaffirm = self._maybe_reaffirm_motion(now_s=now, intent=intent, obs=obs, memory=memory)
            if reaffirm is not None:
                return self._finalize(reaffirm, now_s=now)
            return self._finalize(ActionPlan(
                primitive=PrimitiveKind.ORIENT_TO_ZONE,
                command=OrientToZoneCommand(zone=intent.target_zone or "center", amp_rad=0.22, rate_rad_s=1.5),
                confidence=max(0.58, intent.confidence),
                explanation=f"{intent.reason};realizer_assist_orient",
                style="focused",
                cancel_current=True,
            ), now_s=now)

        if intent.mode == "track":
            if not intent.act_now:
                reaffirm = self._maybe_reaffirm_motion(now_s=now, intent=intent, obs=obs, memory=memory)
                if reaffirm is not None:
                    return self._finalize(reaffirm, now_s=now)
                return self._finalize(ActionPlan(
                    primitive=PrimitiveKind.HOLD,
                    command=HoldCommand(),
                    confidence=max(0.5, intent.confidence),
                    explanation=f"{intent.reason};realizer_track_hold",
                    style="curious",
                    cancel_current=False,
                ), now_s=now)
            gaze = self._maybe_gaze(st=st, style="curious", conf=max(intent.confidence, 0.6), reason="realizer_track_gaze")
            if gaze is not None:
                return self._finalize(gaze, now_s=now)
            reaffirm = self._maybe_reaffirm_motion(now_s=now, intent=intent, obs=obs, memory=memory)
            if reaffirm is not None:
                return self._finalize(reaffirm, now_s=now)
            return self._finalize(ActionPlan(
                primitive=PrimitiveKind.ORIENT_TO_ZONE,
                command=OrientToZoneCommand(zone=intent.target_zone or "center", amp_rad=0.21, rate_rad_s=1.4),
                confidence=max(0.56, intent.confidence),
                explanation=f"{intent.reason};realizer_track_orient",
                style="curious",
                cancel_current=True,
            ), now_s=now)

        if intent.mode == "idle":
            if proposed.primitive in {PrimitiveKind.HOLD, PrimitiveKind.BREATH}:
                return proposed
            return self._finalize(ActionPlan(
                primitive=PrimitiveKind.BREATH,
                command=BreathCommand(amp_rad=0.06, period_s=6.8, rate_rad_s=1.0),
                confidence=max(0.45, intent.confidence),
                explanation=f"{intent.reason};realizer_idle_breath",
                style="calm",
                cancel_current=False,
            ), now_s=now)

        # Fallback.
        return self._finalize(ActionPlan(
            primitive=PrimitiveKind.HOLD,
            command=HoldCommand(),
            confidence=max(0.4, intent.confidence),
            explanation=f"{intent.reason};realizer_fallback_hold",
            style="calm",
            cancel_current=False,
        ), now_s=now)

    def _script_step(self, *, mode: str, now_s: float) -> _Step:
        if mode != self._active_script_mode or not self._active_script:
            self._active_script_mode = mode
            self._active_script = _script_for_mode(mode)
            self._step_idx = 0
            self._step_started_s = now_s

        if self._step_idx < (len(self._active_script) - 1):
            step = self._active_script[self._step_idx]
            if (now_s - self._step_started_s) >= step.duration_s:
                self._step_idx += 1
                self._step_started_s = now_s
        return self._active_script[self._step_idx]

    def _realize_step(
        self,
        *,
        step: _Step,
        now_s: float,
        intent: BehaviorIntent,
        st: Optional[PerceptionState],
        obs: SceneObservation,
        memory: SceneMemorySnapshot,
        proposed: ActionPlan,
    ) -> ActionPlan:
        if step.label == "ack_nod":
            if not self._cfg.enable_acknowledge:
                if st is not None and st.primary_person is not None:
                    gaze = self._maybe_gaze(st=st, style="curious", conf=max(0.6, intent.confidence), reason="realizer_ack_skip_to_gaze")
                    if gaze is not None:
                        return self._finalize(gaze, now_s=now_s)
                return self._finalize(ActionPlan(
                    primitive=PrimitiveKind.ORIENT_TO_ZONE,
                    command=OrientToZoneCommand(zone=intent.target_zone or obs.zone or "center", amp_rad=0.21, rate_rad_s=1.4),
                    confidence=max(0.55, intent.confidence),
                    explanation=f"{intent.reason};realizer_ack_skip_orient",
                    style="curious",
                    cancel_current=True,
                ), now_s=now_s)
            self._last_expressive_s = now_s
            return self._finalize(ActionPlan(
                primitive=PrimitiveKind.NOD,
                command=NodCommand(amp_rad=0.18, duration_s=0.45, cycles=1, rate_rad_s=1.7),
                confidence=max(0.68, intent.confidence),
                explanation=f"{intent.reason};realizer_ack_nod",
                style="curious",
                cancel_current=True,
            ), now_s=now_s)
        if step.label == "reacquire_glance_a":
            return self._finalize(ActionPlan(
                primitive=PrimitiveKind.GLANCE,
                command=GlanceCommand(direction=memory.likely_zone or "left", amp_rad=0.22, duration_s=0.55, rate_rad_s=1.5),
                confidence=max(0.56, intent.confidence),
                explanation=f"{intent.reason};realizer_reacquire_a",
                style="curious",
                cancel_current=True,
            ), now_s=now_s)
        if step.label == "reacquire_glance_b":
            opposite = "right" if (memory.likely_zone or "left") == "left" else "left"
            return self._finalize(ActionPlan(
                primitive=PrimitiveKind.GLANCE,
                command=GlanceCommand(direction=opposite, amp_rad=0.22, duration_s=0.55, rate_rad_s=1.5),
                confidence=max(0.56, intent.confidence),
                explanation=f"{intent.reason};realizer_reacquire_b",
                style="curious",
                cancel_current=True,
            ), now_s=now_s)
        # settle step
        if st is not None and st.primary_person is not None:
            gaze = self._maybe_gaze(st=st, style="curious", conf=max(intent.confidence, 0.6), reason="realizer_script_settle_gaze")
            if gaze is not None:
                return self._finalize(gaze, now_s=now_s)
        return self._finalize(ActionPlan(
            primitive=PrimitiveKind.ORIENT_TO_ZONE,
            command=OrientToZoneCommand(zone=intent.target_zone or obs.zone or "center", amp_rad=0.21, rate_rad_s=1.4),
            confidence=max(0.55, intent.confidence),
            explanation=f"{intent.reason};realizer_script_settle_orient",
            style="curious",
            cancel_current=True,
        ), now_s=now_s)

    def _maybe_expressive_nod(self, *, now_s: float, intent: BehaviorIntent) -> Optional[ActionPlan]:
        if (now_s - self._last_expressive_s) < self._cfg.expressive_period_s:
            return None
        self._last_expressive_s = now_s
        return ActionPlan(
            primitive=PrimitiveKind.NOD,
            command=NodCommand(amp_rad=0.12, duration_s=0.4, cycles=1, rate_rad_s=1.6),
            confidence=max(0.55, intent.confidence),
            explanation=f"{intent.reason};realizer_expressive_nod",
            style="focused",
            cancel_current=True,
        )

    def _maybe_reaffirm_motion(
        self,
        *,
        now_s: float,
        intent: BehaviorIntent,
        obs: SceneObservation,
        memory: SceneMemorySnapshot,
    ) -> Optional[ActionPlan]:
        if not obs.person_present:
            return None
        if (now_s - self._last_motion_emit_s) < self._cfg.proactive_motion_s:
            return None

        direction = intent.target_zone or obs.zone or memory.likely_zone
        if direction in {None, "center"}:
            self._reaffirm_toggle += 1
            direction = "left" if (self._reaffirm_toggle % 2) == 0 else "right"
        else:
            direction = "left" if direction == "left" else "right"
        return ActionPlan(
            primitive=PrimitiveKind.GLANCE,
            command=GlanceCommand(
                direction=direction,
                amp_rad=self._cfg.reaffirm_glance_amp_rad,
                duration_s=0.35,
                rate_rad_s=1.45,
            ),
            confidence=max(0.54, intent.confidence),
            explanation=f"{intent.reason};realizer_reaffirm_motion",
            style="curious",
            cancel_current=True,
        )

    def _maybe_remote_passthrough(
        self,
        *,
        now_s: float,
        proposed: ActionPlan,
        intent: BehaviorIntent,
        obs: SceneObservation,
    ) -> Optional[ActionPlan]:
        if (not intent.act_now) and intent.urgency == "low" and intent.remote_confidence < 0.75:
            return None
        if proposed.confidence < self._cfg.remote_passthrough_conf:
            return None
        if proposed.primitive not in {
            PrimitiveKind.GLANCE,
            PrimitiveKind.NOD,
            PrimitiveKind.ORIENT_TO_ZONE,
            PrimitiveKind.GAZE_TO,
        }:
            return None
        if proposed.primitive == PrimitiveKind.ORIENT_TO_ZONE and isinstance(proposed.command, OrientToZoneCommand):
            if proposed.command.zone == "center" and intent.mode in {"track", "assist"} and obs.person_present:
                return None
        if proposed.primitive == PrimitiveKind.GAZE_TO and not obs.person_present:
            return None

        signature = f"{proposed.primitive.value}:{proposed.command!r}:{proposed.style}"
        repeated = signature == self._last_remote_signature
        if repeated and (now_s - self._last_remote_emit_s) < self._cfg.remote_repeat_holdoff_s:
            return None

        self._last_remote_signature = signature
        self._last_remote_emit_s = now_s
        return ActionPlan(
            primitive=proposed.primitive,
            command=proposed.command,
            confidence=max(0.55, proposed.confidence),
            explanation=f"{proposed.explanation or ''};realizer_remote_passthrough",
            style=proposed.style or ("curious" if intent.mode in {"track", "engage"} else "calm"),
            cancel_current=bool(proposed.cancel_current or proposed.primitive not in {PrimitiveKind.HOLD, PrimitiveKind.BREATH}),
        )

    def _maybe_gaze(self, *, st: Optional[PerceptionState], style: str, conf: float, reason: str) -> Optional[ActionPlan]:
        if st is None or st.primary_person is None:
            return None
        now = self._clock()
        yaw = _clamp(((float(st.primary_person.cx) - 0.5) * 2.0) * self._cfg.gaze_yaw_max_rad, self._cfg.gaze_yaw_max_rad)
        pitch = _clamp(((0.5 - float(st.primary_person.cy)) * 2.0) * self._cfg.gaze_pitch_max_rad, self._cfg.gaze_pitch_max_rad)
        target = (yaw, pitch)
        if self._last_gaze_target is not None:
            alpha = max(0.05, min(1.0, float(self._cfg.gaze_smoothing_alpha)))
            target = (
                (alpha * target[0]) + ((1.0 - alpha) * self._last_gaze_target[0]),
                (alpha * target[1]) + ((1.0 - alpha) * self._last_gaze_target[1]),
            )
            dy = target[0] - self._last_gaze_target[0]
            dp = target[1] - self._last_gaze_target[1]
            delta = math.hypot(dy, dp)
            if delta < self._cfg.gaze_min_delta_rad and (now - self._last_gaze_emit_s) < self._cfg.gaze_update_min_s:
                return None
        self._last_gaze_target = target
        self._last_gaze_emit_s = now
        return ActionPlan(
            primitive=PrimitiveKind.GAZE_TO,
            command=GazeToCommand(
                yaw_rad=target[0],
                pitch_rad=target[1],
                rate_rad_s=_style_rate(style),
                dwell_s=0.0,
                timeout_s=0.9,
            ),
            confidence=conf,
            explanation=f"{reason}:yaw={target[0]:+.3f},pitch={target[1]:+.3f}",
            style=style,
            cancel_current=True,
        )

    def _finalize(self, plan: ActionPlan, *, now_s: float) -> ActionPlan:
        if plan.primitive not in {PrimitiveKind.HOLD, PrimitiveKind.BREATH}:
            self._last_motion_emit_s = now_s
        return plan


def _script_for_mode(mode: str) -> list[_Step]:
    if mode == "engage":
        return [
            _Step(label="ack_nod", duration_s=0.5),
            _Step(label="settle", duration_s=1.0),
        ]
    if mode == "reacquire":
        return [
            _Step(label="reacquire_glance_a", duration_s=0.6),
            _Step(label="reacquire_glance_b", duration_s=0.6),
            _Step(label="settle", duration_s=0.9),
        ]
    return [_Step(label="settle", duration_s=0.9)]


def _style_rate(style: str) -> float:
    token = str(style or "").strip().lower()
    if token == "focused":
        return 1.9
    if token == "curious":
        return 1.6
    return 1.3


def _clamp(value: float, max_abs: float) -> float:
    lo = -abs(max_abs)
    hi = abs(max_abs)
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
