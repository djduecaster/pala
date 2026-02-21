from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional

from ..types import (
    ActionPlan,
    PerceptionState,
    PrimitiveKind,
    HoldCommand,
    BreathCommand,
    GlanceCommand,
    NodCommand,
    GazeToCommand,
    OrientToZoneCommand,
)
from .models import SceneObservation
from .scene_memory import SceneMemorySnapshot

_TRACK_PRIMITIVES = {
    PrimitiveKind.HOLD,
    PrimitiveKind.BREATH,
    PrimitiveKind.ORIENT_TO_ZONE,
    PrimitiveKind.GAZE_TO,
}


@dataclass
class BehaviorDirectorConfig:
    enabled: bool = True
    enable_acknowledge: bool = True
    enable_gaze_tracking: bool = True
    enable_reacquire: bool = True
    recently_seen_reacquire_s: float = 2.5
    mode_idle_s: float = 3.0
    mode_engage_s: float = 1.4
    mode_track_s: float = 2.5
    mode_assist_s: float = 3.0
    mode_reacquire_s: float = 2.0
    gaze_yaw_max_rad: float = 0.42
    gaze_pitch_max_rad: float = 0.28
    gaze_update_min_s: float = 0.35
    gaze_min_delta_rad: float = 0.05
    gaze_smoothing_alpha: float = 0.45
    ack_cooldown_s: float = 6.0
    remote_high_conf_override: float = 0.86
    blend_remote_weight: float = 0.35
    blend_switch_margin: float = 0.08
    blend_hold_s: float = 0.9


class BehaviorDirector:
    """
    High-level local behavior planner using scene memory + remote semantic hints.

    This layer decides *how* to express semantic intent through finite, executable primitives.
    """

    def __init__(
        self,
        cfg: Optional[BehaviorDirectorConfig] = None,
        *,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._cfg = cfg or BehaviorDirectorConfig()
        self._clock = clock or time.monotonic
        self._mode = "idle"
        self._mode_expires_s = 0.0
        self._last_ack_s = -1_000.0
        self._reacquire_toggle = 0
        self._engage_nod_done = False
        self._last_gaze_emit_s = -1_000.0
        self._last_gaze_target: Optional[tuple[float, float]] = None
        self._last_blend_label: str = "mode"
        self._last_blend_ts_s: float = -1_000.0

    def realize(
        self,
        proposed: ActionPlan,
        st: Optional[PerceptionState],
        obs: SceneObservation,
        memory: SceneMemorySnapshot,
    ) -> ActionPlan:
        if not self._cfg.enabled:
            return proposed
        now = self._clock()
        mode = self._select_mode(proposed=proposed, obs=obs, memory=memory, now_s=now)
        style = proposed.style or _style_for_mode(mode)
        conf = max(0.45, proposed.confidence)
        mode_action = self._mode_action(mode=mode, proposed=proposed, st=st, obs=obs, memory=memory, style=style, conf=conf, now=now)
        return self._blend_actions(
            mode_action=mode_action,
            proposed=proposed,
            obs=obs,
            memory=memory,
            mode=mode,
            now=now,
        )

    def _mode_action(
        self,
        *,
        mode: str,
        proposed: ActionPlan,
        st: Optional[PerceptionState],
        obs: SceneObservation,
        memory: SceneMemorySnapshot,
        style: str,
        conf: float,
        now: float,
    ) -> ActionPlan:
        if mode == "engage":
            if (
                self._cfg.enable_acknowledge
                and (not self._engage_nod_done)
                and (now - self._last_ack_s) >= self._cfg.ack_cooldown_s
            ):
                self._engage_nod_done = True
                self._last_ack_s = now
                return ActionPlan(
                    primitive=PrimitiveKind.NOD,
                    command=NodCommand(amp_rad=0.18, duration_s=0.45, cycles=1, rate_rad_s=1.7),
                    confidence=max(conf, 0.7),
                    explanation="director_engage_ack",
                    style="curious",
                    cancel_current=True,
                )
            return self._track_action(
                proposed=proposed,
                st=st,
                obs=obs,
                memory=memory,
                style=style,
                conf=conf,
                reason="director_engage_track",
            )

        if mode == "track":
            return self._track_action(
                proposed=proposed,
                st=st,
                obs=obs,
                memory=memory,
                style=style,
                conf=conf,
                reason="director_track",
            )

        if mode == "assist":
            if st is not None and st.primary_person is not None:
                gaze = self._maybe_gaze_action(st=st, style="focused", conf=max(conf, 0.62), reason="director_assist_gaze")
                if gaze is not None:
                    return gaze
                return ActionPlan(
                    primitive=PrimitiveKind.HOLD,
                    command=HoldCommand(),
                    confidence=max(conf, 0.5),
                    explanation="director_assist_hold_pose",
                    style="focused",
                    cancel_current=False,
                )
            return ActionPlan(
                primitive=PrimitiveKind.BREATH,
                command=BreathCommand(amp_rad=0.07, period_s=6.2, rate_rad_s=1.0),
                confidence=max(conf, 0.55),
                explanation="director_assist_presence",
                style="focused",
                cancel_current=False,
            )

        if mode == "reacquire":
            if not self._cfg.enable_reacquire:
                return ActionPlan(
                    primitive=PrimitiveKind.HOLD,
                    command=HoldCommand(),
                    confidence=max(conf, 0.45),
                    explanation="director_reacquire_disabled",
                    style="calm",
                    cancel_current=False,
                )
            direction = self._reacquire_direction(memory)
            return ActionPlan(
                primitive=PrimitiveKind.GLANCE,
                command=GlanceCommand(direction=direction, amp_rad=0.22, duration_s=0.55, rate_rad_s=1.5),
                confidence=max(conf, 0.56),
                explanation=f"director_reacquire:{direction}",
                style="curious",
                cancel_current=True,
            )

        if proposed.primitive in {PrimitiveKind.HOLD, PrimitiveKind.BREATH}:
            return proposed
        return ActionPlan(
            primitive=PrimitiveKind.BREATH,
            command=BreathCommand(amp_rad=0.06, period_s=6.8, rate_rad_s=1.0),
            confidence=max(0.45, proposed.confidence),
            explanation="director_idle_breath",
            style="calm",
            cancel_current=False,
        )

    def _blend_actions(
        self,
        *,
        mode_action: ActionPlan,
        proposed: ActionPlan,
        obs: SceneObservation,
        memory: SceneMemorySnapshot,
        mode: str,
        now: float,
    ) -> ActionPlan:
        mode_score = self._score_action(mode_action, obs=obs, memory=memory, mode=mode, source="mode")
        remote_score = self._score_action(proposed, obs=obs, memory=memory, mode=mode, source="remote")
        remote_conf = max(0.0, min(1.0, float(proposed.confidence)))

        # Hard override for highly confident, specific remote plans.
        if remote_conf >= self._cfg.remote_high_conf_override and proposed.primitive not in {PrimitiveKind.HOLD, PrimitiveKind.BREATH}:
            self._last_blend_label = "remote"
            self._last_blend_ts_s = now
            return proposed

        chosen = mode_action
        chosen_label = "mode"
        if remote_score > mode_score:
            chosen = proposed
            chosen_label = "remote"

        # Hysteresis to avoid ping-pong between local and remote when scores are near-equal.
        gap = abs(mode_score - remote_score)
        recent = (now - self._last_blend_ts_s) < self._cfg.blend_hold_s
        if recent and chosen_label != self._last_blend_label and gap < self._cfg.blend_switch_margin:
            chosen = mode_action if self._last_blend_label == "mode" else proposed
            chosen_label = self._last_blend_label

        self._last_blend_label = chosen_label
        self._last_blend_ts_s = now
        return chosen

    def _score_action(
        self,
        action: ActionPlan,
        *,
        obs: SceneObservation,
        memory: SceneMemorySnapshot,
        mode: str,
        source: str,
    ) -> float:
        score = 0.0
        primitive = action.primitive
        style = str(action.style or "").strip().lower()
        conf = max(0.0, min(1.0, float(action.confidence)))

        if obs.person_present:
            if primitive == PrimitiveKind.GAZE_TO:
                score += 0.8
            elif primitive == PrimitiveKind.ORIENT_TO_ZONE:
                score += 0.55
            elif primitive == PrimitiveKind.NOD:
                score += 0.45 if obs.event == "person_entered" else 0.1
            elif primitive == PrimitiveKind.GLANCE:
                score += 0.35 if obs.zone_changed else 0.05
            elif primitive == PrimitiveKind.BREATH:
                score += 0.18
            elif primitive == PrimitiveKind.HOLD:
                score += 0.06
        else:
            if primitive == PrimitiveKind.GLANCE:
                score += 0.55 if memory.person_recently_seen else 0.08
            elif primitive == PrimitiveKind.HOLD:
                score += 0.34
            elif primitive == PrimitiveKind.BREATH:
                score += 0.28
            elif primitive == PrimitiveKind.ORIENT_TO_ZONE:
                score += 0.15 if memory.person_recently_seen else 0.02
            elif primitive == PrimitiveKind.GAZE_TO:
                score -= 0.25

        if mode == "assist":
            if style == "focused":
                score += 0.18
            if primitive in {PrimitiveKind.GAZE_TO, PrimitiveKind.ORIENT_TO_ZONE}:
                score += 0.2
        elif mode in {"engage", "track", "reacquire"}:
            if style == "curious":
                score += 0.1

        score += 0.24 * (conf - 0.5)
        if source == "remote":
            score += self._cfg.blend_remote_weight * conf
        return score

    def _select_mode(
        self,
        *,
        proposed: ActionPlan,
        obs: SceneObservation,
        memory: SceneMemorySnapshot,
        now_s: float,
    ) -> str:
        # Keep short commitment for continuity unless a hard scene event occurred.
        if now_s < self._mode_expires_s and obs.event not in {"person_entered", "person_exited", "zone_changed"}:
            if obs.person_present and self._mode in {"engage", "track", "assist"}:
                return self._mode
            if (not obs.person_present) and self._mode in {"reacquire", "idle"}:
                return self._mode

        explicit = _intent_from_action(proposed)
        if obs.person_present:
            if explicit == "assist":
                self._set_mode("assist", now_s + self._cfg.mode_assist_s)
                return self._mode
            if explicit == "engage":
                self._set_mode("engage", now_s + self._cfg.mode_engage_s)
                return self._mode
            if explicit == "track":
                self._set_mode("track", now_s + self._cfg.mode_track_s)
                return self._mode
            if obs.event == "person_entered":
                self._set_mode("engage", now_s + self._cfg.mode_engage_s)
                return self._mode
            if obs.zone_changed or memory.zone_transitions_recent >= 2:
                self._set_mode("track", now_s + self._cfg.mode_track_s)
                return self._mode
            if _assist_hint(proposed=proposed, memory=memory):
                self._set_mode("assist", now_s + self._cfg.mode_assist_s)
                return self._mode
            if proposed.primitive in _TRACK_PRIMITIVES:
                self._set_mode("track", now_s + self._cfg.mode_track_s)
                return self._mode
            if self._mode in {"engage", "track", "assist"} and now_s < self._mode_expires_s:
                return self._mode
            self._set_mode("track", now_s + self._cfg.mode_track_s)
            return self._mode

        # Person currently absent.
        if explicit == "reacquire" and self._cfg.enable_reacquire:
            self._set_mode("reacquire", now_s + self._cfg.mode_reacquire_s)
            return self._mode
        if (
            self._cfg.enable_reacquire
            and
            memory.person_recently_seen
            and memory.last_seen_age_s is not None
            and memory.last_seen_age_s <= self._cfg.recently_seen_reacquire_s
        ):
            self._set_mode("reacquire", now_s + self._cfg.mode_reacquire_s)
            return self._mode
        self._set_mode("idle", now_s + self._cfg.mode_idle_s)
        return self._mode

    def _set_mode(self, mode: str, expires_s: float) -> None:
        if mode != self._mode:
            self._engage_nod_done = False
        self._mode = mode
        self._mode_expires_s = expires_s

    def _track_action(
        self,
        *,
        proposed: ActionPlan,
        st: Optional[PerceptionState],
        obs: SceneObservation,
        memory: SceneMemorySnapshot,
        style: str,
        conf: float,
        reason: str,
    ) -> ActionPlan:
        if self._cfg.enable_gaze_tracking and st is not None and st.primary_person is not None:
            gaze = self._maybe_gaze_action(st=st, style=style, conf=max(conf, 0.62), reason=reason)
            if gaze is not None:
                return gaze
            # Keep current gaze pose stable when target change is below deadband.
            return ActionPlan(
                primitive=PrimitiveKind.HOLD,
                command=HoldCommand(),
                confidence=max(conf, 0.5),
                explanation="director_track_hold_pose",
                style=style,
                cancel_current=False,
            )
        zone = memory.likely_zone or obs.zone or "center"
        return ActionPlan(
            primitive=PrimitiveKind.ORIENT_TO_ZONE,
            command=OrientToZoneCommand(zone=zone, amp_rad=0.22, rate_rad_s=1.5),
            confidence=max(conf, 0.6),
            explanation=f"{reason}:zone={zone}",
            style=style,
            cancel_current=True,
        )

    def _maybe_gaze_action(self, *, st: PerceptionState, style: str, conf: float, reason: str) -> Optional[ActionPlan]:
        assert st.primary_person is not None
        yaw = _clamp(((float(st.primary_person.cx) - 0.5) * 2.0) * self._cfg.gaze_yaw_max_rad, self._cfg.gaze_yaw_max_rad)
        pitch = _clamp(((0.5 - float(st.primary_person.cy)) * 2.0) * self._cfg.gaze_pitch_max_rad, self._cfg.gaze_pitch_max_rad)
        target = (yaw, pitch)
        now = self._clock()

        if self._last_gaze_target is not None:
            alpha = max(0.05, min(1.0, float(self._cfg.gaze_smoothing_alpha)))
            target = (
                (alpha * yaw) + ((1.0 - alpha) * self._last_gaze_target[0]),
                (alpha * pitch) + ((1.0 - alpha) * self._last_gaze_target[1]),
            )
            dy = target[0] - self._last_gaze_target[0]
            dp = target[1] - self._last_gaze_target[1]
            delta = (dy * dy + dp * dp) ** 0.5
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

    def _reacquire_direction(self, memory: SceneMemorySnapshot) -> str:
        self._reacquire_toggle += 1
        likely = memory.likely_zone
        if likely == "left":
            return "left" if (self._reacquire_toggle % 3) != 2 else "right"
        if likely == "right":
            return "right" if (self._reacquire_toggle % 3) != 2 else "left"
        direction = "left" if (self._reacquire_toggle % 2) == 0 else "right"
        return direction


def _assist_hint(*, proposed: ActionPlan, memory: SceneMemorySnapshot) -> bool:
    if _intent_from_action(proposed) == "assist":
        return True
    style = str(proposed.style or "").strip().lower()
    if style == "focused":
        return True
    activity = str(memory.dominant_activity or "").strip().lower()
    return activity in {"focused_work", "engaged", "transitioning"}


def _style_rate(style: str) -> float:
    token = str(style or "").strip().lower()
    if token == "focused":
        return 1.9
    if token == "curious":
        return 1.6
    return 1.3


def _style_for_mode(mode: str) -> str:
    if mode == "assist":
        return "focused"
    if mode in {"engage", "track", "reacquire"}:
        return "curious"
    return "calm"


def _intent_from_action(action: ActionPlan) -> Optional[str]:
    text = f"{action.explanation or ''} {action.style or ''} {action.primitive.value}".strip().lower()
    if any(token in text for token in ("assist", "help", "task", "focus")):
        return "assist"
    if any(token in text for token in ("ack", "greet", "welcome", "enter")):
        return "engage"
    if any(token in text for token in ("reacquire", "search", "lost")):
        return "reacquire"
    if any(token in text for token in ("track", "follow", "gaze", "orient")):
        return "track"
    return None


def _clamp(value: float, max_abs: float) -> float:
    lo = -abs(max_abs)
    hi = abs(max_abs)
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
