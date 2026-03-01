from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MacroMode(str, Enum):
    BOOT_AWAKEN = "boot_awaken"
    IDLE_PRESENCE = "idle_presence"
    SOCIAL_INTERACT = "social_interact"
    SEARCH_ASSIST = "search_assist"
    TASK_LIGHTING = "task_lighting"
    RETURN_HOME = "return_home"
    RECOVER_RESET = "recover_reset"


@dataclass(frozen=True)
class ModeSignalsV4:
    person_present: bool = False
    person_conf: float = 0.0
    search_requested: bool = False
    search_complete: bool = False
    assist_complete: bool = False
    user_ack: bool = False
    task_active: bool = False
    home_requested: bool = False
    home_completed: bool = False
    cancel_requested: bool = False
    startup_complete: bool = False
    health_degraded: bool = False


@dataclass(frozen=True)
class ModeSnapshotV4:
    mode: MacroMode
    entered_mono_s: float
    reason: str


@dataclass(frozen=True)
class ModeTransitionV4:
    previous_mode: MacroMode
    next_mode: MacroMode
    reason: str
    transitioned: bool
    dwell_s: float


@dataclass
class ModeFsmV4Config:
    min_mode_dwell_s: float = 1.2
    engage_person_conf: float = 0.45
    disengage_person_conf: float = 0.25
    boot_timeout_s: float = 6.0
    return_home_settle_s: float = 1.2
    recover_settle_s: float = 1.0


class ModeFsmV4:
    """Deterministic macro-mode FSM for Behavior V4."""

    def __init__(self, config: ModeFsmV4Config | None = None) -> None:
        self._cfg = config or ModeFsmV4Config()
        self._snapshot = ModeSnapshotV4(
            mode=MacroMode.BOOT_AWAKEN,
            entered_mono_s=0.0,
            reason="startup",
        )

    @property
    def snapshot(self) -> ModeSnapshotV4:
        return self._snapshot

    def reset(self, *, now_mono_s: float) -> None:
        self._snapshot = ModeSnapshotV4(
            mode=MacroMode.BOOT_AWAKEN,
            entered_mono_s=now_mono_s,
            reason="reset",
        )

    def update(self, *, now_mono_s: float, signals: ModeSignalsV4) -> ModeTransitionV4:
        prev = self._snapshot.mode
        dwell_s = max(0.0, now_mono_s - self._snapshot.entered_mono_s)

        # Health state always preempts to recover mode.
        if signals.health_degraded:
            return self._transition(
                now_mono_s=now_mono_s,
                next_mode=MacroMode.RECOVER_RESET,
                reason="health_degraded",
                dwell_s=dwell_s,
                force=True,
            )

        next_mode = prev
        reason = "hold_mode"
        force = False

        if prev == MacroMode.BOOT_AWAKEN:
            if signals.startup_complete or dwell_s >= max(0.2, float(self._cfg.boot_timeout_s)):
                next_mode = MacroMode.IDLE_PRESENCE
                reason = "startup_complete"
                force = True
        elif prev == MacroMode.RETURN_HOME:
            if signals.home_completed or dwell_s >= max(0.1, float(self._cfg.return_home_settle_s)):
                next_mode = MacroMode.IDLE_PRESENCE
                reason = "home_complete"
                force = True
        elif prev == MacroMode.RECOVER_RESET:
            if (not signals.health_degraded) and dwell_s >= max(0.0, float(self._cfg.recover_settle_s)):
                next_mode = MacroMode.IDLE_PRESENCE
                reason = "recover_to_idle"
                force = True
        elif prev == MacroMode.SEARCH_ASSIST:
            if signals.assist_complete and signals.task_active:
                next_mode = MacroMode.TASK_LIGHTING
                reason = "assist_complete_task_active"
            elif signals.search_complete:
                if signals.person_present and signals.user_ack:
                    next_mode = MacroMode.SOCIAL_INTERACT
                    reason = "search_complete_user_ack"
                else:
                    next_mode = MacroMode.IDLE_PRESENCE
                    reason = "search_complete"
            elif signals.cancel_requested:
                next_mode = MacroMode.IDLE_PRESENCE
                reason = "search_canceled"

        if next_mode == prev:
            desired, desired_reason = self._steady_state_target(prev=prev, signals=signals)
            next_mode = desired
            reason = desired_reason

        return self._transition(
            now_mono_s=now_mono_s,
            next_mode=next_mode,
            reason=reason,
            dwell_s=dwell_s,
            force=force,
        )

    def force_mode(self, *, now_mono_s: float, next_mode: MacroMode, reason: str) -> ModeTransitionV4:
        prev = self._snapshot.mode
        dwell_s = max(0.0, now_mono_s - self._snapshot.entered_mono_s)
        return self._transition(
            now_mono_s=now_mono_s,
            next_mode=next_mode,
            reason=reason,
            dwell_s=dwell_s,
            force=True,
        )

    def _steady_state_target(self, *, prev: MacroMode, signals: ModeSignalsV4) -> tuple[MacroMode, str]:
        if prev != MacroMode.BOOT_AWAKEN and signals.home_requested:
            return MacroMode.RETURN_HOME, "home_requested"

        if signals.search_requested:
            return MacroMode.SEARCH_ASSIST, "search_requested"
        if prev == MacroMode.SOCIAL_INTERACT and signals.task_active and (
            not signals.person_present or signals.person_conf <= max(0.0, float(self._cfg.disengage_person_conf))
        ):
            return MacroMode.TASK_LIGHTING, "task_active_disengaged"
        if prev == MacroMode.TASK_LIGHTING and signals.task_active and signals.person_present and (
            signals.person_conf >= max(0.0, float(self._cfg.engage_person_conf))
        ):
            return MacroMode.SOCIAL_INTERACT, "user_reengaged"
        if prev == MacroMode.TASK_LIGHTING and not signals.task_active:
            return MacroMode.IDLE_PRESENCE, "task_context_lost"
        if signals.task_active:
            return MacroMode.TASK_LIGHTING, "task_active"
        if signals.person_present and signals.person_conf >= max(0.0, float(self._cfg.engage_person_conf)):
            return MacroMode.SOCIAL_INTERACT, "person_present_engage"

        if prev == MacroMode.SOCIAL_INTERACT:
            if signals.person_conf > max(0.0, float(self._cfg.disengage_person_conf)):
                return MacroMode.SOCIAL_INTERACT, "person_hold_social"
            return MacroMode.IDLE_PRESENCE, "person_absent_timeout"

        if prev == MacroMode.TASK_LIGHTING and signals.task_active:
            return MacroMode.TASK_LIGHTING, "task_hold"

        return MacroMode.IDLE_PRESENCE, "idle_presence"

    def _transition(
        self,
        *,
        now_mono_s: float,
        next_mode: MacroMode,
        reason: str,
        dwell_s: float,
        force: bool,
    ) -> ModeTransitionV4:
        prev = self._snapshot.mode
        if next_mode != prev and (not force) and dwell_s < max(0.0, float(self._cfg.min_mode_dwell_s)):
            next_mode = prev
            reason = "min_mode_dwell_hold"

        transitioned = next_mode != prev
        if transitioned:
            self._snapshot = ModeSnapshotV4(
                mode=next_mode,
                entered_mono_s=now_mono_s,
                reason=reason,
            )
        return ModeTransitionV4(
            previous_mode=prev,
            next_mode=next_mode,
            reason=reason,
            transitioned=transitioned,
            dwell_s=dwell_s,
        )
