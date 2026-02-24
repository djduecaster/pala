from __future__ import annotations

from dataclasses import dataclass

from .decision_types import BehaviorMode, ModeDecision, ModeSignals, ModeSnapshot


@dataclass
class ModeManagerConfig:
    min_mode_dwell_s: float = 1.0
    engage_person_conf: float = 0.45
    disengage_person_conf: float = 0.20
    novelty_for_ack: float = 0.45
    activity_for_scan: float = 0.30


class ModeManager:
    """Deterministic mode FSM with moderate hysteresis."""

    def __init__(self, config: ModeManagerConfig | None = None):
        self._cfg = config or ModeManagerConfig()
        self._snapshot = ModeSnapshot(
            mode=BehaviorMode.IDLE_PRESENCE,
            entered_mono_s=0.0,
            reason="startup",
        )

    @property
    def snapshot(self) -> ModeSnapshot:
        return self._snapshot

    def reset(self, *, now_mono_s: float) -> None:
        self._snapshot = ModeSnapshot(
            mode=BehaviorMode.IDLE_PRESENCE,
            entered_mono_s=now_mono_s,
            reason="reset",
        )

    def update(self, *, now_mono_s: float, signals: ModeSignals) -> ModeDecision:
        prev = self._snapshot.mode
        dwell_s = max(0.0, now_mono_s - self._snapshot.entered_mono_s)

        # Health dominates behavior state.
        if signals.planner_open_breaker or signals.perception_degraded:
            return self._transition(now_mono_s=now_mono_s, next_mode=BehaviorMode.RECOVER_RESET, reason="health_degraded")

        next_mode = prev
        reason = "hold_mode"

        if signals.person_present and signals.person_conf >= self._cfg.engage_person_conf:
            if signals.novelty >= self._cfg.novelty_for_ack:
                next_mode = BehaviorMode.ACKNOWLEDGE
                reason = "presence_novelty_ack"
            else:
                next_mode = BehaviorMode.ENGAGE_TRACK
                reason = "presence_track"
        else:
            if signals.activity_level >= self._cfg.activity_for_scan or signals.env_delta >= 0.35:
                next_mode = BehaviorMode.SCAN_EXPLORE
                reason = "activity_scan"
            else:
                next_mode = BehaviorMode.IDLE_PRESENCE
                reason = "idle_presence"

        # Acknowledge is transient; fall through to engage once novelty drops.
        if prev == BehaviorMode.ACKNOWLEDGE and not signals.person_present:
            next_mode = BehaviorMode.SCAN_EXPLORE if signals.activity_level >= self._cfg.activity_for_scan else BehaviorMode.IDLE_PRESENCE
            reason = "ack_to_idle_or_scan"

        if prev == BehaviorMode.ENGAGE_TRACK and signals.person_conf <= self._cfg.disengage_person_conf:
            next_mode = BehaviorMode.SCAN_EXPLORE if signals.activity_level >= self._cfg.activity_for_scan else BehaviorMode.IDLE_PRESENCE
            reason = "disengage_presence_drop"

        if next_mode != prev and dwell_s < max(0.0, self._cfg.min_mode_dwell_s):
            next_mode = prev
            reason = "min_dwell_hold"

        return self._transition(now_mono_s=now_mono_s, next_mode=next_mode, reason=reason)

    def _transition(self, *, now_mono_s: float, next_mode: BehaviorMode, reason: str) -> ModeDecision:
        prev = self._snapshot.mode
        transitioned = next_mode != prev
        if transitioned:
            self._snapshot = ModeSnapshot(mode=next_mode, entered_mono_s=now_mono_s, reason=reason)
        return ModeDecision(previous_mode=prev, next_mode=next_mode, reason=reason, transitioned=transitioned)
