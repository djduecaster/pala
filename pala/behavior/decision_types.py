from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BehaviorMode(str, Enum):
    IDLE_PRESENCE = "idle_presence"
    ENGAGE_TRACK = "engage_track"
    ACKNOWLEDGE = "acknowledge"
    SCAN_EXPLORE = "scan_explore"
    RECOVER_RESET = "recover_reset"


@dataclass(frozen=True)
class ModeSnapshot:
    mode: BehaviorMode
    entered_mono_s: float
    reason: str


@dataclass(frozen=True)
class ModeSignals:
    person_present: bool
    person_conf: float
    activity_level: float
    novelty: float
    env_delta: float
    planner_open_breaker: bool
    perception_degraded: bool


@dataclass(frozen=True)
class ModeDecision:
    previous_mode: BehaviorMode
    next_mode: BehaviorMode
    reason: str
    transitioned: bool


@dataclass(frozen=True)
class StalenessSignal:
    age_s: float
    penalty: float
    expired: bool


@dataclass(frozen=True)
class ArbiterDebug:
    threshold: float
    effective_current: float
    margin: float
    best_utility: float
    reason: str
    chosen_primitive: Optional[str]
