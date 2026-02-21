from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Callable, Optional

from ..types import ActionPlan, GlanceCommand, OrientToZoneCommand, PrimitiveKind
from .intents import BehaviorIntent
from .models import SceneObservation
from .scene_memory import SceneMemorySnapshot

_MODES = {"idle", "engage", "track", "assist", "reacquire"}


@dataclass
class IntentPlannerConfig:
    mode_hold_s: float = 1.2
    remote_high_conf: float = 0.85
    remote_bias_weight: float = 0.30
    assist_min_present_s: float = 1.5
    reacquire_absent_max_s: float = 4.0
    proactive_interval_s: float = 2.2
    act_interval_track_s: float = 1.1
    act_interval_assist_s: float = 1.8
    act_interval_idle_s: float = 3.0
    remote_first: bool = False
    remote_min_conf: float = 0.45
    remote_first_strict: bool = False
    remote_act_now_conf: float = 0.8
    remote_authoritative: bool = True


class IntentPlanner:
    """Decision-tree intent planner with guarded remote influence."""

    def __init__(
        self,
        cfg: Optional[IntentPlannerConfig] = None,
        *,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._cfg = cfg or IntentPlannerConfig()
        self._clock = clock or time.monotonic
        self._mode = "idle"
        self._mode_expires_s = 0.0
        self._last_act_s = -1_000.0

    def plan(
        self,
        *,
        proposed: ActionPlan,
        obs: SceneObservation,
        memory: SceneMemorySnapshot,
    ) -> BehaviorIntent:
        now = self._clock()
        local_mode, local_reason = _mode_from_local(obs=obs, memory=memory, cfg=self._cfg)
        remote_mode, remote_explicit = _mode_from_remote(proposed)
        remote_conf = max(0.0, min(1.0, float(proposed.confidence)))
        hard_event = obs.event in {"person_entered", "person_exited", "zone_changed"}
        remote_urgency = _extract_urgency(str(proposed.explanation or ""))
        authoritative_remote = (
            self._cfg.remote_authoritative
            and self._cfg.remote_first
            and remote_mode in _MODES
            and remote_conf >= self._cfg.remote_min_conf
        )
        scores = self._build_mode_scores(
            local_mode=local_mode,
            remote_mode=remote_mode,
            remote_conf=remote_conf,
            remote_explicit=remote_explicit,
            obs=obs,
            memory=memory,
            hard_event=hard_event,
        )

        if authoritative_remote and remote_mode is not None:
            mode = _remote_mode_with_local_safety(remote_mode=remote_mode, obs=obs, memory=memory)
            self._mode = mode
            self._mode_expires_s = now + max(0.2, float(self._cfg.mode_hold_s))
        else:
            mode = self._select_mode(
                now_s=now,
                scores=scores,
                hard_event=hard_event,
            )
        target_zone = _select_target_zone(obs=obs, memory=memory, proposed=proposed)
        conf = max(0.35, min(1.0, 0.45 * proposed.confidence + (0.55 if obs.person_present else 0.35)))
        urgency = _urgency_for(obs=obs, memory=memory, mode=mode, remote_conf=remote_conf)
        if remote_urgency is not None and remote_conf >= self._cfg.remote_min_conf:
            urgency = remote_urgency
        act_now = _should_act_now(
            obs=obs,
            memory=memory,
            mode=mode,
            remote_mode=remote_mode,
            remote_conf=remote_conf,
            hard_event=hard_event,
            remote_explicit=remote_explicit,
            proactive_interval_s=self._cfg.proactive_interval_s,
            remote_first=self._cfg.remote_first,
            remote_act_now_conf=self._cfg.remote_act_now_conf,
        )
        if (
            self._cfg.remote_first
            and remote_mode in _MODES
            and remote_conf >= self._cfg.remote_min_conf
            and remote_explicit
        ):
            act_now = True
        min_interval = _min_act_interval_s(mode=mode, cfg=self._cfg)
        if authoritative_remote and remote_explicit:
            min_interval = 0.0
        if act_now and urgency != "high" and (now - self._last_act_s) < min_interval:
            act_now = False
        if act_now:
            self._last_act_s = now
        reason = (
            f"intent:{mode};local={local_mode}[{local_reason}];"
            f"remote={remote_mode or '-'};explicit_remote={int(remote_explicit)};"
            f"authoritative_remote={int(authoritative_remote)};"
            f"event={obs.event};urgency={urgency};act_now={int(act_now)};"
            f"scores={_top_mode_scores(scores)}"
        )
        allow_interrupt = urgency in {"high", "medium"} or mode in {"engage", "track", "reacquire"}
        return BehaviorIntent(
            mode=mode,
            target_zone=target_zone,
            confidence=conf,
            reason=reason,
            allow_interrupt=allow_interrupt,
            urgency=urgency,
            act_now=act_now,
            remote_confidence=remote_conf,
        )

    def _select_mode(
        self,
        *,
        now_s: float,
        scores: dict[str, float],
        hard_event: bool,
    ) -> str:
        mode = max(scores.items(), key=lambda kv: kv[1])[0]
        if self._cfg.remote_first_strict and mode in _MODES:
            contender = scores.get(mode, 0.0)
            incumbent = scores.get(self._mode, 0.0)
            if mode != self._mode and (contender - incumbent) >= 0.02:
                self._mode = mode
                self._mode_expires_s = now_s + max(0.2, float(self._cfg.mode_hold_s))
                return mode
        if not hard_event and now_s < self._mode_expires_s and self._mode in _MODES:
            incumbent = scores.get(self._mode, 0.0)
            contender = scores.get(mode, 0.0)
            if mode != self._mode and (contender - incumbent) < 0.18:
                return self._mode

        self._mode = mode
        self._mode_expires_s = now_s + max(0.2, float(self._cfg.mode_hold_s))
        return mode

    def _build_mode_scores(
        self,
        *,
        local_mode: str,
        remote_mode: Optional[str],
        remote_conf: float,
        remote_explicit: bool,
        obs: SceneObservation,
        memory: SceneMemorySnapshot,
        hard_event: bool,
    ) -> dict[str, float]:
        scores = {mode: 0.0 for mode in _MODES}
        scores[local_mode] += 0.72

        if obs.person_present:
            scores["track"] += 0.15
            if obs.event == "person_entered":
                scores["engage"] += 0.4
            if obs.zone_changed or memory.zone_transitions_recent >= 2:
                scores["track"] += 0.28
            if memory.present_streak_s >= self._cfg.assist_min_present_s:
                activity = str(memory.dominant_activity or "").strip().lower()
                if activity in {"focused_work", "engaged", "transitioning"}:
                    scores["assist"] += 0.35
        else:
            scores["idle"] += 0.28
            if memory.person_recently_seen:
                age = memory.last_seen_age_s if memory.last_seen_age_s is not None else 0.0
                scores["reacquire"] += max(0.15, 0.55 - (0.1 * age))
                scores["reacquire"] += 0.35

        if remote_mode in _MODES:
            remote_weight = 1.1 if remote_explicit else 0.6
            scores[remote_mode] += remote_weight * remote_conf
            if remote_mode == "idle" and obs.person_present:
                scores[remote_mode] -= 0.2
            if remote_mode == "idle" and (not obs.person_present) and memory.person_recently_seen:
                scores[remote_mode] -= 0.35
            if self._cfg.remote_first and remote_conf >= self._cfg.remote_min_conf:
                scores[remote_mode] += 0.5 + (0.35 * remote_conf)
                if remote_explicit:
                    scores[remote_mode] += 0.35

        if not hard_event and self._mode in _MODES:
            scores[self._mode] += (0.02 if self._cfg.remote_first_strict else 0.12)

        return scores


def _mode_from_local(*, obs: SceneObservation, memory: SceneMemorySnapshot, cfg: IntentPlannerConfig) -> tuple[str, str]:
    # Step 1: immediate safety/availability gate.
    if not obs.person_present:
        if memory.person_recently_seen and memory.absent_streak_s <= cfg.reacquire_absent_max_s:
            return "reacquire", "person_recently_lost"
        return "idle", "no_person"

    # Step 2: high-salience events.
    if obs.event == "person_entered" or memory.present_streak_s < 0.7:
        return "engage", "recent_entry"
    if obs.zone_changed or memory.zone_transitions_recent >= 2:
        return "track", "movement_detected"

    # Step 3: task-like opportunities.
    activity = str(memory.dominant_activity or "").strip().lower()
    if activity in {"focused_work", "engaged", "transitioning"} and memory.present_streak_s >= cfg.assist_min_present_s:
        return "assist", "task_opportunity"

    # Step 4: continuity default.
    return "track", "steady_presence"


def _mode_from_remote(action: ActionPlan) -> tuple[Optional[str], bool]:
    explanation = str(action.explanation or "").strip().lower()
    explicit = _extract_explicit_mode(explanation)
    if explicit is not None:
        return explicit, True

    text = f"{explanation} {str(action.style or '').strip().lower()} {action.primitive.value}".strip().lower()
    if any(token in text for token in ("reacquire", "search", "lost")):
        return "reacquire", False
    if any(token in text for token in ("ack", "greet", "welcome", "enter")):
        return "engage", False
    if any(token in text for token in ("assist", "help", "task")):
        return "assist", False

    if action.primitive in {PrimitiveKind.HOLD, PrimitiveKind.BREATH}:
        return "idle", False
    if action.primitive in {PrimitiveKind.GAZE_TO, PrimitiveKind.ORIENT_TO_ZONE}:
        return "track", False
    if action.primitive == PrimitiveKind.NOD:
        return "engage", False
    if action.primitive == PrimitiveKind.GLANCE:
        if any(token in explanation for token in ("reacquire", "lost", "search")):
            return "reacquire", False
        return "track", False
    return None, False


def _extract_explicit_mode(text: str) -> Optional[str]:
    for match in re.findall(r"(?:target_state|state|intent|mode)\s*[:=]\s*([a-z_]+)", text):
        mode = _normalize_mode_token(match)
        if mode is not None:
            return mode
    return None


def _normalize_mode_token(token: str) -> Optional[str]:
    raw = token.strip().lower()
    mapping = {
        "idle": "idle",
        "idle_presence": "idle",
        "hold": "idle",
        "track": "track",
        "tracking": "track",
        "track_transition": "track",
        "engage": "engage",
        "engaging": "engage",
        "acknowledge": "engage",
        "assist": "assist",
        "assist_with_task": "assist",
        "help": "assist",
        "reacquire": "reacquire",
        "search": "reacquire",
    }
    return mapping.get(raw)


def _select_target_zone(*, obs: SceneObservation, memory: SceneMemorySnapshot, proposed: ActionPlan) -> Optional[str]:
    if proposed.primitive == PrimitiveKind.ORIENT_TO_ZONE and isinstance(proposed.command, OrientToZoneCommand):
        return proposed.command.zone
    if proposed.primitive == PrimitiveKind.GLANCE and isinstance(proposed.command, GlanceCommand):
        if proposed.command.direction in {"left", "right"}:
            return proposed.command.direction
    if obs.zone is not None:
        return obs.zone
    if memory.likely_zone is not None:
        return memory.likely_zone
    text = str(proposed.explanation or "").lower()
    for token in ("left", "center", "right"):
        if token in text:
            return token
    return None


def _urgency_for(
    *,
    obs: SceneObservation,
    memory: SceneMemorySnapshot,
    mode: str,
    remote_conf: float,
) -> str:
    if obs.event in {"person_entered", "person_exited"}:
        return "high"
    if mode == "reacquire":
        return "high" if memory.absent_streak_s <= 1.2 else "medium"
    if obs.zone_changed or memory.zone_transitions_recent >= 2:
        return "medium"
    if mode in {"assist", "track"} and remote_conf >= 0.8:
        return "medium"
    return "low"


def _should_act_now(
    *,
    obs: SceneObservation,
    memory: SceneMemorySnapshot,
    mode: str,
    remote_mode: Optional[str],
    remote_conf: float,
    hard_event: bool,
    remote_explicit: bool,
    proactive_interval_s: float,
    remote_first: bool,
    remote_act_now_conf: float,
) -> bool:
    if hard_event:
        return True
    if remote_first and remote_mode in _MODES and remote_conf >= remote_act_now_conf:
        return True
    if mode in {"engage", "reacquire"}:
        return True
    if remote_explicit and remote_conf >= 0.8:
        return True
    if mode == "assist":
        return memory.present_streak_s >= 1.8
    if mode == "track":
        if obs.zone_changed or memory.zone_transitions_recent >= 1:
            return True
        return memory.present_streak_s >= max(0.5, proactive_interval_s)
    return False


def _min_act_interval_s(*, mode: str, cfg: IntentPlannerConfig) -> float:
    if mode == "track":
        return cfg.act_interval_track_s
    if mode == "assist":
        return cfg.act_interval_assist_s
    if mode == "idle":
        return cfg.act_interval_idle_s
    return 0.3


def _top_mode_scores(scores: dict[str, float]) -> str:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:2]
    return ",".join(f"{k}:{v:.2f}" for k, v in top)


def _extract_urgency(text: str) -> Optional[str]:
    lowered = text.strip().lower()
    for match in re.findall(r"urgency\s*[:=]\s*([a-z_]+)", lowered):
        token = match.strip().lower()
        if token in {"low", "medium", "high"}:
            return token
    return None
