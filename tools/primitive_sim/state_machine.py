from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pala.behavior.decision_types import BehaviorMode, ModeSignals
from pala.behavior.governor import _allowed_primitives_for_mode
from pala.behavior.idle_engine import IdleEngine, IdleEngineConfig
from pala.behavior.mode_manager import ModeManager, ModeManagerConfig


_DEFAULT_SIGNALS: dict[str, Any] = {
    "person_present": False,
    "person_conf": 0.0,
    "activity_level": 0.0,
    "novelty": 0.0,
    "env_delta": 0.0,
    "planner_open_breaker": False,
    "perception_degraded": False,
}

_GRAPH_NODES: tuple[dict[str, str], ...] = (
    {"id": BehaviorMode.IDLE_PRESENCE.value, "label": "Idle"},
    {"id": BehaviorMode.SCAN_EXPLORE.value, "label": "Scan"},
    {"id": BehaviorMode.ENGAGE_TRACK.value, "label": "Engage"},
    {"id": BehaviorMode.ACKNOWLEDGE.value, "label": "Acknowledge"},
    {"id": BehaviorMode.RECOVER_RESET.value, "label": "Recover"},
)

_GRAPH_EDGES: tuple[dict[str, str], ...] = (
    {"from": "idle_presence", "to": "engage_track", "reason": "presence_track", "label": "person>=engage"},
    {"from": "idle_presence", "to": "acknowledge", "reason": "presence_novelty_ack", "label": "novelty>=ack"},
    {"from": "idle_presence", "to": "scan_explore", "reason": "activity_scan", "label": "activity/env delta"},
    {"from": "engage_track", "to": "acknowledge", "reason": "presence_novelty_ack", "label": "novelty>=ack"},
    {"from": "engage_track", "to": "scan_explore", "reason": "disengage_presence_drop", "label": "conf<=disengage"},
    {"from": "engage_track", "to": "idle_presence", "reason": "disengage_presence_drop", "label": "drop + low activity"},
    {"from": "acknowledge", "to": "engage_track", "reason": "presence_track", "label": "novelty drops"},
    {"from": "acknowledge", "to": "scan_explore", "reason": "ack_to_idle_or_scan", "label": "person absent"},
    {"from": "acknowledge", "to": "idle_presence", "reason": "ack_to_idle_or_scan", "label": "absent + low activity"},
    {"from": "scan_explore", "to": "engage_track", "reason": "presence_track", "label": "person>=engage"},
    {"from": "scan_explore", "to": "acknowledge", "reason": "presence_novelty_ack", "label": "person + novelty"},
    {"from": "scan_explore", "to": "idle_presence", "reason": "idle_presence", "label": "activity low"},
    {"from": "recover_reset", "to": "idle_presence", "reason": "idle_presence", "label": "health restored"},
    {"from": "recover_reset", "to": "scan_explore", "reason": "activity_scan", "label": "health + activity"},
    {"from": "recover_reset", "to": "engage_track", "reason": "presence_track", "label": "health + person"},
    {"from": "recover_reset", "to": "acknowledge", "reason": "presence_novelty_ack", "label": "health + novelty"},
)


def _clamp01(value: Any, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on", "y"}:
            return True
        if token in {"0", "false", "no", "off", "n", ""}:
            return False
    return bool(value)


def _coerce_zone_hint(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if token in {"left", "center", "right"}:
        return token
    return None


def _signals_from_mapping(raw: Mapping[str, Any] | None) -> tuple[ModeSignals, dict[str, Any]]:
    src = raw if isinstance(raw, Mapping) else {}
    normalized = {
        "person_present": _coerce_bool(src.get("person_present"), default=bool(_DEFAULT_SIGNALS["person_present"])),
        "person_conf": _clamp01(src.get("person_conf"), default=float(_DEFAULT_SIGNALS["person_conf"])),
        "activity_level": _clamp01(src.get("activity_level"), default=float(_DEFAULT_SIGNALS["activity_level"])),
        "novelty": _clamp01(src.get("novelty"), default=float(_DEFAULT_SIGNALS["novelty"])),
        "env_delta": _clamp01(src.get("env_delta"), default=float(_DEFAULT_SIGNALS["env_delta"])),
        "planner_open_breaker": _coerce_bool(
            src.get("planner_open_breaker"),
            default=bool(_DEFAULT_SIGNALS["planner_open_breaker"]),
        ),
        "perception_degraded": _coerce_bool(
            src.get("perception_degraded"),
            default=bool(_DEFAULT_SIGNALS["perception_degraded"]),
        ),
    }
    out = ModeSignals(
        person_present=bool(normalized["person_present"]),
        person_conf=float(normalized["person_conf"]),
        activity_level=float(normalized["activity_level"]),
        novelty=float(normalized["novelty"]),
        env_delta=float(normalized["env_delta"]),
        planner_open_breaker=bool(normalized["planner_open_breaker"]),
        perception_degraded=bool(normalized["perception_degraded"]),
    )
    return out, normalized


def _proposal_payload(item: Any, *, allowed: set[str]) -> dict[str, Any]:
    primitive = str(item.primitive)
    return {
        "intent": str(item.intent),
        "primitive": primitive,
        "command": dict(item.command),
        "style": str(item.style),
        "score": float(item.score),
        "confidence": float(item.confidence),
        "urgency": float(item.urgency),
        "risk": str(item.risk),
        "allow_interrupt": bool(item.allow_interrupt),
        "min_dwell_ms": item.min_dwell_ms,
        "max_duration_ms": item.max_duration_ms,
        "evidence": [str(v) for v in item.evidence],
        "rationale_short": str(item.rationale_short),
        "allowed_in_mode": primitive in allowed,
    }


@dataclass
class LampStateMachineSimulator:
    mode_config: ModeManagerConfig
    idle_config: IdleEngineConfig
    mode_manager: ModeManager
    idle_engine: IdleEngine
    tick_index: int = 0
    now_s: float = 0.0
    no_commit_s: float = 0.0

    @classmethod
    def create(
        cls,
        *,
        mode_config: ModeManagerConfig,
        idle_config: IdleEngineConfig,
    ) -> LampStateMachineSimulator:
        manager = ModeManager(mode_config)
        manager.reset(now_mono_s=0.0)
        return cls(
            mode_config=mode_config,
            idle_config=idle_config,
            mode_manager=manager,
            idle_engine=IdleEngine(idle_config),
        )

    def reset(self) -> dict[str, Any]:
        self.mode_manager = ModeManager(self.mode_config)
        self.mode_manager.reset(now_mono_s=0.0)
        self.tick_index = 0
        self.now_s = 0.0
        self.no_commit_s = 0.0
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        mode = self.mode_manager.snapshot.mode
        allowed = sorted(_allowed_primitives_for_mode(mode))
        return {
            "tick_index": int(self.tick_index),
            "now_s": float(self.now_s),
            "no_commit_s": float(self.no_commit_s),
            "mode": mode.value,
            "allowed_primitives": allowed,
        }

    def meta(self) -> dict[str, Any]:
        allowed_by_mode = {
            mode.value: sorted(_allowed_primitives_for_mode(mode))
            for mode in BehaviorMode
        }
        return {
            "modes": [mode.value for mode in BehaviorMode],
            "graph": {
                "nodes": list(_GRAPH_NODES),
                "edges": list(_GRAPH_EDGES),
            },
            "mode_config": {
                "min_mode_dwell_s": float(self.mode_config.min_mode_dwell_s),
                "engage_person_conf": float(self.mode_config.engage_person_conf),
                "disengage_person_conf": float(self.mode_config.disengage_person_conf),
                "novelty_for_ack": float(self.mode_config.novelty_for_ack),
                "activity_for_scan": float(self.mode_config.activity_for_scan),
            },
            "idle_config": {
                "idle_after_s": float(self.idle_config.idle_after_s),
                "glance_after_s": float(self.idle_config.glance_after_s),
            },
            "default_signals": dict(_DEFAULT_SIGNALS),
            "allowed_primitives_by_mode": allowed_by_mode,
            "state": self.snapshot(),
        }

    def step(
        self,
        *,
        dt_s: float,
        signals: Mapping[str, Any] | None,
        zone_hint: Any = None,
        commit: bool = False,
    ) -> dict[str, Any]:
        try:
            dt = float(dt_s)
        except Exception:
            dt = 0.35
        dt = max(0.05, dt)

        mode_signals, normalized_signals = _signals_from_mapping(signals)

        self.now_s += dt
        self.no_commit_s += dt
        self.tick_index += 1

        decision = self.mode_manager.update(now_mono_s=self.now_s, signals=mode_signals)
        mode = decision.next_mode
        allowed = set(_allowed_primitives_for_mode(mode))

        signals_map: dict[str, float | bool | None] = {
            "person_present": normalized_signals["person_present"],
            "person_conf": normalized_signals["person_conf"],
            "activity_level": normalized_signals["activity_level"],
            "novelty": normalized_signals["novelty"],
            "env_delta": normalized_signals["env_delta"],
            "planner_open_breaker": normalized_signals["planner_open_breaker"],
            "perception_degraded": normalized_signals["perception_degraded"],
        }
        zone = _coerce_zone_hint(zone_hint)
        proposals = self.idle_engine.propose(
            mode=mode,
            no_commit_s=self.no_commit_s,
            zone_hint=zone,
            tick_index=self.tick_index,
            signals=signals_map,
        )
        proposal_rows = [_proposal_payload(item, allowed=allowed) for item in proposals]
        proposal_rows = sorted(
            proposal_rows,
            key=lambda row: (
                0 if bool(row["allowed_in_mode"]) else 1,
                -float(row["score"]),
                -float(row["confidence"]),
            ),
        )

        if commit:
            self.no_commit_s = 0.0

        recommended = proposal_rows[0] if proposal_rows else None
        return {
            "tick_index": int(self.tick_index),
            "dt_s": float(dt),
            "now_s": float(self.now_s),
            "no_commit_s": float(self.no_commit_s),
            "signals": normalized_signals,
            "zone_hint": zone,
            "mode_decision": {
                "from": decision.previous_mode.value,
                "to": decision.next_mode.value,
                "reason": str(decision.reason),
                "transitioned": bool(decision.transitioned),
            },
            "allowed_primitives": sorted(allowed),
            "proposals": proposal_rows,
            "recommended": recommended,
        }
