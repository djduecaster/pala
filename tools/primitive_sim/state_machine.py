from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pala.behavior.mode_fsm_v4 import MacroMode, ModeFsmV4, ModeFsmV4Config, ModeSignalsV4
from pala.behavior.skills_v4 import (
    allowed_primitives_for,
    allowed_skills_for_mode,
    default_action_payload_for_mode,
    default_skill_for_mode,
    skill_spec_v4,
)


_MACRO_MODES: tuple[MacroMode, ...] = (
    MacroMode.BOOT_AWAKEN,
    MacroMode.IDLE_PRESENCE,
    MacroMode.SOCIAL_INTERACT,
    MacroMode.SEARCH_ASSIST,
    MacroMode.TASK_LIGHTING,
    MacroMode.RETURN_HOME,
    MacroMode.RECOVER_RESET,
)

_MODE_LABELS: dict[MacroMode, str] = {
    MacroMode.BOOT_AWAKEN: "Boot / Awaken",
    MacroMode.IDLE_PRESENCE: "Idle Presence",
    MacroMode.SOCIAL_INTERACT: "Social Interact",
    MacroMode.SEARCH_ASSIST: "Search Assist",
    MacroMode.TASK_LIGHTING: "Task Lighting",
    MacroMode.RETURN_HOME: "Return Home",
    MacroMode.RECOVER_RESET: "Recover / Reset",
}

_PRIMITIVE_ORDER: tuple[str, ...] = (
    "hold",
    "home",
    "move_to",
    "gaze_to",
    "glance",
    "nod",
    "breath",
    "orient_to_zone",
)

_DEFAULT_COMMANDS: dict[str, dict[str, Any]] = {
    "hold": {},
    "home": {"rate_rad_s": 1.0},
    "move_to": {"target_rad": [0.0, 0.0, 0.0, 0.0, 0.0], "relative": False, "rate_rad_s": 1.0, "timeout_s": 2.0},
    "gaze_to": {"yaw_rad": 0.0, "pitch_rad": 0.0, "rate_rad_s": 1.2, "dwell_s": 0.1, "timeout_s": 1.5},
    "glance": {"direction": "left", "amp_rad": 0.22, "duration_s": 0.6, "rate_rad_s": 1.2},
    "nod": {"amp_rad": 0.18, "duration_s": 0.9, "cycles": 1, "rate_rad_s": 1.8},
    "breath": {"amp_rad": 0.07, "period_s": 6.0, "rate_rad_s": 0.9},
    "orient_to_zone": {"zone": "center", "amp_rad": 0.2, "rate_rad_s": 1.1},
}

_DEFAULT_SIGNALS: dict[str, Any] = {
    "person_present": False,
    "person_conf": 0.0,
    "search_requested": False,
    "search_complete": False,
    "assist_complete": False,
    "user_ack": False,
    "task_active": False,
    "home_requested": False,
    "home_completed": False,
    "cancel_requested": False,
    "startup_complete": False,
    "health_degraded": False,
    # Legacy aliases retained for older UI payloads.
    "activity_level": 0.0,
    "novelty": 0.0,
    "env_delta": 0.0,
    "planner_open_breaker": False,
    "perception_degraded": False,
}

_SM_OVERRIDE_KEYS: tuple[str, ...] = (
    "min_mode_dwell_s",
    "engage_person_conf",
    "disengage_person_conf",
    "boot_timeout_s",
    "return_home_settle_s",
    "recover_settle_s",
    "search_request_activity",
    "user_ack_novelty",
    "task_active_env_delta",
)

_SM_OVERRIDE_ALIASES: dict[str, str] = {
    "mode_min_dwell_s": "min_mode_dwell_s",
    "mode_engage_person_conf": "engage_person_conf",
    "mode_disengage_person_conf": "disengage_person_conf",
    "activity_for_scan": "search_request_activity",
    "novelty_for_ack": "user_ack_novelty",
}


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on", "y"}:
            return True
        if token in {"0", "false", "no", "off", "n", ""}:
            return False
    return bool(value)


def _clamp01(value: Any, default: float = 0.0) -> float:
    out = _coerce_float(value, default)
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _extract_sim_overrides(raw_config: Mapping[str, Any]) -> dict[str, float]:
    candidates: list[Any] = []

    root_section = raw_config.get("state_machine")
    if isinstance(root_section, Mapping):
        candidates.append(root_section)

    primitive_sim = raw_config.get("primitive_sim")
    if isinstance(primitive_sim, Mapping):
        nested = primitive_sim.get("state_machine")
        if isinstance(nested, Mapping):
            candidates.append(nested)

    tools = raw_config.get("tools")
    if isinstance(tools, Mapping):
        tools_ps = tools.get("primitive_sim")
        if isinstance(tools_ps, Mapping):
            nested = tools_ps.get("state_machine")
            if isinstance(nested, Mapping):
                candidates.append(nested)

    source = next((item for item in candidates if isinstance(item, Mapping)), None)
    if not isinstance(source, Mapping):
        return {}

    out: dict[str, float] = {}
    for key in _SM_OVERRIDE_KEYS:
        raw = source.get(key)
        if raw is None:
            continue
        try:
            out[key] = float(raw)
        except Exception:
            continue
    for alias, canonical in _SM_OVERRIDE_ALIASES.items():
        if canonical in out:
            continue
        raw = source.get(alias)
        if raw is None:
            continue
        try:
            out[canonical] = float(raw)
        except Exception:
            continue
    return out


def _build_mode_fsm_config(cfg: Any, raw_config: Mapping[str, Any]) -> tuple[ModeFsmV4Config, dict[str, float]]:
    defaults = ModeFsmV4Config()
    cosmos = getattr(cfg, "cosmos", None)
    overrides = _extract_sim_overrides(raw_config)

    min_mode_dwell_s = float(defaults.min_mode_dwell_s)
    engage_person_conf = float(defaults.engage_person_conf)
    disengage_person_conf = float(defaults.disengage_person_conf)
    boot_timeout_s = float(defaults.boot_timeout_s)
    return_home_settle_s = float(defaults.return_home_settle_s)
    recover_settle_s = float(defaults.recover_settle_s)

    if cosmos is not None:
        min_mode_dwell_s = _coerce_float(getattr(cosmos, "mode_min_dwell_s", min_mode_dwell_s), min_mode_dwell_s)
        engage_person_conf = _coerce_float(
            getattr(cosmos, "mode_engage_person_conf", engage_person_conf),
            engage_person_conf,
        )
        disengage_person_conf = _coerce_float(
            getattr(cosmos, "mode_disengage_person_conf", disengage_person_conf),
            disengage_person_conf,
        )
        boot_timeout_s = _coerce_float(
            _coerce_float(getattr(cosmos, "startup_wake_settle_s", 0.7), 0.7) + 6.0,
            boot_timeout_s,
        )

    min_mode_dwell_s = _coerce_float(overrides.get("min_mode_dwell_s"), min_mode_dwell_s)
    engage_person_conf = _coerce_float(overrides.get("engage_person_conf"), engage_person_conf)
    disengage_person_conf = _coerce_float(overrides.get("disengage_person_conf"), disengage_person_conf)
    boot_timeout_s = _coerce_float(overrides.get("boot_timeout_s"), boot_timeout_s)
    return_home_settle_s = _coerce_float(overrides.get("return_home_settle_s"), return_home_settle_s)
    recover_settle_s = _coerce_float(overrides.get("recover_settle_s"), recover_settle_s)

    config = ModeFsmV4Config(
        min_mode_dwell_s=max(0.0, min_mode_dwell_s),
        engage_person_conf=_clamp01(engage_person_conf, defaults.engage_person_conf),
        disengage_person_conf=_clamp01(disengage_person_conf, defaults.disengage_person_conf),
        boot_timeout_s=max(0.2, boot_timeout_s),
        return_home_settle_s=max(0.0, return_home_settle_s),
        recover_settle_s=max(0.0, recover_settle_s),
    )
    return config, overrides


def _state_graph() -> dict[str, Any]:
    nodes = [{"id": mode.value, "label": _MODE_LABELS[mode]} for mode in _MACRO_MODES]

    edges: list[dict[str, str]] = []

    def add_edge(from_mode: MacroMode, to_mode: MacroMode, reason: str, label: str | None = None) -> None:
        edges.append(
            {
                "from": from_mode.value,
                "to": to_mode.value,
                "reason": reason,
                "label": label or reason,
            }
        )

    add_edge(MacroMode.BOOT_AWAKEN, MacroMode.IDLE_PRESENCE, "startup_complete")
    add_edge(MacroMode.IDLE_PRESENCE, MacroMode.SOCIAL_INTERACT, "person_present_engage")
    add_edge(MacroMode.SOCIAL_INTERACT, MacroMode.IDLE_PRESENCE, "person_absent_timeout")
    add_edge(MacroMode.IDLE_PRESENCE, MacroMode.SEARCH_ASSIST, "search_requested")
    add_edge(MacroMode.SOCIAL_INTERACT, MacroMode.SEARCH_ASSIST, "search_requested")
    add_edge(MacroMode.TASK_LIGHTING, MacroMode.SEARCH_ASSIST, "search_requested")
    add_edge(MacroMode.SEARCH_ASSIST, MacroMode.IDLE_PRESENCE, "search_complete")
    add_edge(MacroMode.SEARCH_ASSIST, MacroMode.SOCIAL_INTERACT, "search_complete_user_ack")
    add_edge(MacroMode.SEARCH_ASSIST, MacroMode.TASK_LIGHTING, "assist_complete_task_active")
    add_edge(MacroMode.SEARCH_ASSIST, MacroMode.IDLE_PRESENCE, "search_canceled")
    add_edge(MacroMode.IDLE_PRESENCE, MacroMode.TASK_LIGHTING, "task_active")
    add_edge(MacroMode.SOCIAL_INTERACT, MacroMode.TASK_LIGHTING, "task_active_disengaged")
    add_edge(MacroMode.TASK_LIGHTING, MacroMode.SOCIAL_INTERACT, "user_reengaged")
    add_edge(MacroMode.TASK_LIGHTING, MacroMode.IDLE_PRESENCE, "task_context_lost")
    add_edge(MacroMode.IDLE_PRESENCE, MacroMode.RETURN_HOME, "home_requested")
    add_edge(MacroMode.SOCIAL_INTERACT, MacroMode.RETURN_HOME, "home_requested")
    add_edge(MacroMode.SEARCH_ASSIST, MacroMode.RETURN_HOME, "home_requested")
    add_edge(MacroMode.TASK_LIGHTING, MacroMode.RETURN_HOME, "home_requested")
    add_edge(MacroMode.RETURN_HOME, MacroMode.IDLE_PRESENCE, "home_complete")
    add_edge(MacroMode.RECOVER_RESET, MacroMode.IDLE_PRESENCE, "recover_to_idle")
    for mode in _MACRO_MODES:
        if mode == MacroMode.RECOVER_RESET:
            continue
        add_edge(mode, MacroMode.RECOVER_RESET, "health_degraded")
    return {"nodes": nodes, "edges": edges}


def _ordered_primitives(primitives: set[str], preferred: str | None = None) -> list[str]:
    out = [p for p in _PRIMITIVE_ORDER if p in primitives]
    for p in sorted(primitives):
        if p not in out:
            out.append(p)
    if preferred and preferred in out:
        out.remove(preferred)
        out.insert(0, preferred)
    return out


def _parse_mode_token(value: Any) -> MacroMode:
    token = str(value or "").strip().lower()
    if not token:
        raise ValueError("mode token is required")
    try:
        return MacroMode(token)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in _MACRO_MODES)
        raise ValueError(f"invalid mode token '{token}' (allowed: {allowed})") from exc


@dataclass
class LampStateMachineSimulator:
    mode_fsm_config: ModeFsmV4Config
    signal_thresholds: dict[str, float] = field(default_factory=dict)
    tick_index: int = 0
    now_s: float = 0.0
    no_commit_s: float = 0.0
    _fsm: ModeFsmV4 = field(init=False)
    _last_signals: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_SIGNALS))
    _last_transition: dict[str, Any] = field(
        default_factory=lambda: {
            "from": MacroMode.BOOT_AWAKEN.value,
            "to": MacroMode.BOOT_AWAKEN.value,
            "reason": "startup",
            "transitioned": False,
            "dwell_s": 0.0,
        }
    )

    def __post_init__(self) -> None:
        self._fsm = ModeFsmV4(config=self.mode_fsm_config)
        self._fsm.reset(now_mono_s=0.0)

    @classmethod
    def create(
        cls,
        *,
        mode_config: Any | None = None,
        idle_config: Any | None = None,
        signal_thresholds: Mapping[str, Any] | None = None,
    ) -> "LampStateMachineSimulator":
        del idle_config
        config = mode_config if isinstance(mode_config, ModeFsmV4Config) else ModeFsmV4Config()
        thresholds: dict[str, float] = {
            "search_request_activity": 0.65,
            "user_ack_novelty": 0.45,
            "task_active_env_delta": 0.65,
        }
        if isinstance(signal_thresholds, Mapping):
            for key in ("search_request_activity", "user_ack_novelty", "task_active_env_delta"):
                raw = signal_thresholds.get(key)
                if raw is None:
                    continue
                thresholds[key] = _clamp01(raw, thresholds[key])
        return cls(mode_fsm_config=config, signal_thresholds=thresholds)

    def _normalize_signals(self, raw_signals: Mapping[str, Any] | None) -> tuple[ModeSignalsV4, dict[str, Any]]:
        signals = raw_signals or {}

        person_conf = _clamp01(signals.get("person_conf"), 0.0)
        person_present = _coerce_bool(signals.get("person_present"), person_conf > 0.15)

        activity_level = _clamp01(signals.get("activity_level"), 0.0)
        novelty = _clamp01(signals.get("novelty"), 0.0)
        env_delta = _clamp01(signals.get("env_delta"), 0.0)

        search_requested = _coerce_bool(
            signals.get("search_requested"),
            activity_level >= float(self.signal_thresholds.get("search_request_activity", 0.65)),
        )
        user_ack = _coerce_bool(
            signals.get("user_ack"),
            novelty >= float(self.signal_thresholds.get("user_ack_novelty", 0.45)),
        )
        task_active = _coerce_bool(
            signals.get("task_active"),
            env_delta >= float(self.signal_thresholds.get("task_active_env_delta", 0.65)),
        )

        legacy_breaker = _coerce_bool(signals.get("planner_open_breaker"), False) or _coerce_bool(
            signals.get("perception_degraded"),
            False,
        )
        health_degraded = _coerce_bool(signals.get("health_degraded"), legacy_breaker)

        startup_complete = _coerce_bool(signals.get("startup_complete"), False)

        mode_signals = ModeSignalsV4(
            person_present=person_present,
            person_conf=person_conf,
            search_requested=search_requested,
            search_complete=_coerce_bool(signals.get("search_complete"), False),
            assist_complete=_coerce_bool(signals.get("assist_complete"), False),
            user_ack=user_ack,
            task_active=task_active,
            home_requested=_coerce_bool(signals.get("home_requested"), False),
            home_completed=_coerce_bool(signals.get("home_completed"), False),
            cancel_requested=_coerce_bool(signals.get("cancel_requested"), False),
            startup_complete=startup_complete,
            health_degraded=health_degraded,
        )

        normalized = {
            "person_present": bool(mode_signals.person_present),
            "person_conf": float(mode_signals.person_conf),
            "search_requested": bool(mode_signals.search_requested),
            "search_complete": bool(mode_signals.search_complete),
            "assist_complete": bool(mode_signals.assist_complete),
            "user_ack": bool(mode_signals.user_ack),
            "task_active": bool(mode_signals.task_active),
            "home_requested": bool(mode_signals.home_requested),
            "home_completed": bool(mode_signals.home_completed),
            "cancel_requested": bool(mode_signals.cancel_requested),
            "startup_complete": bool(mode_signals.startup_complete),
            "health_degraded": bool(mode_signals.health_degraded),
            # Legacy aliases kept for older pages.
            "activity_level": activity_level,
            "novelty": novelty,
            "env_delta": env_delta,
            "planner_open_breaker": legacy_breaker,
            "perception_degraded": _coerce_bool(signals.get("perception_degraded"), False),
        }
        return mode_signals, normalized

    def _default_command_for_primitive(self, primitive: str, *, zone_hint: str | None = None) -> dict[str, Any]:
        base = dict(_DEFAULT_COMMANDS.get(primitive, {}))
        if primitive == "orient_to_zone":
            zone = str(zone_hint or base.get("zone") or "center").strip().lower()
            if zone not in {"left", "center", "right"}:
                zone = "center"
            base["zone"] = zone
        return base

    def _build_proposals(
        self,
        *,
        mode: MacroMode,
        transition_reason: str,
        transitioned: bool,
        zone_hint: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
        active_skill = default_skill_for_mode(mode)
        spec = skill_spec_v4(active_skill)
        allowed = allowed_primitives_for(mode, active_skill)
        fallback_payload = default_action_payload_for_mode(mode, reason=transition_reason)
        recommended_primitive = str(fallback_payload.get("primitive") or "hold")
        recommended_style = str(fallback_payload.get("style") or "calm")
        ordered_primitives = list(spec.allowed_primitives) if spec is not None else _ordered_primitives(allowed)
        if recommended_primitive not in ordered_primitives:
            ordered_primitives = _ordered_primitives(set(ordered_primitives) | set(allowed), recommended_primitive)
        else:
            ordered_primitives = _ordered_primitives(set(ordered_primitives), recommended_primitive)

        urgency = 0.65 if transitioned else 0.35
        evidence = [
            f"mode:{mode.value}",
            f"skill:{active_skill}",
            f"reason:{transition_reason}",
        ]

        proposals: list[dict[str, Any]] = []
        for idx, primitive in enumerate(ordered_primitives):
            command = self._default_command_for_primitive(primitive, zone_hint=zone_hint if primitive == "orient_to_zone" else None)
            style = recommended_style
            confidence = max(0.15, min(1.0, _coerce_float(fallback_payload.get("confidence"), 0.4) - (0.08 * idx)))
            score = max(0.2, 1.0 - (0.13 * idx))
            proposal = {
                "intent": mode.value,
                "primitive": primitive,
                "command": command,
                "style": style,
                "score": float(score),
                "confidence": float(confidence),
                "urgency": float(urgency),
                "risk": "low",
                "allow_interrupt": True,
                "min_dwell_ms": int(max(0.0, self.mode_fsm_config.min_mode_dwell_s) * 1000.0),
                "max_duration_ms": None,
                "evidence": list(evidence),
                "rationale_short": f"{mode.value}:{active_skill}:{transition_reason}",
                "allowed_in_mode": primitive in allowed,
            }
            proposals.append(proposal)

        recommended = proposals[0] if proposals else {
            "intent": mode.value,
            "primitive": "hold",
            "command": {},
            "style": "calm",
            "score": 0.1,
            "confidence": 0.1,
            "urgency": urgency,
            "risk": "low",
            "allow_interrupt": True,
            "min_dwell_ms": int(max(0.0, self.mode_fsm_config.min_mode_dwell_s) * 1000.0),
            "max_duration_ms": None,
            "evidence": list(evidence),
            "rationale_short": f"{mode.value}:{active_skill}:{transition_reason}",
            "allowed_in_mode": True,
        }
        return proposals, recommended, active_skill

    def snapshot(self) -> dict[str, Any]:
        mode = self._fsm.snapshot.mode
        active_skill = default_skill_for_mode(mode)
        allowed = sorted(allowed_primitives_for(mode, active_skill))
        return {
            "tick_index": int(self.tick_index),
            "now_s": float(self.now_s),
            "no_commit_s": float(self.no_commit_s),
            "mode": mode.value,
            "active_skill": active_skill,
            "allowed_skills": sorted(allowed_skills_for_mode(mode)),
            "allowed_primitives": allowed,
            "mode_reason": str(self._fsm.snapshot.reason),
            "entered_s": float(self._fsm.snapshot.entered_mono_s),
        }

    def meta(self) -> dict[str, Any]:
        allowed_by_mode: dict[str, list[str]] = {}
        skills_by_mode: dict[str, list[str]] = {}
        default_skill: dict[str, str] = {}
        for mode in _MACRO_MODES:
            skill = default_skill_for_mode(mode)
            default_skill[mode.value] = skill
            skills_by_mode[mode.value] = sorted(allowed_skills_for_mode(mode))
            allowed_by_mode[mode.value] = sorted(allowed_primitives_for(mode, skill))
        return {
            "legacy_disabled": False,
            "message": "Behavior V4 simulator active.",
            "modes": [mode.value for mode in _MACRO_MODES],
            "graph": _state_graph(),
            "default_signals": dict(_DEFAULT_SIGNALS),
            "signal_thresholds": dict(self.signal_thresholds),
            "allowed_primitives_by_mode": allowed_by_mode,
            "allowed_skills_by_mode": skills_by_mode,
            "default_skill_by_mode": default_skill,
            "mode_fsm_config": {
                "min_mode_dwell_s": float(self.mode_fsm_config.min_mode_dwell_s),
                "engage_person_conf": float(self.mode_fsm_config.engage_person_conf),
                "disengage_person_conf": float(self.mode_fsm_config.disengage_person_conf),
                "boot_timeout_s": float(self.mode_fsm_config.boot_timeout_s),
                "return_home_settle_s": float(self.mode_fsm_config.return_home_settle_s),
                "recover_settle_s": float(self.mode_fsm_config.recover_settle_s),
            },
            "state": self.snapshot(),
        }

    def reset(self) -> dict[str, Any]:
        self.tick_index = 0
        self.now_s = 0.0
        self.no_commit_s = 0.0
        self._last_signals = dict(_DEFAULT_SIGNALS)
        self._last_transition = {
            "from": MacroMode.BOOT_AWAKEN.value,
            "to": MacroMode.BOOT_AWAKEN.value,
            "reason": "reset",
            "transitioned": False,
            "dwell_s": 0.0,
        }
        self._fsm.reset(now_mono_s=0.0)
        return self.snapshot()

    def step(
        self,
        *,
        dt_s: float,
        signals: Mapping[str, Any] | None,
        zone_hint: Any = None,
        commit: bool = False,
    ) -> dict[str, Any]:
        dt = max(0.05, _coerce_float(dt_s, 0.35))
        self.now_s += dt
        self.tick_index += 1
        if commit:
            self.no_commit_s = 0.0
        else:
            self.no_commit_s += dt

        mode_signals, normalized_signals = self._normalize_signals(signals)
        transition = self._fsm.update(now_mono_s=self.now_s, signals=mode_signals)
        self._last_signals = normalized_signals
        self._last_transition = {
            "from": transition.previous_mode.value,
            "to": transition.next_mode.value,
            "reason": transition.reason,
            "transitioned": bool(transition.transitioned),
            "dwell_s": float(transition.dwell_s),
        }

        proposals, recommended, active_skill = self._build_proposals(
            mode=transition.next_mode,
            transition_reason=transition.reason,
            transitioned=transition.transitioned,
            zone_hint=zone_hint,
        )
        return self._result_payload(
            dt=dt,
            normalized_signals=normalized_signals,
            zone_hint=zone_hint,
            active_skill=active_skill,
            mode=transition.next_mode,
            proposals=proposals,
            recommended=recommended,
        )

    def force_mode(
        self,
        *,
        next_mode: Any,
        reason: Any = "force_mode",
        dt_s: float = 0.0,
        signals: Mapping[str, Any] | None = None,
        zone_hint: Any = None,
        commit: bool = False,
    ) -> dict[str, Any]:
        dt = max(0.0, _coerce_float(dt_s, 0.0))
        self.now_s += dt
        self.tick_index += 1
        if commit:
            self.no_commit_s = 0.0
        else:
            self.no_commit_s += dt

        mode = _parse_mode_token(next_mode)
        reason_token = str(reason or "force_mode").strip() or "force_mode"
        mode_signals, normalized_signals = self._normalize_signals(signals)
        self._last_signals = normalized_signals
        transition = self._fsm.force_mode(
            now_mono_s=self.now_s,
            next_mode=mode,
            reason=reason_token,
        )
        self._last_transition = {
            "from": transition.previous_mode.value,
            "to": transition.next_mode.value,
            "reason": transition.reason,
            "transitioned": bool(transition.transitioned),
            "dwell_s": float(transition.dwell_s),
        }

        if mode_signals.health_degraded and transition.next_mode != MacroMode.RECOVER_RESET:
            # Keep explicit health context visible even when operator forces another mode.
            normalized_signals["health_degraded"] = True

        proposals, recommended, active_skill = self._build_proposals(
            mode=transition.next_mode,
            transition_reason=transition.reason,
            transitioned=transition.transitioned,
            zone_hint=zone_hint,
        )
        return self._result_payload(
            dt=dt,
            normalized_signals=normalized_signals,
            zone_hint=zone_hint,
            active_skill=active_skill,
            mode=transition.next_mode,
            proposals=proposals,
            recommended=recommended,
        )

    def _result_payload(
        self,
        *,
        dt: float,
        normalized_signals: Mapping[str, Any],
        zone_hint: Any,
        active_skill: str,
        mode: MacroMode,
        proposals: list[dict[str, Any]],
        recommended: dict[str, Any],
    ) -> dict[str, Any]:
        allowed = sorted(allowed_primitives_for(mode, active_skill))
        allowed_skills = sorted(allowed_skills_for_mode(mode))
        return {
            "tick_index": int(self.tick_index),
            "dt_s": float(dt),
            "now_s": float(self.now_s),
            "no_commit_s": float(self.no_commit_s),
            "signals": dict(normalized_signals),
            "zone_hint": str(zone_hint) if zone_hint is not None and str(zone_hint).strip() else None,
            "mode_decision": dict(self._last_transition),
            "mode_reason": str(self._fsm.snapshot.reason),
            "active_skill": active_skill,
            "allowed_skills": allowed_skills,
            "allowed_primitives": allowed,
            "proposals": proposals,
            "recommended": recommended,
        }


def create_state_machine_simulator(*, cfg: Any, raw_config: Mapping[str, Any]) -> LampStateMachineSimulator:
    mode_fsm_cfg, overrides = _build_mode_fsm_config(cfg, raw_config)
    signal_thresholds = {
        "search_request_activity": _clamp01(overrides.get("search_request_activity"), 0.65),
        "user_ack_novelty": _clamp01(overrides.get("user_ack_novelty"), 0.45),
        "task_active_env_delta": _clamp01(overrides.get("task_active_env_delta"), 0.65),
    }
    return LampStateMachineSimulator.create(
        mode_config=mode_fsm_cfg,
        idle_config=None,
        signal_thresholds=signal_thresholds,
    )
