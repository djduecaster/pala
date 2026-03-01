from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set

from .mode_fsm_v4 import MacroMode


@dataclass(frozen=True)
class SkillSpecV4:
    name: str
    mode: MacroMode
    allowed_primitives: tuple[str, ...]
    max_dwell_s: float
    timeout_fallback: str


_SKILL_SPECS: Dict[str, SkillSpecV4] = {
    "wake_sequence": SkillSpecV4(
        name="wake_sequence",
        mode=MacroMode.BOOT_AWAKEN,
        allowed_primitives=("hold", "glance", "orient_to_zone", "nod", "breath"),
        max_dwell_s=6.0,
        timeout_fallback="observe_settle",
    ),
    "observe_settle": SkillSpecV4(
        name="observe_settle",
        mode=MacroMode.BOOT_AWAKEN,
        allowed_primitives=("hold", "breath", "orient_to_zone"),
        max_dwell_s=4.0,
        timeout_fallback="observe_settle",
    ),
    "idle_presence": SkillSpecV4(
        name="idle_presence",
        mode=MacroMode.IDLE_PRESENCE,
        allowed_primitives=("hold", "breath", "glance"),
        max_dwell_s=12.0,
        timeout_fallback="idle_scan",
    ),
    "idle_scan": SkillSpecV4(
        name="idle_scan",
        mode=MacroMode.IDLE_PRESENCE,
        allowed_primitives=("glance", "orient_to_zone", "hold"),
        max_dwell_s=8.0,
        timeout_fallback="idle_presence",
    ),
    "greet_user": SkillSpecV4(
        name="greet_user",
        mode=MacroMode.SOCIAL_INTERACT,
        allowed_primitives=("orient_to_zone", "nod", "glance", "hold"),
        max_dwell_s=6.0,
        timeout_fallback="social_ack",
    ),
    "social_ack": SkillSpecV4(
        name="social_ack",
        mode=MacroMode.SOCIAL_INTERACT,
        allowed_primitives=("nod", "glance", "breath", "hold"),
        max_dwell_s=8.0,
        timeout_fallback="social_ack",
    ),
    "expressive_search": SkillSpecV4(
        name="expressive_search",
        mode=MacroMode.SEARCH_ASSIST,
        allowed_primitives=("orient_to_zone", "glance", "hold"),
        max_dwell_s=15.0,
        timeout_fallback="return_home",
    ),
    "point_and_hold": SkillSpecV4(
        name="point_and_hold",
        mode=MacroMode.SEARCH_ASSIST,
        allowed_primitives=("orient_to_zone", "hold", "nod"),
        max_dwell_s=10.0,
        timeout_fallback="return_home",
    ),
    "task_light_adjust": SkillSpecV4(
        name="task_light_adjust",
        mode=MacroMode.TASK_LIGHTING,
        allowed_primitives=("orient_to_zone", "hold", "breath"),
        max_dwell_s=20.0,
        timeout_fallback="return_home",
    ),
    "return_home": SkillSpecV4(
        name="return_home",
        mode=MacroMode.RETURN_HOME,
        allowed_primitives=("home", "hold"),
        max_dwell_s=8.0,
        timeout_fallback="return_home",
    ),
    "recover_hold": SkillSpecV4(
        name="recover_hold",
        mode=MacroMode.RECOVER_RESET,
        allowed_primitives=("home", "hold", "breath"),
        max_dwell_s=10.0,
        timeout_fallback="recover_hold",
    ),
}

_MODE_DEFAULT_SKILL: Dict[MacroMode, str] = {
    MacroMode.BOOT_AWAKEN: "wake_sequence",
    MacroMode.IDLE_PRESENCE: "idle_presence",
    MacroMode.SOCIAL_INTERACT: "greet_user",
    MacroMode.SEARCH_ASSIST: "expressive_search",
    MacroMode.TASK_LIGHTING: "task_light_adjust",
    MacroMode.RETURN_HOME: "return_home",
    MacroMode.RECOVER_RESET: "recover_hold",
}

_MODE_DEFAULT_ACTIONS: Dict[MacroMode, Dict[str, Any]] = {
    MacroMode.BOOT_AWAKEN: {"primitive": "hold", "command": {}, "style": "curious", "confidence": 0.20},
    MacroMode.IDLE_PRESENCE: {
        "primitive": "breath",
        "command": {"amp_rad": 0.07, "period_s": 6.0, "rate_rad_s": 0.9},
        "style": "calm",
        "confidence": 0.25,
    },
    MacroMode.SOCIAL_INTERACT: {
        "primitive": "orient_to_zone",
        "command": {"zone": "center", "amp_rad": 0.20, "rate_rad_s": 1.0},
        "style": "curious",
        "confidence": 0.40,
    },
    MacroMode.SEARCH_ASSIST: {
        "primitive": "glance",
        "command": {"direction": "left", "amp_rad": 0.22, "duration_s": 0.5, "rate_rad_s": 1.2},
        "style": "focused",
        "confidence": 0.40,
    },
    MacroMode.TASK_LIGHTING: {
        "primitive": "orient_to_zone",
        "command": {"zone": "center", "amp_rad": 0.18, "rate_rad_s": 1.1},
        "style": "focused",
        "confidence": 0.40,
    },
    MacroMode.RETURN_HOME: {
        "primitive": "home",
        "command": {"rate_rad_s": 1.0},
        "style": "calm",
        "confidence": 0.35,
    },
    MacroMode.RECOVER_RESET: {
        "primitive": "hold",
        "command": {},
        "style": "calm",
        "confidence": 0.20,
    },
}

_MODE_ALLOWED_MOODS: Dict[MacroMode, Set[str]] = {
    MacroMode.BOOT_AWAKEN: {"curious"},
    MacroMode.IDLE_PRESENCE: {"calm", "curious"},
    MacroMode.SOCIAL_INTERACT: {"calm", "curious", "excited"},
    MacroMode.SEARCH_ASSIST: {"focused", "curious"},
    MacroMode.TASK_LIGHTING: {"focused", "calm"},
    MacroMode.RETURN_HOME: {"calm"},
    MacroMode.RECOVER_RESET: {"neutral", "calm"},
}


def skill_spec_v4(name: str) -> Optional[SkillSpecV4]:
    return _SKILL_SPECS.get(str(name or "").strip().lower())


def default_skill_for_mode(mode: MacroMode) -> str:
    return _MODE_DEFAULT_SKILL[mode]


def allowed_skills_for_mode(mode: MacroMode) -> Set[str]:
    return {
        spec.name
        for spec in _SKILL_SPECS.values()
        if spec.mode == mode
    }


def allowed_primitives_for(mode: MacroMode, skill: str) -> Set[str]:
    skill_name = str(skill or "").strip().lower()
    spec = _SKILL_SPECS.get(skill_name)
    if spec is not None and spec.mode == mode:
        return set(spec.allowed_primitives)
    return _mode_allowed_primitives(mode)


def default_action_payload_for_mode(mode: MacroMode, *, reason: str, style: Optional[str] = None) -> Dict[str, Any]:
    base = dict(_MODE_DEFAULT_ACTIONS[mode])
    if style:
        base["style"] = str(style).strip().lower() or base.get("style", "calm")
    base["cancel_current"] = False
    base["explanation"] = reason
    return base


def _mode_allowed_primitives(mode: MacroMode) -> Set[str]:
    primitives: Set[str] = set()
    for spec in _SKILL_SPECS.values():
        if spec.mode == mode:
            primitives.update(spec.allowed_primitives)
    if primitives:
        return primitives
    # Idle mode can always hold/breath by convention.
    if mode == MacroMode.IDLE_PRESENCE:
        return {"hold", "breath", "glance"}
    return {"hold"}


def iter_skill_specs() -> Iterable[SkillSpecV4]:
    return _SKILL_SPECS.values()


def allowed_moods_for_mode(mode: MacroMode) -> Set[str]:
    return set(_MODE_ALLOWED_MOODS.get(mode, {"calm"}))
