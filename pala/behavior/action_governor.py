from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional

from ..types import (
    ActionPlan,
    PrimitiveKind,
    HoldCommand,
    OrientToZoneCommand,
)
from .models import SceneObservation

_PERSISTENT_PRIMITIVES = {PrimitiveKind.HOLD, PrimitiveKind.BREATH}


@dataclass
class ActionGovernorConfig:
    max_hold_s: float = 2.5
    max_breath_s: float = 8.0
    zone_change_nudge_cooldown_s: float = 0.8
    force_interrupt_on_primitive_change: bool = True


class ActionGovernor:
    """Execution guardrails to keep behavior lively and interruptible."""

    def __init__(
        self,
        cfg: Optional[ActionGovernorConfig] = None,
        *,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._cfg = cfg or ActionGovernorConfig()
        self._clock = clock or time.monotonic
        self._last_action: Optional[ActionPlan] = None
        self._last_action_ts_s: Optional[float] = None
        self._last_signature: Optional[str] = None
        self._last_zone_nudge_ts_s = 0.0

    def apply(self, proposed: ActionPlan, obs: SceneObservation) -> ActionPlan:
        now = self._clock()
        out = proposed
        prev = self._last_action
        prev_ts = self._last_action_ts_s

        if prev is not None and prev_ts is not None:
            prev_age_s = max(0.0, now - prev_ts)
            out = self._interrupt_on_primitive_change(prev, out)
            out = self._refresh_after_persistent(prev, prev_age_s, out, obs)

        out = self._nudge_zone_tracking(out, obs, now)
        out = self._suppress_duplicate_cancel(out)

        self._last_action = out
        self._last_action_ts_s = now
        self._last_signature = _signature(out)
        return out

    def _interrupt_on_primitive_change(self, prev: ActionPlan, proposed: ActionPlan) -> ActionPlan:
        if not self._cfg.force_interrupt_on_primitive_change:
            return proposed
        if prev.primitive not in _PERSISTENT_PRIMITIVES:
            return proposed
        if proposed.primitive == prev.primitive:
            return proposed
        if proposed.cancel_current:
            return proposed
        return _copy_action(
            proposed,
            cancel_current=True,
            explanation_prefix="governor_preempt_persistent",
        )

    def _refresh_after_persistent(
        self,
        prev: ActionPlan,
        prev_age_s: float,
        proposed: ActionPlan,
        obs: SceneObservation,
    ) -> ActionPlan:
        if prev.primitive not in _PERSISTENT_PRIMITIVES:
            return proposed
        if proposed.primitive != prev.primitive:
            return proposed
        if not obs.person_present:
            return proposed

        max_age_s = self._cfg.max_hold_s if prev.primitive == PrimitiveKind.HOLD else self._cfg.max_breath_s
        if prev_age_s < max_age_s:
            return proposed
        zone = obs.zone or "center"
        return ActionPlan(
            primitive=PrimitiveKind.ORIENT_TO_ZONE,
            command=OrientToZoneCommand(zone=zone, amp_rad=0.22, rate_rad_s=1.5),
            confidence=max(0.55, proposed.confidence),
            explanation=f"governor_refresh_after_persistent:{zone}",
            style="curious",
            cancel_current=True,
        )

    def _nudge_zone_tracking(self, proposed: ActionPlan, obs: SceneObservation, now: float) -> ActionPlan:
        if not obs.person_present or not obs.zone_changed or obs.zone is None:
            return proposed
        if _targets_zone(proposed, obs.zone):
            return proposed
        if (now - self._last_zone_nudge_ts_s) < self._cfg.zone_change_nudge_cooldown_s:
            return proposed
        self._last_zone_nudge_ts_s = now
        return ActionPlan(
            primitive=PrimitiveKind.ORIENT_TO_ZONE,
            command=OrientToZoneCommand(zone=obs.zone, amp_rad=0.22, rate_rad_s=1.5),
            confidence=max(0.6, proposed.confidence),
            explanation=f"governor_zone_change_nudge:{obs.zone}",
            style="curious",
            cancel_current=True,
        )

    def _suppress_duplicate_cancel(self, proposed: ActionPlan) -> ActionPlan:
        if not proposed.cancel_current:
            return proposed
        if self._last_signature is None:
            return proposed
        if _signature(proposed) != self._last_signature:
            return proposed
        return _copy_action(proposed, cancel_current=False)


def _targets_zone(action: ActionPlan, zone: str) -> bool:
    if action.primitive != PrimitiveKind.ORIENT_TO_ZONE:
        return False
    command = action.command
    if not isinstance(command, OrientToZoneCommand):
        return False
    return command.zone == zone


def _copy_action(
    action: ActionPlan,
    *,
    cancel_current: Optional[bool] = None,
    explanation_prefix: Optional[str] = None,
) -> ActionPlan:
    explanation = action.explanation
    if explanation_prefix:
        if explanation:
            explanation = f"{explanation_prefix};{explanation}"
        else:
            explanation = explanation_prefix
    return ActionPlan(
        primitive=action.primitive,
        command=action.command,
        confidence=action.confidence,
        explanation=explanation,
        style=action.style,
        action_id=action.action_id,
        cancel_current=action.cancel_current if cancel_current is None else cancel_current,
    )


def _signature(action: ActionPlan) -> str:
    return f"{action.primitive.value}:{action.style}:{action.command!r}"

