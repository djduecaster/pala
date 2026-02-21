from __future__ import annotations

import time
import logging
from typing import Callable, Optional

from ..types import PerceptionState, ActionPlan
from ..planner import PlannerInterface
from ..control.primitives import (
    PrimitiveKind,
    HoldCommand,
    GlanceCommand,
    NodCommand,
)
from .action_governor import ActionGovernor, ActionGovernorConfig
from .models import SceneObservation
from .scene_interpreter import SceneInterpreter

logger = logging.getLogger(__name__)

class BehaviorPolicy:
    """Behavior orchestrator: planner semantics + local scene/event governance."""

    def __init__(
        self,
        planner: PlannerInterface,
        dwell_s: float = 2.0,
        cooldown_s: float = 1.0,
        *,
        max_hold_s: float = 2.5,
        max_breath_s: float = 8.0,
        zone_change_nudge_cooldown_s: float = 0.8,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.planner = planner
        self.dwell_s = float(dwell_s)
        self.cooldown_s = float(cooldown_s)
        self._clock = clock or time.monotonic
        self._curr_zone: Optional[str] = None
        self._zone_start = self._clock()
        self._last_trigger = 0.0
        self._active_signature: Optional[str] = None
        self._planner_owns_semantics = bool(getattr(planner, "owns_semantic_behavior", False))
        self._scene = SceneInterpreter(clock=self._clock)
        self._governor = ActionGovernor(
            ActionGovernorConfig(
                max_hold_s=max_hold_s,
                max_breath_s=max_breath_s,
                zone_change_nudge_cooldown_s=zone_change_nudge_cooldown_s,
                force_interrupt_on_primitive_change=True,
            ),
            clock=self._clock,
        )

    def step(self, st: Optional[PerceptionState]) -> ActionPlan:
        obs = self._scene.observe(st)
        proposed = self._propose(st, obs)
        return self._arbitrate(proposed, obs)

    def _propose(self, st: Optional[PerceptionState], obs: SceneObservation) -> ActionPlan:
        now = self._clock()
        if st is None:
            return _hold_action(conf=0.3, reason="no_perception")

        if self._planner_owns_semantics:
            return self._planner_or_hold(st, reason="planner_owned")

        if not obs.person_present:
            return _hold_action(conf=0.3, reason="no_person")

        zone = obs.zone or "center"
        if zone != self._curr_zone:
            self._curr_zone = zone
            self._zone_start = now

        dwell = now - self._zone_start
        if dwell >= self.dwell_s and (now - self._last_trigger) >= self.cooldown_s:
            self._last_trigger = now
            if zone == "left":
                return ActionPlan(
                    primitive=PrimitiveKind.GLANCE,
                    command=GlanceCommand(direction="left", duration_s=0.6),
                    confidence=0.8,
                    explanation="user_dwell_left",
                    style="curious",
                    cancel_current=True,
                )
            if zone == "right":
                return ActionPlan(
                    primitive=PrimitiveKind.GLANCE,
                    command=GlanceCommand(direction="right", duration_s=0.6),
                    confidence=0.8,
                    explanation="user_dwell_right",
                    style="curious",
                    cancel_current=True,
                )
            return ActionPlan(
                primitive=PrimitiveKind.NOD,
                command=NodCommand(duration_s=0.4, amp_rad=0.2),
                confidence=0.7,
                explanation="user_dwell_center",
                style="focused",
                cancel_current=True,
            )
        return self._planner_or_hold(st, reason="planner_default")

    def _planner_or_hold(self, st: PerceptionState, *, reason: str) -> ActionPlan:
        try:
            proposed = self.planner.plan(st)
        except Exception as exc:  # noqa: BLE001 - planner errors should not break runtime
            logger.warning("behavior planner error (%s): %s", reason, exc)
            return _hold_action(conf=0.25, reason=f"planner_error:{reason}")
        if not isinstance(proposed, ActionPlan):
            return _hold_action(conf=0.25, reason=f"planner_invalid:{reason}")
        return proposed

    def _arbitrate(self, proposed: ActionPlan, obs: Optional[SceneObservation] = None) -> ActionPlan:
        if obs is None:
            obs = SceneObservation(
                ts_mono_s=self._clock(),
                person_present=False,
                person_conf=None,
                zone=None,
                zone_changed=False,
                zone_dwell_s=0.0,
                activity_hint=None,
                event="internal",
            )
        governed = self._governor.apply(proposed, obs)
        sig = self._signature(governed)
        if self._active_signature == sig:
            return ActionPlan(
                primitive=governed.primitive,
                command=governed.command,
                confidence=governed.confidence,
                explanation=governed.explanation,
                style=governed.style,
                action_id=governed.action_id,
                cancel_current=False,
            )

        self._active_signature = sig
        return ActionPlan(
            primitive=governed.primitive,
            command=governed.command,
            confidence=governed.confidence,
            explanation=governed.explanation,
            style=governed.style,
            action_id=governed.action_id,
            cancel_current=bool(governed.cancel_current),
        )

    @staticmethod
    def _signature(action: ActionPlan) -> str:
        return f"{action.primitive.value}:{action.style}:{action.command!r}"
def _hold_action(*, conf: float, reason: str) -> ActionPlan:
    return ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=conf,
        style="calm",
        explanation=reason,
    )
