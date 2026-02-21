from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from ..planner import PlannerInterface
from ..types import ActionPlan, HoldCommand, PerceptionState, PrimitiveKind
from .action_governor import ActionGovernor, ActionGovernorConfig
from .action_realizer import ActionRealizer, ActionRealizerConfig
from .intent_planner import IntentPlanner, IntentPlannerConfig
from .models import SceneObservation
from .scene_interpreter import SceneInterpreter
from .scene_memory import SceneMemory

logger = logging.getLogger(__name__)
_ACTION_ID_REUSE_PRIMITIVES = {PrimitiveKind.HOLD, PrimitiveKind.BREATH}


class BehaviorPolicy:
    """
    Behavior-layer orchestrator:
    perception -> scene events -> short-term memory -> semantic realization -> control-safe action.
    """

    def __init__(
        self,
        planner: PlannerInterface,
        dwell_s: float = 2.0,  # kept for backward compatibility
        cooldown_s: float = 1.0,  # kept for backward compatibility
        *,
        max_hold_s: float = 2.5,
        max_breath_s: float = 8.0,
        zone_change_nudge_cooldown_s: float = 0.8,
        director_enabled: bool = True,  # legacy alias for intent+realizer stack enable
        director_recently_seen_reacquire_s: float = 2.5,
        gaze_yaw_max_rad: float = 0.42,
        gaze_pitch_max_rad: float = 0.28,
        # legacy options retained to avoid breaking existing callsites/tests
        fusion_enable_ack: bool = True,
        fusion_enable_gaze: bool = True,
        fusion_enable_reacquire: bool = True,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.planner = planner
        self.dwell_s = float(dwell_s)
        self.cooldown_s = float(cooldown_s)
        self._clock = clock or time.monotonic
        self._active_signature: Optional[str] = None
        self._active_signature_action_id: Optional[str] = None
        self._planner_owns_semantics = bool(getattr(planner, "owns_semantic_behavior", False))
        # Remote planner ownership is the default semantics path. Local semantic
        # mode selection stays disabled unless planner does not own semantics.
        self._semantic_stack_enabled = bool(director_enabled) and (not self._planner_owns_semantics)

        self._scene = SceneInterpreter(clock=self._clock)
        self._scene_memory = SceneMemory(recently_seen_s=director_recently_seen_reacquire_s)
        intent_cfg = IntentPlannerConfig(mode_hold_s=1.2)
        realizer_cfg = ActionRealizerConfig(
            enable_acknowledge=bool(fusion_enable_ack),
            enable_reacquire=bool(fusion_enable_reacquire),
            gaze_yaw_max_rad=gaze_yaw_max_rad,
            gaze_pitch_max_rad=gaze_pitch_max_rad,
        )
        if self._planner_owns_semantics:
            # Keep these defaults available when local semantic stack is
            # explicitly re-enabled for experiments.
            intent_cfg.remote_first = True
            intent_cfg.remote_first_strict = True
            intent_cfg.remote_min_conf = 0.30
            intent_cfg.remote_act_now_conf = 0.40
            intent_cfg.mode_hold_s = 0.45
            realizer_cfg.remote_passthrough_conf = 0.40
            realizer_cfg.remote_repeat_holdoff_s = 0.9
            realizer_cfg.proactive_motion_s = 1.2
        self._intent_planner: Optional[IntentPlanner]
        self._realizer: Optional[ActionRealizer]
        if self._semantic_stack_enabled:
            self._intent_planner = IntentPlanner(
                intent_cfg,
                clock=self._clock,
            )
            self._realizer = ActionRealizer(
                realizer_cfg,
                clock=self._clock,
            )
        else:
            self._intent_planner = None
            self._realizer = None
        # Legacy feature toggles preserved for compatibility.
        if not director_enabled and self._intent_planner is not None:
            self._intent_planner._cfg.remote_bias_weight = 0.0  # noqa: SLF001
        if not fusion_enable_ack and self._realizer is not None:
            self._realizer._cfg.expressive_period_s = 1e9  # noqa: SLF001
        if not fusion_enable_gaze and self._realizer is not None:
            self._realizer._cfg.gaze_yaw_max_rad = 0.01  # noqa: SLF001
            self._realizer._cfg.gaze_pitch_max_rad = 0.01  # noqa: SLF001
        if not fusion_enable_reacquire and self._intent_planner is not None:
            self._intent_planner._cfg.remote_high_conf = 1.1  # noqa: SLF001

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
        memory = self._scene_memory.update(obs)
        proposed = self._planner_or_hold(st, reason=("planner_owned" if self._planner_owns_semantics else "planner_default"))
        if self._semantic_stack_enabled and self._intent_planner is not None and self._realizer is not None:
            intent = self._intent_planner.plan(proposed=proposed, obs=obs, memory=memory)
            realized = self._realizer.realize(intent=intent, proposed=proposed, st=st, obs=obs, memory=memory)
        else:
            realized = proposed
        return self._arbitrate(realized, obs)

    def _planner_or_hold(self, st: Optional[PerceptionState], *, reason: str) -> ActionPlan:
        if st is None:
            return _hold_action(conf=0.2, reason="no_perception")
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
        if self._active_signature == sig and governed.primitive in _ACTION_ID_REUSE_PRIMITIVES:
            action_id = self._active_signature_action_id or governed.action_id
            return ActionPlan(
                primitive=governed.primitive,
                command=governed.command,
                confidence=governed.confidence,
                explanation=governed.explanation,
                style=governed.style,
                action_id=action_id,
                cancel_current=False,
            )

        self._active_signature = sig
        self._active_signature_action_id = governed.action_id
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
