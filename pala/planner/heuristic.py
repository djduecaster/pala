from __future__ import annotations

from ..types import ActionPlan, BreathCommand, HoldCommand, OrientToZoneCommand, PerceptionState, PrimitiveKind


class HeuristicPlanner:
    """Minimal local planner fallback."""

    owns_semantic_behavior = False

    def plan(self, st: PerceptionState | None) -> ActionPlan:
        if st is None:
            return ActionPlan(
                primitive=PrimitiveKind.HOLD,
                command=HoldCommand(),
                confidence=0.1,
                explanation="no perception state",
                style="calm",
                cancel_current=False,
            )

        zone = "center"
        if isinstance(st.debug, dict):
            raw_zone = st.debug.get("zone_hint")
            if isinstance(raw_zone, str) and raw_zone.strip().lower() in {"left", "center", "right"}:
                zone = raw_zone.strip().lower()

        if st.primary_person_conf is not None and st.primary_person_conf >= 0.35:
            return ActionPlan(
                primitive=PrimitiveKind.ORIENT_TO_ZONE,
                command=OrientToZoneCommand(zone=zone, amp_rad=0.25, rate_rad_s=1.2),
                confidence=0.5,
                explanation=f"local heuristic tracking zone={zone}",
                style="calm",
                cancel_current=False,
            )

        return ActionPlan(
            primitive=PrimitiveKind.BREATH,
            command=BreathCommand(amp_rad=0.06, period_s=6.5, rate_rad_s=1.0),
            confidence=0.2,
            explanation="local heuristic idle breath",
            style="calm",
            cancel_current=False,
        )

