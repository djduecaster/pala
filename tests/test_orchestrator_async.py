import time

from pala.planner.orchestrator_async import (
    _decision_to_action,
    _local_decision,
    _parse_decision_content,
)
from pala.planner.state_models import SceneSummary, SessionMemory
from pala.control.primitives import PrimitiveKind, OrientToZoneCommand


def test_orchestrator_parse_decision_uses_target_zone():
    content = (
        '{"intent":"maintain_presence","style":"curious","primitive_hint":"orient_to_zone",'
        '"target_zone":"right","confidence":0.73,"rationale":"person present off-center"}'
    )
    decision = _parse_decision_content(content)
    assert decision is not None
    action = _decision_to_action(decision)
    assert action.primitive == PrimitiveKind.ORIENT_TO_ZONE
    assert isinstance(action.command, OrientToZoneCommand)
    assert action.command.zone == "right"
    assert action.style == "curious"


def test_orchestrator_local_center_maps_to_focused_nod():
    summary = SceneSummary(
        timestamp_monotonic_s=time.monotonic(),
        person_present=True,
        zone_hint="center",
        primary_person_conf=0.8,
        activity_hint="focused_work",
    )
    memory = SessionMemory(
        interaction_state="engaged",
        task_hypothesis="focus_work",
        last_transition_s=time.monotonic(),
        staleness_ms=0.0,
        recent_intents=[],
    )
    decision = _local_decision(summary, memory)
    assert decision.intent == "engaged_focus"
    assert decision.style == "focused"
    action = _decision_to_action(decision)
    assert action.primitive == PrimitiveKind.NOD
    assert action.style == "focused"
