from __future__ import annotations

import json

from pala.control.primitives import PrimitiveKind
from pala.planner import (
    AsyncOrchestratorPlanner,
    AsyncSceneSummarizer,
    HeuristicPlanner,
    TimelineConfig,
    TimelineWriter,
)
from pala.types import ActionPlan, BBoxNorm, BreathCommand, HoldCommand, OrientToZoneCommand, PerceptionState


def _state(conf: float | None, zone_hint: object = None) -> PerceptionState:
    debug = {}
    if zone_hint is not None:
        debug["zone_hint"] = zone_hint
    return PerceptionState(
        timestamp_monotonic_s=1.0,
        primary_person=BBoxNorm(cx=0.5, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=conf,
        debug=debug,
    )


def test_heuristic_planner_holds_without_perception():
    planner = HeuristicPlanner()
    out = planner.plan(None)
    assert out.primitive == PrimitiveKind.HOLD
    assert isinstance(out.command, HoldCommand)
    assert out.cancel_current is False


def test_heuristic_planner_breathes_for_low_confidence():
    planner = HeuristicPlanner()
    out = planner.plan(_state(0.2, zone_hint="right"))
    assert out.primitive == PrimitiveKind.BREATH
    assert isinstance(out.command, BreathCommand)
    assert out.cancel_current is False


def test_heuristic_planner_uses_valid_zone_hint():
    planner = HeuristicPlanner()
    out = planner.plan(_state(0.9, zone_hint="RIGHT"))
    assert out.primitive == PrimitiveKind.ORIENT_TO_ZONE
    assert isinstance(out.command, OrientToZoneCommand)
    assert out.command.zone == "right"


def test_heuristic_planner_defaults_to_center_for_invalid_zone_hint():
    planner = HeuristicPlanner()
    out = planner.plan(_state(0.9, zone_hint="up"))
    assert out.primitive == PrimitiveKind.ORIENT_TO_ZONE
    assert isinstance(out.command, OrientToZoneCommand)
    assert out.command.zone == "center"


def test_async_orchestrator_planner_passthrough_and_shutdown():
    class _Fallback:
        def __init__(self) -> None:
            self.calls = 0

        def plan(self, _st):
            self.calls += 1
            return ActionPlan(
                primitive=PrimitiveKind.HOLD,
                command=HoldCommand(),
                confidence=0.2,
                style="calm",
            )

    fallback = _Fallback()
    planner = AsyncOrchestratorPlanner(fallback=fallback, ignored_kwarg=True)
    out = planner.plan(_state(0.8, zone_hint="left"))
    assert out.primitive == PrimitiveKind.HOLD
    assert fallback.calls == 1
    assert planner.shutdown() is None


def test_scene_summarizer_stub_contract():
    summarizer = AsyncSceneSummarizer(foo="bar")
    assert summarizer.enabled is True
    assert summarizer.latest() is None
    assert summarizer.shutdown() is None


def test_timeline_writer_disabled_does_not_create_file(tmp_path):
    path = tmp_path / "timeline.jsonl"
    writer = TimelineWriter(TimelineConfig(enabled=False, jsonl_path=str(path)))
    writer.append("decision_event", {"primitive": "hold"})
    assert path.exists() is False


def test_timeline_writer_enabled_appends_jsonl_events(tmp_path):
    path = tmp_path / "timeline.jsonl"
    writer = TimelineWriter(TimelineConfig(enabled=True, jsonl_path=str(path)))

    payload = {"primitive": "hold"}
    writer.append("decision_event", payload)
    payload["primitive"] = "breath"

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "decision_event"
    assert event["payload"] == {"primitive": "hold"}
    assert isinstance(event["ts_wall_s"], float)
