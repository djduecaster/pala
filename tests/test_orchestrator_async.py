import time

import numpy as np

from pala.planner.orchestrator_async import (
    AsyncOrchestratorPlanner,
    _decision_to_action,
    _local_decision,
    _parse_decision_content,
)
from pala.perception.frame_cache import LatestFrameCache
from pala.types import BBoxNorm, PerceptionState
from pala.planner.state_models import SceneSummary
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
    decision = _local_decision(summary)
    assert decision.intent == "engaged_focus"
    assert decision.style == "focused"
    action = _decision_to_action(decision)
    assert action.primitive == PrimitiveKind.NOD
    assert action.style == "focused"


def test_orchestrator_remote_payload_includes_multi_frame_sequence(monkeypatch):
    captured = {"payload": None}

    def _fake_post(url, payload, *, timeout_s, api_key):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"intent":"maintain_presence","style":"curious","primitive_hint":"orient_to_zone",'
                            '"target_zone":"left","confidence":0.7,"rationale":"test"}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr("pala.planner.orchestrator_async._post_json", _fake_post)

    cache = LatestFrameCache()
    planner = AsyncOrchestratorPlanner(
        frame_cache=cache,
        provider="brev",
        base_url="http://127.0.0.1:8000",
        orchestrator_hz=100.0,
        max_frame_age_ms=1000,
        video_window_s=5.0,
        video_max_frames=3,
    )

    try:
        for i in range(6):
            cache.set(np.full((6, 6, 3), i, dtype=np.uint8), mono_ns=time.monotonic_ns(), pts_ns=None)
            st = PerceptionState(
                timestamp_monotonic_s=time.monotonic(),
                primary_person=BBoxNorm(cx=0.2, cy=0.5, w=0.2, h=0.4),
                primary_person_conf=0.8,
                debug={"zone_hint": "left"},
            )
            planner.plan(st)
            if captured["payload"] is not None:
                break
            time.sleep(0.03)

        deadline = time.monotonic() + 1.0
        while captured["payload"] is None and time.monotonic() < deadline:
            planner.plan(
                PerceptionState(
                    timestamp_monotonic_s=time.monotonic(),
                    primary_person=BBoxNorm(cx=0.2, cy=0.5, w=0.2, h=0.4),
                    primary_person_conf=0.8,
                    debug={"zone_hint": "left"},
                )
            )
            time.sleep(0.02)

        assert captured["payload"] is not None
        user_content = captured["payload"]["messages"][1]["content"]
        assert isinstance(user_content, list)
        image_items = [item for item in user_content if isinstance(item, dict) and item.get("type") == "image_url"]
        assert image_items
        assert len(image_items) <= 3
    finally:
        planner.shutdown()
