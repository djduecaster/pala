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


def test_orchestrator_prunes_history_without_new_frames():
    planner = AsyncOrchestratorPlanner(
        frame_cache=LatestFrameCache(),
        provider="local",
        orchestrator_hz=1.0,
        video_window_s=1.0,
        video_max_frames=4,
    )
    try:
        old = np.zeros((4, 4, 3), dtype=np.uint8)
        planner._frame_history.append((time.monotonic() - 5.0, 1, old))
        planner._frame_history.append((time.monotonic() - 4.0, 2, old))
        planner._update_frame_history(time.monotonic())
        assert len(planner._frame_history) == 0
    finally:
        planner.shutdown()


def test_orchestrator_sampling_includes_newest_frame():
    planner = AsyncOrchestratorPlanner(
        frame_cache=LatestFrameCache(),
        provider="local",
        orchestrator_hz=1.0,
        video_window_s=10.0,
        video_max_frames=1,
    )
    try:
        now = time.monotonic()
        planner._frame_history.append((now - 2.0, 1, np.full((2, 2, 3), 1, dtype=np.uint8)))
        planner._frame_history.append((now - 1.0, 2, np.full((2, 2, 3), 9, dtype=np.uint8)))
        sampled = planner._sample_frame_history()
        assert len(sampled) == 1
        assert int(sampled[0][0, 0, 0]) == 9
    finally:
        planner.shutdown()


def test_local_fallback_recent_absence_prefers_reacquire():
    summary = SceneSummary(
        timestamp_monotonic_s=time.monotonic(),
        person_present=False,
        zone_hint=None,
        primary_person_conf=None,
        activity_hint="away",
    )
    decision = _local_decision(summary, recent_absence=True, last_seen_zone="left")
    assert decision.intent == "reacquire_attention"
    assert decision.primitive_hint == "orient_to_zone"
    assert decision.target_zone == "left"
