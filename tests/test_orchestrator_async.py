import time
import json

import numpy as np

from pala.planner.orchestrator_async import (
    AsyncOrchestratorPlanner,
    _decision_to_action,
    _extract_reasoning,
    _extract_think_content,
    _parse_frame_fetch_request,
    _parse_decision_content,
)
from pala.perception.frame_cache import LatestFrameCache
from pala.types import ActionPlan, BBoxNorm, PerceptionState
from pala.control.primitives import HoldCommand, PrimitiveKind, OrientToZoneCommand


def test_orchestrator_parse_decision_uses_target_zone():
    content = (
        '{"target_state":"tracking","intent":"maintain_presence","style":"curious","primitive_hint":"orient_to_zone",'
        '"target_zone":"right","allow_interrupt":true,"urgency":"medium","confidence":0.73,'
        '"rationale":"person present off-center"}'
    )
    decision = _parse_decision_content(content)
    assert decision is not None
    action = _decision_to_action(decision)
    assert action.primitive == PrimitiveKind.ORIENT_TO_ZONE
    assert isinstance(action.command, OrientToZoneCommand)
    assert action.command.zone == "right"
    assert action.style == "curious"


def test_orchestrator_remote_payload_includes_multi_frame_sequence(monkeypatch):
    captured = {"payload": None}

    def _fake_post(url, payload, *, timeout_s, api_key):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"target_state":"tracking","intent":"maintain_presence","style":"curious","primitive_hint":"orient_to_zone",'
                            '"target_zone":"left","allow_interrupt":false,"urgency":"low","confidence":0.7,"rationale":"test"}'
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
        assert image_items == []
        text_items = [item for item in user_content if isinstance(item, dict) and item.get("type") == "text"]
        assert text_items
        context = text_items[0]["text"]
        assert "\"control_state\"" in context
        assert "\"summary_memory\"" in context
        assert "\"recent_decisions\"" in context
        assert "\"perception_state_raw\"" not in context
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


def test_compiler_preserves_remote_zone_choice():
    content = (
        '{"target_state":"tracking","intent":"track_transition","style":"calm","primitive_hint":"orient_to_zone",'
        '"target_zone":"center","allow_interrupt":true,"urgency":"medium","confidence":0.8,'
        '"rationale":"maintain center orientation"}'
    )
    decision = _parse_decision_content(content)
    assert decision is not None
    action = _decision_to_action(decision)
    assert action.primitive == PrimitiveKind.ORIENT_TO_ZONE
    assert isinstance(action.command, OrientToZoneCommand)
    assert action.command.zone == "center"


def test_remote_enabled_does_not_semantically_fallback_to_local(monkeypatch):
    def _fake_post(url, payload, *, timeout_s, api_key):
        return {"choices": [{"message": {"content": "not-json"}}]}

    monkeypatch.setattr("pala.planner.orchestrator_async._post_json", _fake_post)

    cache = LatestFrameCache()
    planner = AsyncOrchestratorPlanner(
        frame_cache=cache,
        provider="brev",
        base_url="http://127.0.0.1:8000",
        orchestrator_hz=50.0,
        max_frame_age_ms=1000,
    )
    try:
        for _ in range(5):
            cache.set(np.full((6, 6, 3), 7, dtype=np.uint8), mono_ns=time.monotonic_ns(), pts_ns=None)
            action = planner.plan(
                PerceptionState(
                    timestamp_monotonic_s=time.monotonic(),
                    primary_person=BBoxNorm(cx=0.1, cy=0.5, w=0.2, h=0.4),
                    primary_person_conf=0.8,
                    debug={"zone_hint": "left"},
                )
            )
            assert action.primitive == PrimitiveKind.HOLD
            time.sleep(0.02)
    finally:
        planner.shutdown()


def test_extract_reasoning_reads_reasoning_content():
    response = {
        "choices": [
            {
                "message": {
                    "content": "{\"target_state\":\"tracking\",\"intent\":\"track\"}",
                    "reasoning_content": "I track because person moved from left to right.",
                }
            }
        ]
    }
    reasoning = _extract_reasoning(response)
    assert reasoning is not None
    assert "person moved" in reasoning


def test_parse_decision_rejects_string_none_intent():
    content = (
        '{"target_state":"tracking","intent":"None","style":"focused","primitive_hint":"orient_to_zone",'
        '"target_zone":"center","allow_interrupt":false,"urgency":"low","confidence":0.7,'
        '"rationale":"reason"}'
    )
    assert _parse_decision_content(content) is None


def test_extract_think_content_parses_tagged_output():
    raw = "<think>\nstep one\nstep two\n</think>\n\n<answer>\nhello\n</answer>"
    think = _extract_think_content(raw)
    assert think is not None
    assert "step one" in think


def test_parse_decision_accepts_prediction_action_details_schema():
    content = (
        '{'
        '"inference":"person is transitioning and needs assistance",'
        '"prediction":{"target_state":"engaging","style":"focused","target_zone":"left","confidence":0.73,"allow_interrupt":false},'
        '"action_details":{"primitive":"reach_for_object","urgency":"medium"}'
        '}'
    )
    decision = _parse_decision_content(content)
    assert decision is None


def test_parse_decision_accepts_prediction_string_schema():
    content = (
        '{'
        '"prediction":"reach_for_object",'
        '"action_details":{"target_zone":"left","style":"focused","confidence":0.66,"allow_interrupt":true},'
        '"inference":"robot should help with desk task"'
        '}'
    )
    decision = _parse_decision_content(content)
    assert decision is None


def test_reasoning_probe_writes_reasoning_event(monkeypatch):
    def _fake_post(url, payload, *, timeout_s, api_key):
        user_parts = payload["messages"][1]["content"]
        user_text = next((item["text"] for item in user_parts if item.get("type") == "text"), "")
        if "Use the provided context and frame sequence to decide next action." in user_text:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"target_state":"tracking","intent":"track_transition","style":"curious",'
                                '"primitive_hint":"orient_to_zone","target_zone":"left","allow_interrupt":true,'
                                '"urgency":"medium","confidence":0.7,"rationale":"follow movement"}'
                            )
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": "<think>person shifts left to right</think><answer>track movement</answer>"
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
        orchestrator_hz=30.0,
        max_frame_age_ms=1000,
        reasoning_probe_enabled=True,
        reasoning_probe_hz=20.0,
    )
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            cache.set(np.full((6, 6, 3), 5, dtype=np.uint8), mono_ns=time.monotonic_ns(), pts_ns=None)
            planner.plan(
                PerceptionState(
                    timestamp_monotonic_s=time.monotonic(),
                    primary_person=BBoxNorm(cx=0.4, cy=0.5, w=0.2, h=0.4),
                    primary_person_conf=0.8,
                    debug={"zone_hint": "left"},
                )
            )
            if planner._latest_reasoning:
                break
            time.sleep(0.02)
        assert planner._latest_reasoning is not None
    finally:
        planner.shutdown()


def test_payload_includes_policy_version_and_blocks(monkeypatch):
    captured = {"payload": None}

    def _fake_post(url, payload, *, timeout_s, api_key):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"target_state":"tracking","intent":"track_transition","style":"curious",'
                            '"primitive_hint":"orient_to_zone","target_zone":"left","allow_interrupt":true,'
                            '"urgency":"medium","confidence":0.7,"rationale":"follow movement"}'
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
        policy_version="vtest",
        orchestrator_hz=50.0,
    )
    try:
        deadline = time.monotonic() + 1.0
        while captured["payload"] is None and time.monotonic() < deadline:
            cache.set(np.full((6, 6, 3), 4, dtype=np.uint8), mono_ns=time.monotonic_ns(), pts_ns=None)
            planner.plan(
                PerceptionState(
                    timestamp_monotonic_s=time.monotonic(),
                    primary_person=BBoxNorm(cx=0.4, cy=0.5, w=0.2, h=0.4),
                    primary_person_conf=0.8,
                    debug={"zone_hint": "left"},
                )
            )
            time.sleep(0.02)
        assert captured["payload"] is not None
        system = captured["payload"]["messages"][0]["content"]
        assert system == "You are a helpful assistant."
        user_items = captured["payload"]["messages"][1]["content"]
        assert user_items[0]["type"] == "text"
        user_text = next(item["text"] for item in user_items if item.get("type") == "text")
        assert "[policy_version=vtest]" in user_text
        assert '"control_state"' in user_text
        assert '"summary_memory"' in user_text
    finally:
        planner.shutdown()


def test_transcript_window_applies_role_and_char_caps():
    planner = AsyncOrchestratorPlanner(
        frame_cache=LatestFrameCache(),
        provider="local",
        context_transcript_max_items=4,
        context_transcript_per_type_max_items=2,
        context_transcript_max_chars=120,
    )
    try:
        planner._append_transcript("observation", "obs1 " + ("x" * 40))
        planner._append_transcript("observation", "obs2 " + ("x" * 40))
        planner._append_transcript("observation", "obs3 " + ("x" * 40))
        planner._append_transcript("reasoning", "reason1 " + ("y" * 40))
        planner._append_transcript("decision", "dec1 " + ("z" * 40))
        window = planner._build_transcript_window()
        assert len(window) <= 4
        assert sum(len(line) + 1 for line in window) <= 120
        assert all(("decision:" in line) or ("reasoning:" in line) for line in window)
    finally:
        planner.shutdown()


def test_parse_decision_accepts_markdown_fenced_json():
    content = (
        "```json\n"
        '{"target_state":"tracking","intent":"track_transition","style":"curious","primitive_hint":"orient_to_zone",'
        '"target_zone":"right","allow_interrupt":true,"urgency":"medium","confidence":0.7,'
        '"rationale":"follow movement"}\n'
        "```"
    )
    decision = _parse_decision_content(content)
    assert decision is not None
    assert decision.intent == "track_transition"
    assert decision.target_zone == "right"


def test_parse_decision_bool_string_false_is_false():
    content = (
        '{"target_state":"tracking","intent":"track_transition","style":"curious","primitive_hint":"orient_to_zone",'
        '"target_zone":"right","allow_interrupt":"false","urgency":"medium","confidence":0.7,'
        '"rationale":"follow movement"}'
    )
    decision = _parse_decision_content(content)
    assert decision is not None
    assert decision.allow_interrupt is False


def test_parse_decision_uses_selected_action_schema():
    content = (
        '{'
        '"rationale":"candidate scoring prefers orient",'
        '"selected_action":{"primitive":"orient_to_zone","target_zone":"right","style":"curious","intent":"track_transition","target_state":"tracking","allow_interrupt":"false","urgency":"medium","confidence":0.82}'
        '}'
    )
    decision = _parse_decision_content(content)
    assert decision is None


def test_parse_decision_act_now_false_forces_hold():
    content = (
        '{"target_state":"tracking","intent":"track_transition","style":"curious","primitive_hint":"orient_to_zone",'
        '"target_zone":"right","allow_interrupt":true,"urgency":"medium","confidence":0.7,'
        '"rationale":"pause actuation",'
        '"act_now":"false"}'
    )
    decision = _parse_decision_content(content)
    assert decision is not None
    assert decision.primitive_hint == "hold"
    assert decision.allow_interrupt is False


def test_parse_frame_fetch_request():
    content = '{"tool_call":{"name":"frame_fetch","reason":"need current visual detail"}}'
    req = _parse_frame_fetch_request(content)
    assert req is not None
    assert "visual detail" in req.reason


def test_orchestrator_frame_fetch_roundtrip(monkeypatch):
    calls = {"n": 0, "payloads": []}

    def _fake_post(url, payload, *, timeout_s, api_key):
        calls["n"] += 1
        calls["payloads"].append(payload)
        if calls["n"] == 1:
            return {"choices": [{"message": {"content": '{"tool_call":{"name":"frame_fetch","reason":"uncertain"}}'}}]}
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"target_state":"tracking","intent":"track_transition","style":"curious",'
                            '"primitive_hint":"orient_to_zone","target_zone":"left","allow_interrupt":true,'
                            '"urgency":"medium","confidence":0.7,"rationale":"follow movement"}'
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
        orchestrator_hz=40.0,
        planner_allow_frame_fetch=True,
        planner_max_tool_calls_per_cycle=1,
    )
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            cache.set(np.full((6, 6, 3), 3, dtype=np.uint8), mono_ns=time.monotonic_ns(), pts_ns=None)
            action = planner.plan(
                PerceptionState(
                    timestamp_monotonic_s=time.monotonic(),
                    primary_person=BBoxNorm(cx=0.4, cy=0.5, w=0.2, h=0.4),
                    primary_person_conf=0.8,
                    debug={"zone_hint": "left"},
                )
            )
            if calls["n"] >= 2 and action.primitive == PrimitiveKind.ORIENT_TO_ZONE:
                break
            time.sleep(0.02)
        assert calls["n"] >= 2
        first_user_content = calls["payloads"][0]["messages"][1]["content"]
        second_user_content = calls["payloads"][1]["messages"][1]["content"]
        assert not any(item.get("type") == "image_url" for item in first_user_content if isinstance(item, dict))
        assert any(item.get("type") == "image_url" for item in second_user_content if isinstance(item, dict))
    finally:
        planner.shutdown()


def test_inflight_guard_prevents_request_pileup(monkeypatch):
    call_count = {"n": 0}

    def _fake_post(url, payload, *, timeout_s, api_key):
        call_count["n"] += 1
        time.sleep(0.12)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"target_state":"tracking","intent":"track_transition","style":"curious",'
                            '"primitive_hint":"orient_to_zone","target_zone":"left","allow_interrupt":true,'
                            '"urgency":"medium","confidence":0.7,"rationale":"follow movement"}'
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
        orchestrator_hz=200.0,
        inflight_guard_enabled=True,
    )
    try:
        start = time.monotonic()
        while time.monotonic() - start < 0.35:
            cache.set(np.full((6, 6, 3), 9, dtype=np.uint8), mono_ns=time.monotonic_ns(), pts_ns=None)
            planner.plan(
                PerceptionState(
                    timestamp_monotonic_s=time.monotonic(),
                    primary_person=BBoxNorm(cx=0.4, cy=0.5, w=0.2, h=0.4),
                    primary_person_conf=0.8,
                    debug={"zone_hint": "left"},
                )
            )
            time.sleep(0.005)
        # We spammed plan calls; guarded submission should still keep remote calls low.
        assert call_count["n"] <= 4
    finally:
        planner.shutdown()


def test_timeline_writes_request_lifecycle(monkeypatch, tmp_path):
    timeline_path = tmp_path / "timeline.jsonl"

    def _fake_post(url, payload, *, timeout_s, api_key):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"target_state":"tracking","intent":"track_transition","style":"curious",'
                            '"primitive_hint":"orient_to_zone","target_zone":"left","allow_interrupt":true,'
                            '"urgency":"medium","confidence":0.7,"rationale":"follow movement"}'
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
        orchestrator_hz=50.0,
        timeline_jsonl_path=str(timeline_path),
    )
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            cache.set(np.full((6, 6, 3), 6, dtype=np.uint8), mono_ns=time.monotonic_ns(), pts_ns=None)
            planner.plan(
                PerceptionState(
                    timestamp_monotonic_s=time.monotonic(),
                    primary_person=BBoxNorm(cx=0.4, cy=0.5, w=0.2, h=0.4),
                    primary_person_conf=0.8,
                    debug={"zone_hint": "left"},
                )
            )
            if timeline_path.exists():
                rows = [json.loads(line) for line in timeline_path.read_text().splitlines() if line.strip()]
                if rows and any(r.get("type") == "decision_event" for r in rows):
                    break
            time.sleep(0.02)
        rows = [json.loads(line) for line in timeline_path.read_text().splitlines() if line.strip()]
        kinds = {row.get("type") for row in rows}
        assert "run_start" in kinds
        assert "request_start" in kinds
        assert "request_end" in kinds
        assert "decision_event" in kinds
    finally:
        planner.shutdown()


def test_remote_stale_action_falls_back_to_neutral_hold():
    planner = AsyncOrchestratorPlanner(
        frame_cache=LatestFrameCache(),
        provider="brev",
        base_url="http://127.0.0.1:8000",
        response_ttl_ms=100,
    )
    try:
        planner._latest_action = ActionPlan(
            primitive=PrimitiveKind.ORIENT_TO_ZONE,
            command=OrientToZoneCommand(zone="left"),
            confidence=0.9,
            style="focused",
            cancel_current=False,
        )
        planner._latest_action_ts_s = time.monotonic() - 1.0
        action = planner.plan(
            PerceptionState(
                timestamp_monotonic_s=time.monotonic(),
                primary_person=BBoxNorm(cx=0.2, cy=0.5, w=0.2, h=0.4),
                primary_person_conf=0.8,
                debug={"zone_hint": "left"},
            )
        )
        assert action.primitive == PrimitiveKind.HOLD
        assert isinstance(action.command, HoldCommand)
        assert action.cancel_current is True
    finally:
        planner.shutdown()
