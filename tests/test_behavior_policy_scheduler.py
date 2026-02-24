from __future__ import annotations

import time

import numpy as np

from pala.behavior.policy import BehaviorPolicy, BehaviorPolicyConfig
from pala.behavior.world_state_store import WorldStateStore, WorldStateStoreConfig


def _make_policy(tmp_path, **cfg_overrides) -> BehaviorPolicy:
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    cfg_kwargs = dict(
        remote_enabled=True,
        base_url="http://unit.test",
        api_key=None,
        env_hz=1.0,
        planner_hz=1.0,
        env_log_path=None,
        planner_log_path=None,
        reasoning_log_path=None,
        trace_log_path=None,
    )
    cfg_kwargs.update(cfg_overrides)
    cfg = BehaviorPolicyConfig(**cfg_kwargs)
    return BehaviorPolicy(world_state=store, config=cfg)


def test_build_payloads_include_media_first(tmp_path):
    policy = _make_policy(tmp_path)

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    policy._frame_window.add_frame(frame, mono_ns=time.monotonic_ns())

    env_payload = policy._build_env_payload(st=None)
    planner_payload = policy._build_planner_payload(st=None, now=0.0)

    assert env_payload is not None
    assert planner_payload is not None

    env_request = env_payload["request"]
    planner_request = planner_payload["request"]
    env_user_content = env_request.messages[1]["content"]
    planner_user_content = planner_request.messages[1]["content"]
    assert env_user_content[0]["type"] == "image_url"
    assert env_user_content[-1]["type"] == "text"
    assert planner_user_content[0]["type"] == "image_url"
    assert planner_user_content[-1]["type"] == "text"
    assert env_request.response_format["type"] == "json_schema"
    assert planner_request.response_format["type"] == "json_schema"
    assert env_request.top_p == 0.3
    assert planner_request.top_p == 0.3


def test_build_payloads_use_json_schema_for_gemini_provider(tmp_path):
    policy = _make_policy(tmp_path, remote_provider="gemini")
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    policy._frame_window.add_frame(frame, mono_ns=time.monotonic_ns())

    env_payload = policy._build_env_payload(st=None)
    planner_payload = policy._build_planner_payload(st=None, now=0.0)
    assert env_payload is not None
    assert planner_payload is not None
    assert env_payload["request"].response_format["type"] == "json_schema"
    assert planner_payload["request"].response_format["type"] == "json_schema"


def test_latest_only_pending_markers_when_inflight(tmp_path):
    policy = _make_policy(tmp_path)

    policy._env_summarizer.submit_or_replace({"req": "initial"})
    policy._intent_proposer.submit_or_replace({"req": "initial"})
    policy._env_inflight = object()
    policy._planner_inflight = None

    policy._maybe_schedule_env(st=None, now=10.0)
    policy._env_inflight = None
    policy._planner_inflight = object()
    policy._maybe_schedule_planner(st=None, now=10.0)

    env_pending = policy._env_summarizer.take_latest_pending()
    planner_pending = policy._intent_proposer.take_latest_pending()
    assert env_pending is not None
    assert planner_pending is not None


def test_cross_component_guard_does_not_mark_phantom_inflight(tmp_path):
    policy = _make_policy(tmp_path)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    policy._frame_window.add_frame(frame, mono_ns=time.monotonic_ns())

    # Env scheduling remains independent from planner in-flight.
    policy._planner_inflight = object()
    policy._maybe_schedule_env(st=None, now=10.0)
    assert policy._env_summarizer.in_flight is True

    # Planner remains schedulable even if env is in flight.
    policy._planner_inflight = None
    policy._env_inflight = object()
    policy._maybe_schedule_planner(st=None, now=10.0)
    assert policy._intent_proposer.in_flight is True
