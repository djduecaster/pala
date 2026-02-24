from __future__ import annotations

import time

import numpy as np

from pala.behavior.policy import BehaviorPolicy, BehaviorPolicyConfig, _InFlightCall
from pala.behavior.model_clients.types import ModelResponse
from pala.behavior.world_state_store import WorldStateStore, WorldStateStoreConfig


class _DoneFuture:
    def __init__(self, value):
        self._value = value

    def done(self):
        return True

    def result(self):
        return self._value

    def cancel(self):
        return False


class _PendingFuture:
    def done(self):
        return False

    def result(self):
        raise RuntimeError("not done")

    def cancel(self):
        return False


def _make_policy(tmp_path, **overrides):
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    cfg = BehaviorPolicyConfig(
        remote_enabled=True,
        base_url="http://unit.test",
        request_timeout_ms=2000,
        env_hz=1.0,
        planner_hz=1.0,
        env_log_path=None,
        planner_log_path=None,
        reasoning_log_path=None,
        trace_log_path=None,
        **overrides,
    )
    return BehaviorPolicy(world_state=store, config=cfg)


def test_policy_payload_shapes_and_context(tmp_path):
    policy = _make_policy(tmp_path, request_min_fresh_frames=1)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    policy._frame_window.add_frame(frame, mono_ns=time.monotonic_ns())  # noqa: SLF001

    env_payload = policy._build_env_payload(st=None)  # noqa: SLF001
    planner_payload = policy._build_planner_payload(st=None, now=10.0)  # noqa: SLF001
    assert env_payload is not None
    assert planner_payload is not None

    env_request = env_payload["request"]
    planner_request = planner_payload["request"]
    assert env_request.response_format is not None
    assert planner_request.response_format is not None
    assert env_request.messages[0]["role"] == "system"
    assert planner_request.messages[1]["content"][-1]["type"] == "text"


def test_policy_latest_remote_candidates_age_and_limit(tmp_path):
    policy = _make_policy(tmp_path, proposer_max_age_s=0.5, planner_max_proposals=1)
    parsed = policy._intent_proposer.complete_request(  # noqa: SLF001
        '{"schema_version":"pala.intent_proposals.v2","proposals":['
        '{"intent":"track_user","primitive":"orient_to_zone","command":{"zone":"left"},"style":"focused","score":0.9,'
        '"confidence":0.8,"urgency":0.5,"risk":"low","allow_interrupt":true,"evidence":[],"rationale_short":"track"},'
        '{"intent":"scan_environment","primitive":"glance","command":{"direction":"right"},"style":"curious","score":0.7,'
        '"confidence":0.7,"urgency":0.4,"risk":"low","allow_interrupt":true,"evidence":[],"rationale_short":"scan"},'
        '{"intent":"idle_presence","primitive":"breath","command":{},"style":"calm","score":0.3,'
        '"confidence":0.5,"urgency":0.2,"risk":"low","allow_interrupt":true,"evidence":[],"rationale_short":"fallback"}]}'
    )
    assert parsed is not None
    policy._latest_remote_proposals = parsed.response  # noqa: SLF001
    policy._latest_remote_wall_s = time.time()  # noqa: SLF001

    candidates = policy._latest_remote_candidates()  # noqa: SLF001
    assert len(candidates) == 1
    assert candidates[0].source == "remote"

    policy._latest_remote_wall_s = time.time() - 5.0  # noqa: SLF001
    assert policy._latest_remote_candidates() == []  # noqa: SLF001


def test_policy_watchdog_timeout_converts_to_transport_error(tmp_path):
    policy = _make_policy(tmp_path)
    pending = _PendingFuture()
    policy._env_inflight = _InFlightCall(  # noqa: SLF001
        request_id=100,
        started_mono_s=1.0,
        payload_meta={},
        future=pending,
    )
    policy._drain_env_inflight(st=None, now=10.0)  # noqa: SLF001
    assert policy._env_inflight is None  # noqa: SLF001
    assert policy._health.env.transport_fail_streak >= 1  # noqa: SLF001


def test_policy_safe_future_result_guard(tmp_path):
    policy = _make_policy(tmp_path)
    call = _InFlightCall(
        request_id=1,
        started_mono_s=0.0,
        payload_meta={},
        future=_DoneFuture("bad"),
    )
    result = policy._safe_future_result(call=call, now=1.0)  # noqa: SLF001
    assert isinstance(result, ModelResponse)
    assert result.ok is False
