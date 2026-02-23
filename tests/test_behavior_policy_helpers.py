from __future__ import annotations

import json
import time

import numpy as np

from pala.behavior.policy import BehaviorPolicy, BehaviorPolicyConfig, _InFlightCall
from pala.behavior.intent_proposer import parse_intent_proposer_response
from pala.behavior.remote_api import RemoteCallResult
from pala.behavior.world_state_store import WorldStateStore, WorldStateStoreConfig
from pala.types import ActionPlan, HoldCommand, PrimitiveKind


class _DoneFuture:
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result


class _PendingFuture:
    def __init__(self):
        self.cancel_called = False

    def done(self):
        return False

    def cancel(self):
        self.cancel_called = True
        return True


class _RaisingFuture:
    def done(self):
        return True

    def result(self):
        raise RuntimeError("boom")


def _make_policy(tmp_path, *, clock=None, **overrides):
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
    policy = BehaviorPolicy(world_state=store, config=cfg, clock=clock)
    policy._executor = None
    return policy


def _valid_intent_raw():
    return json.dumps(
        {
            "schema_version": "pala.intent_proposals.v1",
            "notes_short": "ok",
            "proposals": [
                {
                    "intent": "track_user",
                    "primitive": "orient_to_zone",
                    "command": {"zone": "right", "amp_rad": 0.2, "rate_rad_s": 1.2},
                    "style": "focused",
                    "score": 0.8,
                    "confidence": 0.75,
                    "urgency": 0.4,
                    "risk": "low",
                    "allow_interrupt": True,
                    "evidence": ["frame:latest"],
                    "rationale_short": "track right",
                }
            ],
        }
    )


def _valid_env_raw(*, delta_score: float = 0.8):
    return json.dumps(
        {
            "schema_version": "pala.env_summary.v1",
            "scene": "desk scene",
            "events": "user entered frame",
            "hypotheses": "user seeks interaction",
            "summary_short": "person moved into view",
            "delta_score": delta_score,
            "features": {
                "person_present": True,
                "zone_hint": "left",
                "activity_level": 0.6,
                "novelty": 0.5,
            },
        }
    )


def test_policy_helper_methods_and_backoff(tmp_path):
    policy = _make_policy(tmp_path)
    long = "x" * 400
    assert policy._preview_text(long, max_chars=50).endswith("...")  # noqa: SLF001
    assert policy._preview_text("short", max_chars=50) == "short"  # noqa: SLF001

    a = ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=0.3, style="calm")
    b = ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=0.9, style="calm")
    c = ActionPlan(primitive=PrimitiveKind.HOLD, command=HoldCommand(), confidence=0.9, style="focused")
    assert policy._same_action_signature(a, b) is True  # noqa: SLF001
    assert policy._same_action_signature(a, c) is False  # noqa: SLF001

    assert (
        policy._compute_failure_backoff_s(RemoteCallResult(False, 404, 10.0, None, "x"))  # noqa: SLF001
        >= policy._cfg.client_error_backoff_s
    )
    assert (
        policy._compute_failure_backoff_s(RemoteCallResult(False, 500, 10.0, None, "x"))  # noqa: SLF001
        == policy._cfg.error_backoff_s
    )


def test_policy_zone_hint_and_recent_switch_count(tmp_path):
    policy = _make_policy(tmp_path)
    snapshot = {"latest_env_snapshot": {"features": {"zone_hint": "center"}}}
    assert policy._zone_hint(st=None, snapshot=snapshot) == "center"  # noqa: SLF001
    assert policy._zone_hint(st=None, snapshot={}) == "center"  # noqa: SLF001

    class _State:
        debug = {"zone_hint": "left"}

    assert policy._zone_hint(st=_State(), snapshot={}) == "left"  # noqa: SLF001

    now = time.monotonic()
    policy._recent_commit_times.extend([now - 20.0, now - 1.0, now - 0.5])  # noqa: SLF001
    assert policy._recent_switch_count(now) == 2  # noqa: SLF001


def test_policy_build_payload_branches(tmp_path):
    policy = _make_policy(tmp_path, request_min_fresh_frames=2, planner_use_env_context=False)
    assert policy._build_env_payload(st=None) is None  # noqa: SLF001

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    policy._frame_window.add_frame(frame, mono_ns=time.monotonic_ns())  # noqa: SLF001
    policy._frame_window.add_frame(frame, mono_ns=time.monotonic_ns() + 1)  # noqa: SLF001
    env_payload = policy._build_env_payload(st=None)  # noqa: SLF001
    assert env_payload is not None
    assert env_payload["frames"] >= 2

    planner_payload = policy._build_planner_payload(st=None, now=10.0)  # noqa: SLF001
    assert planner_payload is not None
    user_text = planner_payload["body"]["messages"][1]["content"][-1]["text"]
    context_json = json.loads(user_text.split("context_json=", 1)[1])
    assert context_json["latest_env"]["scene"] == ""
    assert "event_tail" not in context_json


def test_policy_drain_env_and_planner_empty_and_ok_paths(tmp_path):
    policy = _make_policy(tmp_path)

    policy._env_inflight = _InFlightCall(
        request_id=1,
        started_mono_s=0.0,
        payload={},
        future=_DoneFuture(
            RemoteCallResult(
                ok=True,
                status_code=200,
                latency_ms=12.0,
                response_json={"choices": [{"message": {"content": []}}]},
                error=None,
            )
        ),
    )
    policy._drain_env_inflight(st=None, now=5.0)  # noqa: SLF001
    assert policy._health.env.state in {"DEGRADED", "OPEN_BREAKER", "HEALTHY"}  # noqa: SLF001

    policy._planner_inflight = _InFlightCall(
        request_id=2,
        started_mono_s=0.0,
        payload={},
        future=_DoneFuture(
            RemoteCallResult(
                ok=True,
                status_code=200,
                latency_ms=10.0,
                response_json={"choices": [{"message": {"content": _valid_intent_raw()}}]},
                error=None,
            )
        ),
    )
    policy._drain_planner_inflight(st=None, now=6.0)  # noqa: SLF001
    assert policy._latest_remote_proposals is not None  # noqa: SLF001
    assert len(policy._latest_remote_proposals.proposals) == 1  # noqa: SLF001


def test_policy_env_delta_sets_pending_planner_event(tmp_path):
    policy = _make_policy(tmp_path, planner_event_delta_threshold=0.65)
    policy._env_inflight = _InFlightCall(
        request_id=3,
        started_mono_s=0.0,
        payload={},
        future=_DoneFuture(
            RemoteCallResult(
                ok=True,
                status_code=200,
                latency_ms=8.0,
                response_json={"choices": [{"message": {"content": _valid_env_raw(delta_score=0.9)}}]},
                error=None,
            )
        ),
    )

    policy._drain_env_inflight(st=None, now=2.0)  # noqa: SLF001

    assert policy._pending_planner_event is True  # noqa: SLF001
    latest = policy.world_state.snapshot()["latest_env_snapshot"]
    assert latest["delta_score"] == 0.9


def test_policy_latest_remote_candidates_age_and_limit(tmp_path):
    policy = _make_policy(tmp_path, proposer_max_age_s=0.5, planner_max_proposals=1)
    response = policy._intent_proposer.complete_request(_valid_intent_raw())
    assert response is not None
    policy._latest_remote_proposals = response.response  # noqa: SLF001
    policy._latest_remote_wall_s = time.time()  # noqa: SLF001
    assert len(policy._latest_remote_candidates()) == 1  # noqa: SLF001

    policy._latest_remote_wall_s = time.time() - 2.0  # noqa: SLF001
    assert policy._latest_remote_candidates() == []  # noqa: SLF001


def test_policy_watchdog_times_out_stuck_env_call(tmp_path):
    policy = _make_policy(tmp_path)
    pending = _PendingFuture()
    policy._env_inflight = _InFlightCall(
        request_id=101,
        started_mono_s=1.0,
        payload={},
        future=pending,
    )

    policy._drain_env_inflight(st=None, now=4.0)  # noqa: SLF001

    assert policy._env_inflight is None  # noqa: SLF001
    assert pending.cancel_called is True
    assert policy._next_env_allowed_s >= 4.0 + policy._cfg.error_backoff_s  # noqa: SLF001
    assert policy._health.env.transport_fail_streak >= 1  # noqa: SLF001


def test_policy_future_exception_does_not_crash_planner_drain(tmp_path):
    policy = _make_policy(tmp_path)
    policy._planner_inflight = _InFlightCall(
        request_id=202,
        started_mono_s=2.0,
        payload={},
        future=_RaisingFuture(),
    )

    policy._drain_planner_inflight(st=None, now=4.0)  # noqa: SLF001

    assert policy._planner_inflight is None  # noqa: SLF001
    assert policy._next_planner_allowed_s >= 4.0 + policy._cfg.error_backoff_s  # noqa: SLF001
    assert policy._health.planner.transport_fail_streak >= 1  # noqa: SLF001


def test_policy_repeated_no_signal_remote_triggers_bounded_idle_commit(tmp_path):
    class _Clock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

        def advance(self, dt):
            self.now += dt

    clock = _Clock()
    policy = _make_policy(
        tmp_path,
        clock=clock,
        idle_after_s=6.0,
        idle_glance_after_s=10.0,
        arbiter_takeover_no_signal_streak=3,
        arbiter_takeover_no_commit_s=1.0,
        proposer_max_age_s=60.0,
    )

    no_signal_response = parse_intent_proposer_response(
        json.dumps(
            {
                "schema_version": "pala.intent_proposals.v1",
                "notes_short": "no signal",
                "proposals": [
                    {
                        "intent": "track_user",
                        "primitive": "glance",
                        "command": {
                            "direction": "left",
                            "amp_rad": 0.2,
                            "duration_s": 0.4,
                            "rate_rad_s": 1.0,
                        },
                        "style": "calm",
                        "score": 0.08,
                        "confidence": 0.2,
                        "urgency": 0.05,
                        "risk": "low",
                        "allow_interrupt": True,
                        "evidence": ["frame:latest"],
                        "rationale_short": "weak signal glance",
                    }
                ],
            }
        )
    )
    assert no_signal_response is not None

    committed_non_hold = False
    for _ in range(8):
        policy._latest_remote_proposals = no_signal_response.response  # noqa: SLF001
        policy._latest_remote_wall_s = time.time()  # noqa: SLF001
        policy._health.on_planner_result(  # noqa: SLF001
            status="ok",
            latency_ms=20.0,
            response=no_signal_response.response,
        )
        action = policy.step(st=None)
        if action.primitive.value != "hold":
            committed_non_hold = True
            break
        clock.advance(0.5)

    assert committed_non_hold is True
