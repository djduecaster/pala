from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from pala.behavior.arbiter import ArbiterResult
from pala.behavior.model_clients.response_utils import _coerce_text, extract_message_content
from pala.behavior.model_clients.types import ModelResponse
from pala.behavior.policy import (
    BehaviorPolicy,
    BehaviorPolicyConfig,
    _InFlightCall,
    _encode_frame_data_url,
    _response_meta,
)
from pala.behavior.types import GovernedCandidate, IntentProposal, ProposalCandidate
from pala.behavior.world_state_store import WorldStateStore, WorldStateStoreConfig
from pala.types import PrimitiveKind


class _DoneFuture:
    def __init__(self, value):
        self._value = value
        self.cancel_called = False

    def done(self):
        return True

    def result(self):
        return self._value

    def cancel(self):
        self.cancel_called = True
        return True


class _PendingFuture:
    def __init__(self):
        self.cancel_called = False

    def done(self):
        return False

    def result(self):
        raise RuntimeError("not done")

    def cancel(self):
        self.cancel_called = True
        return True


class _MemoryLog:
    def __init__(self, *, fail_write: bool = False, fail_close: bool = False):
        self.items = []
        self.closed = 0
        self._fail_write = fail_write
        self._fail_close = fail_close

    def write(self, payload):
        if self._fail_write:
            raise RuntimeError("write failed")
        self.items.append(payload)

    def close(self):
        self.closed += 1
        if self._fail_close:
            raise RuntimeError("close failed")


class _FakeModelClient:
    def chat(self, request):
        return ModelResponse(ok=True, status_code=200, latency_ms=1.0, response_json={"ok": True}, error=None)


class _FakeExecutor:
    def __init__(self, future_factory):
        self._future_factory = future_factory
        self.calls = []
        self.shutdown_calls = []

    def submit(self, fn, request):
        self.calls.append((fn, request))
        return self._future_factory()

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_calls.append((wait, cancel_futures))


def _make_policy(tmp_path, **cfg_overrides):
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    cfg_kwargs = dict(
        remote_enabled=False,
        base_url="http://unit.test",
        request_timeout_ms=2000,
        env_hz=1.0,
        planner_hz=1.0,
        env_log_path=None,
        planner_log_path=None,
        reasoning_log_path=None,
        trace_log_path=None,
    )
    cfg_kwargs.update(cfg_overrides)
    return BehaviorPolicy(world_state=store, config=BehaviorPolicyConfig(**cfg_kwargs), clock=lambda: 10.0)


def _valid_env_content(delta_score: float = 0.9) -> str:
    return json.dumps(
        {
            "schema_version": "pala.env_summary.v1",
            "scene": "desk",
            "events": "user moved left",
            "hypotheses": "user engaging",
            "summary_short": "user moved left",
            "delta_score": delta_score,
            "features": {
                "person_present": True,
                "zone_hint": "left",
                "activity_level": 0.7,
                "novelty": 0.8,
            },
        }
    )


def _valid_planner_content() -> str:
    return json.dumps(
        {
            "schema_version": "pala.intent_proposals.v2",
            "proposals": [
                {
                    "intent": "track_user",
                    "primitive": "orient_to_zone",
                    "command": {"zone": "left", "amp_rad": 0.2, "rate_rad_s": 1.3},
                    "style": "focused",
                    "score": 0.9,
                    "confidence": 0.8,
                    "urgency": 0.5,
                    "risk": "low",
                    "allow_interrupt": True,
                    "evidence": ["frame:latest"],
                    "rationale_short": "track user",
                },
                {
                    "intent": "scan_environment",
                    "primitive": "glance",
                    "command": {"direction": "right", "amp_rad": 0.2, "duration_s": 0.5, "rate_rad_s": 1.4},
                    "style": "curious",
                    "score": 0.7,
                    "confidence": 0.7,
                    "urgency": 0.4,
                    "risk": "low",
                    "allow_interrupt": True,
                    "evidence": ["frame:latest"],
                    "rationale_short": "scan right",
                },
                {
                    "intent": "idle_presence",
                    "primitive": "breath",
                    "command": {"amp_rad": 0.08, "period_s": 7.0, "rate_rad_s": 1.0},
                    "style": "calm",
                    "score": 0.3,
                    "confidence": 0.5,
                    "urgency": 0.1,
                    "risk": "low",
                    "allow_interrupt": True,
                    "evidence": [],
                    "rationale_short": "fallback breathe",
                },
            ],
        }
    )


def _governed_candidate_with_invalid_compile_command() -> GovernedCandidate:
    proposal = IntentProposal(
        intent="track_user",
        primitive="orient_to_zone",
        command={"zone": "desk"},
        style="calm",
        score=0.9,
        confidence=0.8,
        urgency=0.4,
        risk="low",
        allow_interrupt=True,
        evidence=[],
        rationale_short="invalid zone command",
    )
    return GovernedCandidate(
        candidate=ProposalCandidate(proposal=proposal, source="remote"),
        valid=True,
        reject_reason=None,
        utility=0.95,
    )


def test_policy_shutdown_closes_logs_and_executor_with_fail_closed_behavior(tmp_path):
    policy = _make_policy(tmp_path)
    env_log = _MemoryLog()
    planner_log = _MemoryLog(fail_close=True)
    reasoning_log = _MemoryLog()
    trace = _MemoryLog()
    executor = _FakeExecutor(lambda: _DoneFuture(ModelResponse(True, 200, 1.0, None, None)))

    policy._env_log = env_log  # noqa: SLF001
    policy._planner_log = planner_log  # noqa: SLF001
    policy._reasoning_log = reasoning_log  # noqa: SLF001
    policy._trace = SimpleNamespace(close=trace.close)  # noqa: SLF001
    policy._executor = executor  # noqa: SLF001
    policy.shutdown()

    assert env_log.closed == 1
    assert planner_log.closed == 1
    assert reasoning_log.closed == 1
    assert trace.closed == 1
    assert executor.shutdown_calls == [(False, True)]


def test_policy_step_compile_failure_is_logged_and_persisted(tmp_path, monkeypatch):
    policy = _make_policy(tmp_path, persist_every_step=True)
    governed = _governed_candidate_with_invalid_compile_command()
    arb_result = ArbiterResult(
        decision="commit",
        reason="utility_beats_threshold",
        chosen=governed,
        best_utility=0.9,
        threshold=0.2,
        effective_current=0.1,
        margin=0.1,
    )

    monkeypatch.setattr(policy._governor, "evaluate", lambda candidates, mode, signals: [governed])  # noqa: SLF001
    monkeypatch.setattr(policy._arbiter, "select", lambda **kwargs: arb_result)  # noqa: SLF001
    persisted = []
    monkeypatch.setattr(policy._world_state, "persist", lambda: persisted.append(True))  # noqa: SLF001
    emitted = []
    monkeypatch.setattr(policy._trace, "emit", lambda payload: emitted.append(payload))  # noqa: SLF001

    action = policy.step(st=None)
    assert action.primitive == PrimitiveKind.HOLD
    assert persisted == [True]
    assert emitted[-1]["decision"]["reason"].startswith("compile_fail:")
    assert emitted[-1]["decision"]["committed"] is False


def test_policy_step_calls_remote_schedulers_when_remote_enabled(tmp_path, monkeypatch):
    policy = _make_policy(tmp_path, remote_enabled=True, base_url="http://unit.test")
    calls = []
    monkeypatch.setattr(policy, "_maybe_schedule_env", lambda *, st, now: calls.append(("env", st, now)))
    monkeypatch.setattr(policy, "_maybe_schedule_planner", lambda *, st, now: calls.append(("planner", st, now)))
    monkeypatch.setattr(policy._trace, "emit", lambda payload: None)  # noqa: SLF001

    try:
        policy.step(st=None)
    finally:
        policy.shutdown()

    assert [item[0] for item in calls] == ["env", "planner"]


def test_policy_ingest_latest_frame_handles_error_none_duplicate_and_new(tmp_path):
    policy = _make_policy(tmp_path)

    class _RaisingCache:
        def get(self, max_age_ms):
            raise RuntimeError("cache down")

    policy._frame_cache = _RaisingCache()  # noqa: SLF001
    policy._ingest_latest_frame()  # noqa: SLF001

    called_prune = []
    policy._frame_window.prune = lambda now_ns=None: called_prune.append(now_ns)  # noqa: SLF001

    class _NoneCache:
        def get(self, max_age_ms):
            return None

    policy._frame_cache = _NoneCache()  # noqa: SLF001
    policy._ingest_latest_frame()  # noqa: SLF001
    assert called_prune == [None]

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    snap = SimpleNamespace(frame=frame, mono_ns=1234)

    class _SnapCache:
        def get(self, max_age_ms):
            return snap

    policy._frame_cache = _SnapCache()  # noqa: SLF001
    policy._last_ingested_frame_ns = 1234  # noqa: SLF001
    policy._ingest_latest_frame()  # noqa: SLF001
    assert called_prune[-1] == 1234

    snap.mono_ns = 5678
    policy._ingest_latest_frame()  # noqa: SLF001
    assert policy._last_ingested_frame_ns == 5678  # noqa: SLF001


def test_policy_maybe_schedule_env_and_planner_guard_paths(tmp_path, monkeypatch):
    policy = _make_policy(tmp_path, request_min_fresh_frames=1)
    policy._model_client = _FakeModelClient()  # noqa: SLF001
    executor = _FakeExecutor(lambda: _PendingFuture())
    policy._executor = executor  # noqa: SLF001

    policy._cfg.env_hz = 0.0  # noqa: SLF001
    policy._maybe_schedule_env(st=None, now=10.0)  # noqa: SLF001
    assert policy._env_inflight is None  # noqa: SLF001

    policy._cfg.env_hz = 1.0  # noqa: SLF001
    policy._next_env_allowed_s = 20.0  # noqa: SLF001
    policy._maybe_schedule_env(st=None, now=10.0)  # noqa: SLF001
    assert policy._env_inflight is None  # noqa: SLF001

    policy._next_env_allowed_s = 0.0  # noqa: SLF001
    policy._last_env_submit_s = 9.8  # noqa: SLF001
    policy._maybe_schedule_env(st=None, now=10.0)  # noqa: SLF001
    assert policy._env_inflight is None  # noqa: SLF001

    policy._env_inflight = object()  # noqa: SLF001
    policy._maybe_schedule_env(st=None, now=11.5)  # noqa: SLF001
    assert policy._env_summarizer.take_latest_pending() is not None  # noqa: SLF001
    policy._env_inflight = None  # noqa: SLF001

    monkeypatch.setattr(policy, "_build_env_payload", lambda: None)
    policy._maybe_schedule_env(st=None, now=12.0)  # noqa: SLF001
    assert policy._env_inflight is None  # noqa: SLF001

    monkeypatch.setattr(policy, "_build_env_payload", lambda: {"request": SimpleNamespace(), "frames": 1})
    policy._env_summarizer.submit_or_replace({"req": "blocked"})  # noqa: SLF001
    assert policy._env_summarizer.in_flight is True  # noqa: SLF001
    policy._maybe_schedule_env(st=None, now=13.0)  # noqa: SLF001
    assert policy._env_request_seq == 0  # noqa: SLF001
    policy._env_summarizer.complete_request("")  # noqa: SLF001

    policy._last_env_submit_s = 0.0  # noqa: SLF001
    policy._env_log = _MemoryLog()  # noqa: SLF001
    policy._maybe_schedule_env(st=None, now=20.0)  # noqa: SLF001
    assert policy._env_inflight is not None  # noqa: SLF001
    assert executor.calls
    assert policy._env_log.items[-1]["status"] == "req_start"  # noqa: SLF001

    monkeypatch.setattr(policy, "_build_planner_payload", lambda *, st, now: None)
    policy._maybe_schedule_planner(st=None, now=20.0)  # noqa: SLF001
    assert policy._planner_inflight is None  # noqa: SLF001

    monkeypatch.setattr(policy, "_build_planner_payload", lambda *, st, now: {"request": SimpleNamespace(), "frames": 1})
    policy._pending_planner_event = True  # noqa: SLF001
    policy._last_planner_event_submit_s = 0.0  # noqa: SLF001
    policy._last_planner_submit_s = 20.0  # noqa: SLF001
    policy._planner_log = _MemoryLog()  # noqa: SLF001
    policy._maybe_schedule_planner(st=None, now=22.0)  # noqa: SLF001
    assert policy._planner_inflight is not None  # noqa: SLF001
    assert policy._pending_planner_event is False  # noqa: SLF001
    assert policy._planner_log.items[-1]["status"] == "req_start"  # noqa: SLF001


def test_policy_drain_env_inflight_success_parse_fail_and_pending_resubmit(tmp_path, monkeypatch):
    policy = _make_policy(tmp_path)
    policy._env_log = _MemoryLog()  # noqa: SLF001
    policy._reasoning_log = _MemoryLog()  # noqa: SLF001
    scheduled = []
    monkeypatch.setattr(policy, "_maybe_schedule_env", lambda *, st, now: scheduled.append((st, now)))

    policy._env_summarizer.mark_pending({"queued": True})  # noqa: SLF001
    policy._env_inflight = _InFlightCall(
        request_id=1,
        started_mono_s=1.0,
        future=_DoneFuture(
            ModelResponse(
                ok=True,
                status_code=200,
                latency_ms=12.0,
                response_json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": _valid_env_content(0.91), "reasoning_content": "chain"},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
                error=None,
            )
        ),
    )
    policy._drain_env_inflight(st=None, now=10.0)  # noqa: SLF001
    latest = policy.world_state.snapshot()["latest_env_snapshot"]
    assert latest["delta_score"] == 0.91
    assert policy._pending_planner_event is True  # noqa: SLF001
    assert policy._next_env_allowed_s == 0.0  # noqa: SLF001
    assert policy._env_log.items[-1]["status"] == "ok"  # noqa: SLF001
    assert policy._reasoning_log.items[-1]["component"] == "env_processor"  # noqa: SLF001
    assert scheduled == [(None, 10.0)]

    policy._env_inflight = _InFlightCall(
        request_id=2,
        started_mono_s=2.0,
        future=_DoneFuture(
            ModelResponse(
                ok=True,
                status_code=200,
                latency_ms=10.0,
                response_json={"choices": [{"message": {"content": "{bad json"}}]},
                error=None,
            )
        ),
    )
    policy._drain_env_inflight(st=None, now=12.0)  # noqa: SLF001
    assert policy._env_log.items[-1]["status"] == "parse_fail"  # noqa: SLF001
    assert (policy._env_log.items[-1]["error"] or "").startswith("env_json_parse_failed:")  # noqa: SLF001
    assert policy._next_env_allowed_s >= 12.0 + policy._cfg.error_backoff_s  # noqa: SLF001


def test_policy_drain_planner_inflight_success_watchdog_and_parse_fail(tmp_path, monkeypatch):
    policy = _make_policy(tmp_path, planner_max_proposals=1)
    policy._planner_log = _MemoryLog()  # noqa: SLF001
    policy._reasoning_log = _MemoryLog()  # noqa: SLF001
    scheduled = []
    monkeypatch.setattr(policy, "_maybe_schedule_planner", lambda *, st, now: scheduled.append((st, now)))

    policy._intent_proposer.mark_pending({"queued": True})  # noqa: SLF001
    policy._planner_inflight = _InFlightCall(
        request_id=1,
        started_mono_s=1.0,
        future=_DoneFuture(
            ModelResponse(
                ok=True,
                status_code=200,
                latency_ms=11.0,
                response_json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": _valid_planner_content(), "reasoning_content": "planner-thought"},
                        }
                    ],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
                },
                error=None,
            )
        ),
    )
    policy._drain_planner_inflight(st=None, now=10.0)  # noqa: SLF001
    assert policy._latest_remote_proposals is not None  # noqa: SLF001
    assert len(policy._latest_remote_proposals.proposals) == 1  # noqa: SLF001
    assert policy._planner_log.items[-1]["status"] == "ok"  # noqa: SLF001
    assert policy._reasoning_log.items[-1]["component"] == "planner"  # noqa: SLF001
    assert scheduled == [(None, 10.0)]

    pending = _PendingFuture()
    policy._planner_inflight = _InFlightCall(
        request_id=2,
        started_mono_s=1.0,
        future=pending,
    )
    policy._drain_planner_inflight(st=None, now=10.0)  # noqa: SLF001
    assert pending.cancel_called is True
    assert policy._planner_log.items[-1]["status"] == "transport_error"  # noqa: SLF001

    policy._planner_inflight = _InFlightCall(
        request_id=3,
        started_mono_s=2.0,
        future=_DoneFuture(
            ModelResponse(
                ok=True,
                status_code=200,
                latency_ms=10.0,
                response_json={"choices": [{"message": {"content": "{bad json"}}]},
                error=None,
            )
        ),
    )
    policy._drain_planner_inflight(st=None, now=12.0)  # noqa: SLF001
    assert policy._planner_log.items[-1]["status"] == "parse_fail"  # noqa: SLF001
    assert policy._next_planner_allowed_s >= 12.0 + policy._cfg.error_backoff_s  # noqa: SLF001


def test_policy_helper_methods_and_text_encoding_quality(tmp_path):
    policy = _make_policy(tmp_path)
    policy._recent_commit_times.extend([0.0, 5.0, 9.5])  # noqa: SLF001
    assert policy._recent_switch_count(10.0) == 2  # noqa: SLF001
    assert policy._zone_hint(st=None, snapshot={}) is None  # noqa: SLF001
    snap = {"latest_env_snapshot": {"features": {"zone_hint": "right"}}}
    assert policy._zone_hint(st=None, snapshot=snap) == "right"  # noqa: SLF001
    assert policy._zone_hint(st=None, snapshot={}) == "right"  # noqa: SLF001

    st = SimpleNamespace(primary_person_conf=0.7, debug={"detector_alive": True})
    signals = policy._build_mode_signals(  # noqa: SLF001
        st=st,
        snapshot={"latest_env_snapshot": {"features": {"person_present": True}}},
    )
    assert signals.person_present is True
    assert abs(signals.person_conf - 1.0) < 1e-6

    assert policy._compute_failure_backoff_s(ModelResponse(False, 404, 1.0, None, "x")) == policy._cfg.client_error_backoff_s  # noqa: SLF001
    assert policy._compute_failure_backoff_s(ModelResponse(False, 500, 1.0, None, "x")) == policy._cfg.error_backoff_s  # noqa: SLF001
    assert policy._request_timeout_s() >= 0.25  # noqa: SLF001
    assert policy._watchdog_timeout_s() > policy._request_timeout_s()  # noqa: SLF001

    class _BadCancel:
        def cancel(self):
            raise RuntimeError("cancel")

    policy._cancel_future(_BadCancel())  # noqa: SLF001
    policy._write_log(_MemoryLog(fail_write=True), {"x": 1})  # noqa: SLF001
    assert policy._preview_text("x" * 20, max_chars=10) == "xxxxxxx..."  # noqa: SLF001

    content, reasoning = extract_message_content(
        {
            "choices": [
                {
                    "message": {
                        "content": [{"text": "line1"}, {"text": "line2"}],
                        "reasoning_content": {"text": "r1"},
                    },
                }
            ]
        }
    )
    assert content == "line1\nline2"
    assert reasoning == "r1"
    assert extract_message_content({"choices": []}) == (None, None)
    assert _coerce_text({"text": " x "}) == "x"
    assert _coerce_text([{"text": "a"}, {"bad": "x"}, {"text": "b"}]) == "a\nb"
    assert _coerce_text(7) is None

    data_url = _encode_frame_data_url(frame=np.ones((8, 64, 3), dtype=np.float32), max_width=16, jpeg_quality=5)
    assert data_url.startswith("data:image/jpeg;base64,")

    finish, prompt, completion, total = _response_meta(
        {
            "choices": [{"finish_reason": " stop "}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
    )
    assert finish == "stop"
    assert (prompt, completion, total) == (4, 2, 6)
    assert _response_meta({"choices": [1], "usage": {"prompt_tokens": "x"}}) == (None, None, None, None)
