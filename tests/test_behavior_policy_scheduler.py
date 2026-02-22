from __future__ import annotations

from dataclasses import dataclass

from pala.behavior.policy import BehaviorPolicy, BehaviorPolicyConfig, _InFlightCall
from pala.behavior.planner_client import PlannerDecision
from pala.behavior.remote_api import RemoteCallResult
from pala.behavior.world_state_store import EnvironmentSnapshot, WorldStateStore, WorldStateStoreConfig
from pala.types import ActionPlan, BreathCommand, HomeCommand
from pala.control.primitives import PrimitiveKind


class _DoneFuture:
    def __init__(self, result):
        self._result = result

    def done(self) -> bool:
        return True

    def result(self):
        return self._result


class _NeverDoneFuture:
    def done(self) -> bool:
        return False

    def result(self):  # pragma: no cover - should never be called
        raise AssertionError("result() should not be called when done() is False")


class _CaptureExecutor:
    def __init__(self, futures):
        self._futures = list(futures)
        self.calls = []

    def submit(self, fn, **kwargs):
        self.calls.append({"fn": fn, "kwargs": kwargs})
        if self._futures:
            return self._futures.pop(0)
        return _NeverDoneFuture()


@dataclass
class _CaptureLog:
    items: list

    def write(self, obj) -> None:
        self.items.append(obj)

    def close(self) -> None:
        return None


def _make_policy(tmp_path, **cfg_overrides) -> BehaviorPolicy:
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    cfg_kwargs = {
        "remote_enabled": True,
        "base_url": "http://unit.test",
        "env_hz": 2.0,
        "planner_hz": 2.0,
        "request_timeout_ms": 500,
        "error_backoff_s": 0.0,
        "client_error_backoff_s": 0.0,
        "env_log_path": None,
        "planner_log_path": None,
        "reasoning_log_path": None,
    }
    cfg_kwargs.update(cfg_overrides)
    cfg = BehaviorPolicyConfig(**cfg_kwargs)
    return BehaviorPolicy(planner=None, world_state=store, config=cfg)


def _ok_env_result(delta: float = 0.2) -> RemoteCallResult:
    content = (
        "<scene>desk</scene>"
        "<events>idle</events>"
        "<hypotheses>none</hypotheses>"
        "<opportunities>none</opportunities>"
        "<uncertainties>none</uncertainties>"
        f"<delta_score>{delta}</delta_score>"
        "<summary>steady</summary>"
    )
    return RemoteCallResult(
        ok=True,
        status_code=200,
        latency_ms=12.0,
        response_json={"choices": [{"message": {"content": content}}]},
        error=None,
    )


def test_maybe_schedule_env_respects_inflight_and_period(monkeypatch, tmp_path):
    policy = _make_policy(tmp_path, env_hz=2.0)
    policy._executor = _CaptureExecutor([_NeverDoneFuture(), _NeverDoneFuture()])
    monkeypatch.setattr(policy, "_build_env_payload", lambda st: {"body": {"m": "x"}, "frames": 1})

    policy._maybe_schedule_env(st=None, now=1.0)
    assert len(policy._executor.calls) == 1
    assert policy._env_inflight is not None

    policy._maybe_schedule_env(st=None, now=1.1)
    assert len(policy._executor.calls) == 1
    assert policy._env_processor._pending_payload is not None

    policy._env_processor.complete_request("")
    policy._env_inflight = None
    policy._maybe_schedule_env(st=None, now=1.2)
    assert len(policy._executor.calls) == 1

    policy._maybe_schedule_env(st=None, now=1.6)
    assert len(policy._executor.calls) == 2


def test_maybe_schedule_planner_event_cooldown_and_inflight_pending(monkeypatch, tmp_path):
    policy = _make_policy(tmp_path, planner_hz=0.5, planner_event_cooldown_s=0.7)
    policy._executor = _CaptureExecutor([_NeverDoneFuture()])
    monkeypatch.setattr(policy, "_build_planner_payload", lambda st: {"body": {"m": "x"}, "frames": 0})

    policy._pending_planner_event = True
    policy._last_planner_submit_s = 1.0
    policy._last_planner_event_submit_s = 1.0
    policy._maybe_schedule_planner(st=None, now=1.1)
    assert len(policy._executor.calls) == 0

    policy._maybe_schedule_planner(st=None, now=1.8)
    assert len(policy._executor.calls) == 1
    assert policy._pending_planner_event is False
    assert policy._last_planner_event_submit_s == 1.8

    policy._maybe_schedule_planner(st=None, now=1.9)
    assert len(policy._executor.calls) == 1
    assert policy._planner_client._pending_payload is not None


def test_drain_env_accepts_partial_env_tags_and_updates_snapshot(tmp_path):
    policy = _make_policy(tmp_path)
    env_log = _CaptureLog(items=[])
    policy._env_log = env_log
    policy._env_processor.submit_or_replace({"req": 1})
    policy._env_inflight = _InFlightCall(
        request_id=7,
        started_mono_s=0.0,
        payload={},
        future=_DoneFuture(
            RemoteCallResult(
                ok=True,
                status_code=200,
                latency_ms=15.0,
                response_json={"choices": [{"message": {"content": "<scene>only</scene>"}}]},
                error=None,
            )
        ),
    )

    policy._drain_env_inflight(now=3.0)
    assert policy.world_state.latest_env_snapshot is not None
    assert env_log.items[-1]["status"] == "ok"
    assert env_log.items[-1]["error"] is None
    assert policy._pending_planner_event is False


def test_drain_planner_parse_fail_keeps_current_action_and_logs(tmp_path):
    policy = _make_policy(tmp_path)
    planner_log = _CaptureLog(items=[])
    policy._planner_log = planner_log
    current_id = policy._current_action.action_id
    policy._planner_client.submit_or_replace({"req": 1})
    policy._planner_inflight = _InFlightCall(
        request_id=9,
        started_mono_s=0.0,
        payload={},
        future=_DoneFuture(
            RemoteCallResult(
                ok=True,
                status_code=200,
                latency_ms=20.0,
                response_json={"choices": [{"message": {"content": "<rationale_short>missing decision</rationale_short>"}}]},
                error=None,
            )
        ),
    )

    policy._drain_planner_inflight(now=3.0)
    assert policy._current_action.action_id == current_id
    assert planner_log.items[-1]["status"] == "parse_fail"
    assert planner_log.items[-1]["error"] == "planner_tag_parse_failed"
    assert policy.world_state.snapshot()["decision_tail"] == []


def test_drain_env_replays_latest_pending_immediately(monkeypatch, tmp_path):
    policy = _make_policy(tmp_path, env_hz=1.0)
    executor = _CaptureExecutor([_NeverDoneFuture()])
    policy._executor = executor
    monkeypatch.setattr(policy, "_build_env_payload", lambda st: {"body": {"m": "x"}, "frames": 1})

    policy._env_processor.submit_or_replace({"req": "initial"})
    policy._env_processor.submit_or_replace({"req": "latest"})
    policy._env_inflight = _InFlightCall(
        request_id=1,
        started_mono_s=0.0,
        payload={},
        future=_DoneFuture(_ok_env_result(delta=0.1)),
    )

    policy._drain_env_inflight(now=10.0)
    assert len(executor.calls) == 1
    assert policy._env_inflight is not None
    assert policy._env_request_seq == 1


def test_drain_planner_replays_latest_pending_immediately(monkeypatch, tmp_path):
    policy = _make_policy(tmp_path, planner_hz=1.0)
    executor = _CaptureExecutor([_NeverDoneFuture()])
    policy._executor = executor
    monkeypatch.setattr(policy, "_build_planner_payload", lambda st: {"body": {"m": "x"}, "frames": 0})

    policy._planner_client.submit_or_replace({"req": "initial"})
    policy._planner_client.submit_or_replace({"req": "latest"})
    policy._planner_inflight = _InFlightCall(
        request_id=2,
        started_mono_s=0.0,
        payload={},
        future=_DoneFuture(
            RemoteCallResult(
                ok=False,
                status_code=0,
                latency_ms=4.0,
                response_json=None,
                error="transport:down",
            )
        ),
    )

    policy._drain_planner_inflight(now=10.0)
    assert len(executor.calls) == 1
    assert policy._planner_inflight is not None
    assert policy._planner_request_seq == 1


def test_planner_client_error_applies_backoff(monkeypatch, tmp_path):
    policy = _make_policy(
        tmp_path,
        planner_hz=20.0,
        error_backoff_s=0.5,
        client_error_backoff_s=3.0,
    )
    executor = _CaptureExecutor([_NeverDoneFuture()])
    policy._executor = executor
    monkeypatch.setattr(policy, "_build_planner_payload", lambda st: {"body": {"m": "x"}, "frames": 0})
    policy._planner_client.submit_or_replace({"req": "initial"})
    policy._planner_inflight = _InFlightCall(
        request_id=2,
        started_mono_s=0.0,
        payload={},
        future=_DoneFuture(
            RemoteCallResult(
                ok=False,
                status_code=400,
                latency_ms=4.0,
                response_json=None,
                error="http_400:bad request",
            )
        ),
    )

    policy._drain_planner_inflight(now=10.0)
    assert policy._next_planner_allowed_s >= 13.0

    policy._maybe_schedule_planner(st=None, now=11.0)
    assert len(executor.calls) == 0

    policy._maybe_schedule_planner(st=None, now=13.1)
    assert len(executor.calls) == 1


def test_repeated_home_action_is_suppressed_when_delta_is_low(tmp_path):
    policy = _make_policy(tmp_path)
    policy._current_action = ActionPlan(
        primitive=PrimitiveKind.HOME,
        command=HomeCommand(),
        confidence=0.9,
        style="calm",
    )
    policy.world_state.update_environment(
        EnvironmentSnapshot(
            scene="desk",
            events="minimal motion",
            hypotheses="idle",
            opportunities="none",
            uncertainties="none",
            summary="steady scene",
            delta_score=0.2,
        )
    )
    repeated = ActionPlan(
        primitive=PrimitiveKind.HOME,
        command=HomeCommand(),
        confidence=0.9,
        style="calm",
    )
    assert policy._should_commit_action(repeated) is False


def test_repeated_home_action_allowed_when_delta_is_high(tmp_path):
    policy = _make_policy(tmp_path)
    policy._current_action = ActionPlan(
        primitive=PrimitiveKind.HOME,
        command=HomeCommand(),
        confidence=0.9,
        style="calm",
    )
    policy.world_state.update_environment(
        EnvironmentSnapshot(
            scene="person enters from left quickly",
            events="large transition",
            hypotheses="user returning to desk",
            opportunities="re-center",
            uncertainties="none",
            summary="major scene transition",
            delta_score=0.9,
        )
    )
    repeated = ActionPlan(
        primitive=PrimitiveKind.HOME,
        command=HomeCommand(),
        confidence=0.9,
        style="calm",
    )
    assert policy._should_commit_action(repeated) is True


def test_env_delta_normalization_caps_redundant_static_summaries(tmp_path):
    policy = _make_policy(tmp_path)
    policy.world_state.update_environment(
        EnvironmentSnapshot(
            scene="desk with user",
            events="user seated and typing",
            hypotheses="focused work",
            opportunities="maintain light",
            uncertainties="none",
            summary="user is seated and typing at desk",
            delta_score=0.8,
        )
    )
    snapshot = EnvironmentSnapshot(
        scene="desk with user and monitor",
        events="no observable changes",
        hypotheses="focused work",
        opportunities="maintain light",
        uncertainties="none",
        summary="user is seated and typing at desk",
        delta_score=0.99,
    )
    normalized = policy._normalize_env_delta(snapshot)
    assert normalized == 0.2


def test_repetition_guard_suppresses_repeated_breath_decisions(tmp_path):
    policy = _make_policy(tmp_path)
    policy._current_action = ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command=BreathCommand(),
        confidence=0.5,
        style="calm",
    )
    policy._cfg.repeated_breath_guard_count = 2
    decision = policy._apply_repetition_guard(
        # first breath keeps act_now true
        PlannerDecision(
            act_now=True,
            primitive="breath",
            command={},
            style="calm",
            confidence=0.7,
            rationale_short="continue breathing",
            reasoning_text=None,
            raw_text="raw",
        )
    )
    assert decision.act_now is True
    decision = policy._apply_repetition_guard(
        PlannerDecision(
            act_now=True,
            primitive="breath",
            command={},
            style="calm",
            confidence=0.7,
            rationale_short="continue breathing",
            reasoning_text=None,
            raw_text="raw",
        )
    )
    assert decision.act_now is False
    assert decision.primitive is None


def test_build_env_context_includes_frame_timeline(tmp_path):
    policy = _make_policy(tmp_path)
    timeline = [{"ordinal": 1, "age_s": 2.0}, {"ordinal": 2, "age_s": 0.0}]
    ctx = policy._build_env_context(st=None, frame_timeline=timeline)
    assert ctx["frame_timeline"] == timeline
