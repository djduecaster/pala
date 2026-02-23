from __future__ import annotations

from dataclasses import dataclass

from pala.behavior.policy import (
    BehaviorPolicy,
    BehaviorPolicyConfig,
    _InFlightCall,
    _debug_get,
    _infer_zone_hint_from_text,
    _response_meta,
)
from pala.behavior.remote_api import RemoteCallResult
from pala.behavior.world_state_store import WorldStateStore, WorldStateStoreConfig
from pala.types import PerceptionState


def _make_policy(tmp_path) -> BehaviorPolicy:
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )
    cfg = BehaviorPolicyConfig(
        remote_enabled=False,
        env_log_path=None,
        planner_log_path=None,
        reasoning_log_path=None,
        trace_log_path=None,
    )
    return BehaviorPolicy(world_state=store, config=cfg, clock=lambda: 10.0)


def test_infer_zone_hint_from_text_prefers_earliest_marker():
    assert _infer_zone_hint_from_text(None, "", "   ") == "unknown"
    assert _infer_zone_hint_from_text("person is on my right then moves left", None) == "right"
    assert _infer_zone_hint_from_text("toward middle area", "in front of me", None) == "center"
    assert _infer_zone_hint_from_text("left side and then right side") == "left"


def test_response_meta_parses_and_handles_invalid_shapes():
    finish, prompt, completion, total = _response_meta(
        {
            "choices": [{"finish_reason": " stop "}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
    )
    assert finish == "stop"
    assert (prompt, completion, total) == (11, 7, 18)

    finish, prompt, completion, total = _response_meta(
        {"choices": [{"finish_reason": ""}], "usage": {"prompt_tokens": "11"}}
    )
    assert finish is None
    assert prompt is None
    assert completion is None
    assert total is None

    finish, prompt, completion, total = _response_meta({"choices": [123], "usage": []})
    assert finish is None
    assert prompt is None
    assert completion is None
    assert total is None


def test_response_meta_survives_mapping_get_errors():
    class _BrokenMap(dict):
        def get(self, key, default=None):  # noqa: ANN001
            raise RuntimeError(f"boom:{key}")

    finish, prompt, completion, total = _response_meta(_BrokenMap())
    assert finish is None
    assert prompt is None
    assert completion is None
    assert total is None


def test_policy_safe_future_result_invalid_and_cancel_write_guards(tmp_path):
    policy = _make_policy(tmp_path)

    class _InvalidFuture:
        def result(self):
            return "not-remote-call-result"

    call = _InFlightCall(request_id=1, started_mono_s=8.0, payload={}, future=_InvalidFuture())
    result = policy._safe_future_result(call=call, now=10.0)  # noqa: SLF001
    assert isinstance(result, RemoteCallResult)
    assert result.ok is False
    assert (result.error or "").startswith("future_invalid_result:str")

    class _BadCancelFuture:
        def cancel(self):
            raise RuntimeError("cancel failed")

    # Should not raise.
    policy._cancel_future(_BadCancelFuture())  # noqa: SLF001

    class _BadLog:
        def write(self, payload):  # noqa: ANN001
            raise RuntimeError("write failed")

    # Should not raise.
    policy._write_log(_BadLog(), {"x": 1})  # noqa: SLF001


def test_debug_get_reads_mapping_only():
    assert _debug_get(None, "zone_hint") is None

    st = PerceptionState(timestamp_monotonic_s=0.0, debug={"zone_hint": "left"})
    assert _debug_get(st, "zone_hint") == "left"

    @dataclass
    class _State:
        debug: object

    assert _debug_get(_State(debug=["not", "mapping"]), "zone_hint") is None
