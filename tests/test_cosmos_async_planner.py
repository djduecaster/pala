import json
import time

import numpy as np

from pala.perception.frame_cache import LatestFrameCache
from pala.planner.cosmos_async import AsyncCosmosPlanner
from pala.types import BBoxNorm, PerceptionState
from pala.control.primitives import PrimitiveKind


def _state_with_person() -> PerceptionState:
    return PerceptionState(
        timestamp_monotonic_s=time.monotonic(),
        primary_person=BBoxNorm(cx=0.5, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.9,
        debug={"zone_hint": "center"},
    )


def test_async_cosmos_planner_remote_response_used(monkeypatch):
    calls = {"count": 0}
    prompt = "Be extra gentle and avoid sudden motion."

    def _fake_post(url, payload, *, timeout_s, api_key):
        calls["count"] += 1
        assert url.endswith("/v1/chat/completions")
        assert payload.get("model") == "nvidia/cosmos-reason2-2b"
        assert prompt in payload["messages"][0]["content"]
        user_ctx = json.loads(payload["messages"][1]["content"])
        assert user_ctx.get("planner_prompt") == prompt
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"primitive":"breath","command":{"amp_rad":0.11,"period_s":5.5,"rate_rad_s":1.0},'
                            '"confidence":0.66,"explanation":"remote ok"}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr("pala.planner.cosmos_async._post_json", _fake_post)

    frame_cache = LatestFrameCache()
    frame_cache.set(np.zeros((2, 2, 3), dtype=np.uint8), mono_ns=time.monotonic_ns(), pts_ns=None)
    planner = AsyncCosmosPlanner(
        frame_cache=frame_cache,
        provider="brev",
        base_url="http://127.0.0.1:8000",
        model="nvidia/cosmos-reason2-2b",
        planner_prompt=prompt,
        request_timeout_ms=200,
        max_hz=50.0,
        response_ttl_ms=1500,
    )

    try:
        st = _state_with_person()
        got = None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            got = planner.plan(st)
            if got.explanation == "remote ok":
                break
            time.sleep(0.02)

        assert got is not None
        assert got.explanation == "cosmos_remote:remote ok"
        assert got.primitive == PrimitiveKind.BREATH
        assert calls["count"] >= 1
    finally:
        planner.shutdown()


def test_async_cosmos_planner_remote_invalid_response_falls_back(monkeypatch):
    calls = {"count": 0}

    def _fake_post(url, payload, *, timeout_s, api_key):
        calls["count"] += 1
        return {"choices": [{"message": {"content": "not-json"}}]}

    monkeypatch.setattr("pala.planner.cosmos_async._post_json", _fake_post)

    planner = AsyncCosmosPlanner(
        frame_cache=LatestFrameCache(),
        provider="brev",
        base_url="http://127.0.0.1:8000",
        request_timeout_ms=200,
        max_hz=50.0,
        response_ttl_ms=400,
    )

    try:
        st = _state_with_person()
        actions = []
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            actions.append(planner.plan(st))
            time.sleep(0.02)

        assert calls["count"] >= 1
        assert actions
        assert actions[-1].primitive == PrimitiveKind.BREATH
        assert actions[-1].explanation == "idle presence"
    finally:
        planner.shutdown()
