import time

import numpy as np

from pala.perception.frame_cache import LatestFrameCache
from pala.planner.cosmos_async import AsyncCosmosPlanner
from pala.types import BBoxNorm, PerceptionState


def _state_with_person() -> PerceptionState:
    now = time.monotonic()
    return PerceptionState(
        timestamp_monotonic_s=now,
        timestamp_wall_s=time.time(),
        primary_person=BBoxNorm(cx=0.5, cy=0.5, w=0.2, h=0.4),
        primary_person_conf=0.9,
        debug={"zone_hint": "center"},
    )


def test_latest_frame_cache_stale_filter():
    cache = LatestFrameCache()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    stale_ns = time.monotonic_ns() - 2_000_000_000
    cache.set(frame, mono_ns=stale_ns, pts_ns=None)
    assert cache.get(max_age_ms=100) is None

    fresh_ns = time.monotonic_ns()
    cache.set(frame, mono_ns=fresh_ns, pts_ns=None)
    snap = cache.get(max_age_ms=100)
    assert snap is not None
    assert snap.frame.shape == (2, 2, 3)


def test_async_cosmos_planner_returns_mock_action_eventually():
    cache = LatestFrameCache()
    cache.set(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=time.monotonic_ns(), pts_ns=None)

    planner = AsyncCosmosPlanner(
        frame_cache=cache,
        max_hz=100.0,
        max_frame_age_ms=1000,
        mock_latency_ms=10,
        response_ttl_ms=1000,
    )
    try:
        st = _state_with_person()
        first = planner.plan(st)
        # No async result yet; should be fallback action.
        assert not (first.explanation or "").startswith("cosmos_mock:")

        deadline = time.monotonic() + 1.0
        got_mock = False
        while time.monotonic() < deadline:
            act = planner.plan(st)
            if (act.explanation or "").startswith("cosmos_mock:"):
                got_mock = True
                break
            time.sleep(0.01)

        assert got_mock
    finally:
        planner.shutdown()
