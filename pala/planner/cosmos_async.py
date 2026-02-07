from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import threading
import time

from ..control.primitives import PRIMITIVE_BREATH, PRIMITIVE_HOLD
from ..perception.frame_cache import LatestFrameCache
from ..types import ActionPlan, PerceptionState
from .interface import HeuristicPlanner, PlannerInterface


@dataclass
class _CosmosRequest:
    state: PerceptionState
    frame_mono_ns: Optional[int]
    frame_shape: Optional[tuple[int, int, int]]


class AsyncCosmosPlanner:
    """Phase-1 async planner skeleton for Cosmos/Brev integration.

    This implementation intentionally does not call external services yet.
    It runs a background worker that consumes latest requests and emits
    mock planner actions to validate non-blocking architecture and wiring.
    """

    def __init__(
        self,
        *,
        frame_cache: LatestFrameCache,
        fallback: Optional[PlannerInterface] = None,
        max_hz: float = 1.0,
        max_frame_age_ms: int = 500,
        mock_latency_ms: int = 150,
        response_ttl_ms: int = 1500,
    ) -> None:
        self._frame_cache = frame_cache
        self._fallback = fallback or HeuristicPlanner()
        self._max_hz = max(0.1, float(max_hz))
        self._max_frame_age_ms = int(max_frame_age_ms)
        self._mock_latency_s = max(0.0, float(mock_latency_ms) / 1000.0)
        self._response_ttl_s = max(0.1, float(response_ttl_ms) / 1000.0)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._pending: Optional[_CosmosRequest] = None
        self._last_submit_s = 0.0
        self._latest_action: Optional[ActionPlan] = None
        self._latest_action_ts_s: Optional[float] = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def plan(self, st: PerceptionState) -> ActionPlan:
        now = time.monotonic()
        submit_period_s = 1.0 / self._max_hz
        if (now - self._last_submit_s) >= submit_period_s:
            snap = self._frame_cache.get(max_age_ms=self._max_frame_age_ms)
            frame_shape = None if snap is None else tuple(snap.frame.shape)
            req = _CosmosRequest(
                state=st,
                frame_mono_ns=None if snap is None else snap.mono_ns,
                frame_shape=frame_shape,
            )
            with self._lock:
                self._pending = req
                self._last_submit_s = now
                self._cond.notify_all()

        with self._lock:
            action = self._latest_action
            action_ts = self._latest_action_ts_s

        if action is not None and action_ts is not None and (now - action_ts) <= self._response_ttl_s:
            return action
        return self._fallback.plan(st)

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                while not self._stop.is_set() and self._pending is None:
                    self._cond.wait(timeout=0.1)
                if self._stop.is_set():
                    break
                req = self._pending
                self._pending = None

            if req is None:
                continue

            if self._mock_latency_s > 0:
                time.sleep(self._mock_latency_s)

            action = self._mock_plan(req)
            with self._lock:
                self._latest_action = action
                self._latest_action_ts_s = time.monotonic()

    def _mock_plan(self, req: _CosmosRequest) -> ActionPlan:
        if req.state.primary_person is None:
            return ActionPlan(
                primitive=PRIMITIVE_HOLD,
                params={},
                confidence=0.25,
                explanation="cosmos_mock:no_person",
            )

        has_frame = req.frame_mono_ns is not None and req.frame_shape is not None
        return ActionPlan(
            primitive=PRIMITIVE_BREATH,
            params={"amp_rad": 0.1, "period_s": 6.0, "rate_rad_s": 1.1},
            confidence=0.45,
            explanation=f"cosmos_mock:presence frame={has_frame}",
        )
