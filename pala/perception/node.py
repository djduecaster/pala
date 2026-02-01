from __future__ import annotations

import time
from typing import Optional

from ..types import PerceptionState, BBoxNorm, PointNorm
from .frame_source import FrameSource, DummyFrameSource


class PerceptionNode:
    """Dummy perception node. Produces a moving bbox in normalized coords."""

    def __init__(self, source: Optional[FrameSource] = None):
        self.source = source or DummyFrameSource()
        self._last_ts = None
        self._fps = None

    def step(self) -> PerceptionState:
        ts_mono = self.source.get_timestamp()
        ts_wall = time.time()

        if self._last_ts is not None:
            dt = ts_mono - self._last_ts
            if dt > 0:
                self._fps = 1.0 / dt
        self._last_ts = ts_mono

        # Dummy person bbox moving left/center/right
        if isinstance(self.source, DummyFrameSource):
            cx = self.source.dummy_position()
        else:
            cx = 0.5

        bbox = BBoxNorm(cx=cx, cy=0.5, w=0.2, h=0.4)

        # Optional pointing target: when near right, point to top-right
        pointing = None
        pointing_conf = None
        if cx > 0.7:
            pointing = PointNorm(x=0.85, y=0.2)
            pointing_conf = 0.6

        return PerceptionState(
            timestamp_monotonic_s=ts_mono,
            timestamp_wall_s=ts_wall,
            fps=self._fps,
            latency_ms=5.0,
            primary_person=bbox,
            primary_person_conf=0.8,
            pointing_target=pointing,
            pointing_conf=pointing_conf,
            debug={"zone_hint": _zone_from_cx(cx)},
        )

    def shutdown(self) -> None:
        self.source.shutdown()


def _zone_from_cx(cx: float) -> str:
    if cx < 0.33:
        return "left"
    if cx < 0.66:
        return "center"
    return "right"
