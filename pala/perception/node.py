from __future__ import annotations

from collections import deque
import logging
import time
from typing import Optional

from ..types import PerceptionState
from .frame_source import DummyFrameSource, FramePacket, FrameSource

logger = logging.getLogger(__name__)


class PerceptionNode:
    """Expose camera capture state without adding scene interpretation."""

    def __init__(self, source: Optional[FrameSource] = None) -> None:
        self.source = source or DummyFrameSource()
        self._last_packet: Optional[FramePacket] = None
        self._frame_times: deque[int] = deque(maxlen=30)
        self._frame_id = 0
        self._fps: Optional[float] = None
        self._last_source_error: Optional[str] = None
        self._last_source_warn_s = 0.0

    def step(self) -> PerceptionState:
        packet, is_new = self._acquire_packet()
        now_mono_ns = time.monotonic_ns()
        now_wall_s = time.time()

        if packet is None:
            debug = {"no_frame": True}
            if self._last_source_error is not None:
                debug["source_error"] = self._last_source_error
            return PerceptionState(
                timestamp_monotonic_s=now_mono_ns / 1_000_000_000.0,
                timestamp_wall_s=now_wall_s,
                frame_id=None,
                fps=self._fps,
                latency_ms=None,
                frame_age_ms=None,
                source_alive=self._last_source_error is None,
                is_new_frame=False,
                debug=debug,
            )

        if is_new:
            self._frame_id += 1
            self._frame_times.append(packet.mono_ns)
            self._fps = _fps_from_window(self._frame_times)

        frame_age_ms = max(0.0, (now_mono_ns - packet.mono_ns) / 1_000_000.0)
        debug = {}
        if not is_new:
            debug["stale_frame"] = True
        if self._last_source_error is not None:
            debug["source_error"] = self._last_source_error

        return PerceptionState(
            timestamp_monotonic_s=packet.mono_ns / 1_000_000_000.0,
            timestamp_wall_s=now_wall_s,
            frame_id=self._frame_id,
            fps=self._fps,
            latency_ms=frame_age_ms,
            frame_age_ms=frame_age_ms,
            source_alive=self._last_source_error is None,
            is_new_frame=is_new,
            debug=debug,
        )

    def shutdown(self) -> None:
        try:
            self.source.shutdown()
        except Exception as exc:  # noqa: BLE001 - best effort on teardown
            logger.warning("frame source shutdown failed: %r", exc)

    def latest_packet(self) -> Optional[FramePacket]:
        return self._last_packet

    def _acquire_packet(self) -> tuple[Optional[FramePacket], bool]:
        packet = None
        try:
            if hasattr(self.source, "get_latest"):
                packet = self.source.get_latest(timeout_s=0.01)
            if packet is None and hasattr(self.source, "get_packet"):
                packet = self.source.get_packet()
        except Exception as exc:  # noqa: BLE001 - source health is part of the state
            self._last_source_error = repr(exc)
            now = time.monotonic()
            if (now - self._last_source_warn_s) >= 2.0:
                logger.warning("frame source read failed: %s", self._last_source_error)
                self._last_source_warn_s = now
            if self._last_packet is None:
                return None, False
            return self._last_packet, False

        if packet is None:
            if self._last_packet is None:
                return None, False
            return self._last_packet, False

        self._last_source_error = None
        self._last_packet = packet
        return packet, True


def _fps_from_window(times_ns: deque[int]) -> Optional[float]:
    if len(times_ns) < 2:
        return None
    dt_ns = times_ns[-1] - times_ns[0]
    if dt_ns <= 0:
        return None
    return (len(times_ns) - 1) / (dt_ns / 1_000_000_000.0)
