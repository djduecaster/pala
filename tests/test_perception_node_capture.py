import time

import numpy as np

from pala.perception.frame_source import FramePacket
from pala.perception.node import PerceptionNode


class _NoFrameSource:
    def get_latest(self, timeout_s: float = 0.01):
        time.sleep(min(0.001, timeout_s))
        return None

    def shutdown(self) -> None:
        return None


class _OneFrameSource:
    def __init__(self):
        self._sent = False

    def get_latest(self, timeout_s: float = 0.01):
        if self._sent:
            time.sleep(min(0.001, timeout_s))
            return None
        self._sent = True
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        return FramePacket(frame=frame, pts_ns=123, mono_ns=time.monotonic_ns())

    def shutdown(self) -> None:
        return None


class _TwoFrameSource:
    def __init__(self):
        self._calls = 0
        self._base = time.monotonic_ns()
        self._delta = 50_000_000
        self._frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def get_latest(self, timeout_s: float = 0.01):
        if self._calls == 0:
            self._calls += 1
            return FramePacket(frame=self._frame, pts_ns=1, mono_ns=self._base)
        if self._calls == 1:
            self._calls += 1
            return FramePacket(frame=self._frame, pts_ns=2, mono_ns=self._base + self._delta)
        time.sleep(min(0.001, timeout_s))
        return None

    def shutdown(self) -> None:
        return None


def test_perception_no_frame_reports_capture_state():
    node = PerceptionNode(source=_NoFrameSource())
    state = node.step()

    assert state.frame_id is None
    assert state.is_new_frame is False
    assert state.source_alive is True
    assert state.debug == {"no_frame": True}


def test_perception_reuses_last_frame_as_stale_state():
    node = PerceptionNode(source=_OneFrameSource())
    first = node.step()
    second = node.step()

    assert first.frame_id == 1
    assert first.is_new_frame is True
    assert first.debug == {}
    assert second.frame_id == 1
    assert second.is_new_frame is False
    assert second.debug.get("stale_frame") is True
    assert second.frame_age_ms is not None


def test_perception_fps_and_frame_ids_advance_on_new_frames():
    node = PerceptionNode(source=_TwoFrameSource())
    first = node.step()
    second = node.step()

    assert first.frame_id == 1
    assert second.frame_id == 2
    assert second.fps is not None
    assert abs(second.fps - 20.0) < 1.0


def test_perception_source_error_is_reported_without_crashing():
    class _ErrSource:
        def get_packet(self):
            raise RuntimeError("camera unavailable")

        def shutdown(self) -> None:
            return None

    state = PerceptionNode(source=_ErrSource()).step()

    assert state.frame_id is None
    assert state.source_alive is False
    assert state.debug.get("no_frame") is True
    assert "camera unavailable" in str(state.debug.get("source_error"))


def test_perception_shutdown_closes_source():
    calls = {"source": 0}

    class _Source:
        def shutdown(self) -> None:
            calls["source"] += 1

    PerceptionNode(source=_Source()).shutdown()
    assert calls == {"source": 1}
