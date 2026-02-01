import time
import numpy as np

from pala.perception.node import PerceptionNode
from pala.perception.frame_source import FramePacket


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


def test_perception_no_frame_returns_quickly():
    node = PerceptionNode(source=_NoFrameSource())
    start = time.monotonic()
    st = node.step()
    elapsed = time.monotonic() - start
    assert elapsed < 0.05
    assert st.debug.get("no_frame") is True


def test_perception_stale_frame_fallback():
    node = PerceptionNode(source=_OneFrameSource())
    first = node.step()
    assert first.debug.get("no_frame") is None

    second = node.step()
    assert second.debug.get("stale_frame") is True


class _TwoFrameSource:
    def __init__(self):
        self._calls = 0
        self._base = 1_000_000_000
        self._delta = 50_000_000  # 50ms
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


def test_perception_fps_from_two_frames():
    node = PerceptionNode(source=_TwoFrameSource())
    first = node.step()
    assert first.fps is None
    second = node.step()
    assert second.fps is not None
    assert abs(second.fps - 20.0) < 1.0
