from __future__ import annotations

import time

import numpy as np
import pytest

from pala.perception.frame_source import (
    CameraFrameSource,
    DummyFrameSource,
    FramePacket,
    FrameSource,
    ThreadedFrameSource,
)


class _SlowSource:
    def __init__(self, delay_s: float = 0.05):
        self.delay_s = delay_s
        self._stop = False

    def get_packet(self):
        time.sleep(self.delay_s)
        return FramePacket(frame=np.zeros((2, 2, 3), dtype=np.uint8), pts_ns=7, mono_ns=time.monotonic_ns())

    def shutdown(self) -> None:
        self._stop = True


class _ErrorSource:
    def get_packet(self):
        raise RuntimeError("camera blew up")

    def shutdown(self) -> None:
        return None


class _StubCamera:
    def __init__(self):
        self.shutdown_called = False

    def get_frame(self):
        frame = np.zeros((3, 4, 3), dtype=np.uint8)
        return frame, 1234, 2_500_000_000

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_frame_source_base_methods_raise_not_implemented():
    source = FrameSource()
    with pytest.raises(NotImplementedError):
        source.get_timestamp()
    with pytest.raises(NotImplementedError):
        source.shutdown()


def test_dummy_and_camera_frame_source_timestamp_paths():
    dummy = DummyFrameSource()
    assert isinstance(dummy.get_timestamp(), float)
    dummy.shutdown()

    camera = _StubCamera()
    source = CameraFrameSource(camera)
    pkt = source.get_packet()
    assert pkt.pts_ns == 1234
    assert source.last_frame is pkt.frame
    assert source.last_pts_ns == 1234
    assert source.last_mono_ns == 2_500_000_000
    assert source.get_timestamp() == 2.5
    source.shutdown()
    assert camera.shutdown_called is True


def test_threaded_frame_source_timeout_and_error_paths():
    class _NeverCapturingThreaded(ThreadedFrameSource):
        def _run(self) -> None:  # override to avoid background captures entirely
            self._stop.wait()

    slow = _NeverCapturingThreaded(_SlowSource(delay_s=0.05))
    try:
        assert slow.get_latest(timeout_s=0.01) is None
    finally:
        slow.shutdown()

    errored = ThreadedFrameSource(_ErrorSource())
    try:
        assert errored.get_latest(timeout_s=0.05) is None
        stats = errored.stats()
        assert stats["captured_count"] == 0
        assert isinstance(stats["last_error"], str)
        assert "RuntimeError" in stats["last_error"]
    finally:
        errored.shutdown()


def test_threaded_frame_source_stop_without_unread_hits_none_branch():
    source = ThreadedFrameSource(_SlowSource(delay_s=0.2))
    try:
        source._stop.set()  # noqa: SLF001
        assert source.get_latest(timeout_s=0.05) is None
    finally:
        source.shutdown()


def test_threaded_frame_source_shutdown_warns_if_thread_lingers(monkeypatch):
    source = ThreadedFrameSource(_SlowSource(delay_s=0.2))
    real_thread = source._thread  # noqa: SLF001

    class _FakeThread:
        def __init__(self):
            self.join_called = False

        def join(self, timeout=0.0):
            self.join_called = True

        def is_alive(self):
            return True

    fake = _FakeThread()
    source._thread = fake  # noqa: SLF001
    warned = []
    monkeypatch.setattr("pala.perception.frame_source.logger.warning", lambda msg: warned.append(msg))

    source.shutdown()
    assert fake.join_called is True
    assert warned == ["ThreadedFrameSource thread did not exit"]

    real_thread.join(timeout=0.5)
