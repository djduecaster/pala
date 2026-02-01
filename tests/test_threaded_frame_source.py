import time

from pala.hardware.camera import DummyCamera
from pala.perception.frame_source import CameraFrameSource, ThreadedFrameSource, FramePacket


def test_threaded_frame_source_captures_latest():
    camera = DummyCamera(width=64, height=48)
    source = ThreadedFrameSource(CameraFrameSource(camera), min_interval_s=0.001)

    try:
        time.sleep(0.01)
        packet = source.get_latest()
        assert packet is not None
        assert packet.frame is not None
        assert source.captured_count > 0
        assert source.dropped_count >= 0
    finally:
        source.shutdown()


def test_threaded_frame_source_peek_does_not_consume():
    camera = DummyCamera(width=8, height=8)
    source = ThreadedFrameSource(CameraFrameSource(camera), min_interval_s=0.001)

    try:
        time.sleep(0.01)
        peeked = source.peek_latest()
        assert peeked is not None

        consumed = source.get_latest(timeout_s=0.01)
        assert consumed is not None

        empty = source.get_latest(timeout_s=0.0)
        assert empty is None
    finally:
        source.shutdown()


class _BlockingSource:
    def __init__(self):
        self._stop = False

    def get_packet(self):
        while not self._stop:
            time.sleep(0.01)
        return FramePacket(frame=None, pts_ns=None, mono_ns=time.monotonic_ns())

    def shutdown(self) -> None:
        self._stop = True


def test_threaded_frame_source_shutdown_returns_quickly():
    source = ThreadedFrameSource(_BlockingSource())
    start = time.monotonic()
    source.shutdown()
    elapsed = time.monotonic() - start
    assert elapsed < 0.5
