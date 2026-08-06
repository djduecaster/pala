"""Frame sources.

TODO: Port legacy capture pipeline from ../pala_old/pala_project/src/vision/capture.py
      into pala/perception/frame_source_gst.py
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import threading
import time

import numpy as np
import logging

from ..hardware.camera import CameraInterface

logger = logging.getLogger(__name__)
class FrameSource:
    def get_timestamp(self) -> float:
        """Return a monotonic timestamp for the latest frame."""
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class DummyFrameSource(FrameSource):
    def get_packet(self) -> FramePacket:
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        return FramePacket(frame=frame, pts_ns=None, mono_ns=time.monotonic_ns())

    def get_timestamp(self) -> float:
        return time.monotonic()

    def shutdown(self) -> None:
        pass


@dataclass
class FramePacket:
    frame: object
    pts_ns: Optional[int]
    mono_ns: int


class CameraFrameSource(FrameSource):
    def __init__(self, camera: CameraInterface):
        self._camera = camera
        self.last_frame = None
        self.last_pts_ns = None
        self.last_mono_ns = None

    def get_packet(self) -> FramePacket:
        frame, pts_ns, mono_ns = self._camera.get_frame()
        packet = FramePacket(frame=frame, pts_ns=pts_ns, mono_ns=mono_ns)
        self.last_frame = frame
        self.last_pts_ns = pts_ns
        self.last_mono_ns = mono_ns
        return packet

    def get_timestamp(self) -> float:
        packet = self.get_packet()
        return packet.mono_ns / 1_000_000_000.0

    def shutdown(self) -> None:
        self._camera.shutdown()


class ThreadedFrameSource:
    """Capture frames in a background thread and keep only the latest."""

    def __init__(self, inner: CameraFrameSource, min_interval_s: float = 0.0):
        self._inner = inner
        self._min_interval_s = max(0.0, float(min_interval_s))
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._latest: Optional[FramePacket] = None
        self._has_unread = False
        self.captured_count = 0
        self.dropped_count = 0
        self.last_capture_mono_ns: Optional[int] = None
        self.last_pts_ns: Optional[int] = None
        self._last_error: Optional[str] = None

        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                packet = self._inner.get_packet()
            except Exception as exc:
                with self._lock:
                    self._last_error = repr(exc)
                    self._stop.set()
                    self._cond.notify_all()
                return

            with self._lock:
                if self._has_unread:
                    self.dropped_count += 1
                self._latest = packet
                self._has_unread = True
                self.captured_count += 1
                self.last_capture_mono_ns = packet.mono_ns
                self.last_pts_ns = packet.pts_ns
                self._cond.notify_all()

            if self._min_interval_s > 0:
                time.sleep(self._min_interval_s)

    def get_latest(self, timeout_s: Optional[float] = None) -> Optional[FramePacket]:
        with self._lock:
            if timeout_s is not None and timeout_s <= 0:
                return None if not self._has_unread else self._consume_latest()

            end = None if timeout_s is None else time.monotonic() + timeout_s
            while not self._stop.is_set() and self._last_error is None and self._thread.is_alive() and not self._has_unread:
                if end is None:
                    self._cond.wait()
                    continue
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)

            if self._last_error is not None or not self._thread.is_alive():
                return None
            if not self._has_unread:
                return None
            return self._consume_latest()

    def peek_latest(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest

    def _consume_latest(self) -> Optional[FramePacket]:
        packet = self._latest
        self._has_unread = False
        return packet

    def stats(self) -> dict:
        with self._lock:
            return {
                "captured_count": self.captured_count,
                "dropped_count": self.dropped_count,
                "last_capture_mono_ns": self.last_capture_mono_ns,
                "last_pts_ns": self.last_pts_ns,
                "last_error": self._last_error,
            }

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._cond.notify_all()
        self._inner.shutdown()
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            logger.warning("ThreadedFrameSource thread did not exit")
