"""Frame sources.

TODO: Port legacy capture pipeline from ../pala_old/pala_project/src/vision/capture.py
      into pala/perception/frame_source_gst.py
"""
from __future__ import annotations
from typing import Optional
import math
import time

from ..hardware.camera import CameraInterface

class FrameSource:
    def get_timestamp(self) -> float:
        """Return a monotonic timestamp for the latest frame."""
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class DummyFrameSource(FrameSource):
    def __init__(self, period_s: float = 6.0):
        self._t0 = time.monotonic()
        self._period = max(0.1, float(period_s))

    def get_timestamp(self) -> float:
        return time.monotonic()

    def dummy_position(self) -> float:
        # Oscillate between 0.2 and 0.8
        t = time.monotonic() - self._t0
        phase = (t / self._period) * 2.0 * math.pi
        return 0.5 + 0.3 * math.sin(phase)

    def shutdown(self) -> None:
        pass


class CameraFrameSource(FrameSource):
    def __init__(self, camera: CameraInterface):
        self._camera = camera
        self.last_frame = None
        self.last_pts_ns = None
        self.last_mono_ns = None

    def get_timestamp(self) -> float:
        frame, pts_ns, mono_ns = self._camera.get_frame()
        self.last_frame = frame
        self.last_pts_ns = pts_ns
        self.last_mono_ns = mono_ns
        return mono_ns / 1_000_000_000.0

    def shutdown(self) -> None:
        self._camera.shutdown()
