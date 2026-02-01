"""Camera interfaces.

TODO: Port legacy GStreamer camera from ../pala_old/pala_project/src/hardware/camera.py
      into pala/hardware/camera_gst.py
"""
from __future__ import annotations
from typing import Tuple, Optional
import time
import numpy as np


class CameraInterface:
    def get_frame(self) -> Tuple[np.ndarray, Optional[int], int]:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class DummyCamera(CameraInterface):
    def __init__(self, width: int = 640, height: int = 480):
        self._width = int(width)
        self._height = int(height)

    def get_frame(self) -> Tuple[np.ndarray, Optional[int], int]:
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        return frame, None, time.monotonic_ns()

    def shutdown(self) -> None:
        return None
