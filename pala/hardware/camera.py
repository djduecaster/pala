"""Camera interfaces.

TODO: Port legacy GStreamer camera from ../pala_old/pala_project/src/hardware/camera.py
      into pala/hardware/camera_gst.py
"""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np


class CameraInterface:
    def get_frame(self) -> Tuple[np.ndarray, Optional[int]]:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError
