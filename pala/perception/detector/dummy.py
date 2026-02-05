from __future__ import annotations

import math
import time
from typing import List

import numpy as np

from .interface import Detection, DetectorInterface


class DummyDetector(DetectorInterface):
    def __init__(self, period_s: float = 6.0):
        self._t0 = time.monotonic()
        self._period = max(0.1, float(period_s))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        t = time.monotonic() - self._t0
        phase = (t / self._period) * 2.0 * math.pi
        cx = 0.5 + 0.3 * math.sin(phase)
        cy = 0.5
        bw = 0.2
        bh = 0.4
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        return [Detection(bbox_xyxy_px=(x1, y1, x2, y2), conf=0.8, cls=0)]
