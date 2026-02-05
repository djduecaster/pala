from __future__ import annotations

from typing import List

import numpy as np

from .interface import Detection, DetectorInterface


class JetsonDetector(DetectorInterface):
    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError("JetsonDetector backend not implemented yet")
