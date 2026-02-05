from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Protocol, Tuple

import numpy as np


@dataclass
class Detection:
    bbox_xyxy_px: Tuple[float, float, float, float]
    conf: float
    cls: Optional[int] = None


class DetectorInterface(Protocol):
    def detect(self, frame: np.ndarray) -> List[Detection]:
        ...
