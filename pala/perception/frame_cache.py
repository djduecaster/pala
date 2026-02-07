from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import threading
import time

import numpy as np


@dataclass(frozen=True)
class FrameSnapshot:
    frame: np.ndarray
    mono_ns: int
    pts_ns: Optional[int]


class LatestFrameCache:
    """Thread-safe latest-frame cache for planner-side consumers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[FrameSnapshot] = None

    def set(self, frame: np.ndarray, *, mono_ns: int, pts_ns: Optional[int]) -> None:
        with self._lock:
            self._latest = FrameSnapshot(frame=frame, mono_ns=int(mono_ns), pts_ns=pts_ns)

    def get(self, *, max_age_ms: Optional[float] = None) -> Optional[FrameSnapshot]:
        with self._lock:
            snap = self._latest
        if snap is None:
            return None
        if max_age_ms is not None:
            age_ms = (time.monotonic_ns() - snap.mono_ns) / 1_000_000.0
            if age_ms > float(max_age_ms):
                return None
        return snap
