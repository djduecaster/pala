from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FrameItem:
    mono_ns: int
    frame: np.ndarray


class RollingFrameWindow:
    """Latest-only rolling frame history with bounded time window."""

    def __init__(self, *, window_s: float):
        self._window_ns = int(max(0.1, float(window_s)) * 1_000_000_000.0)
        self._items: Deque[FrameItem] = deque()
        self._last_seen_mono_ns: Optional[int] = None

    def add_frame(self, frame: np.ndarray, *, mono_ns: int) -> bool:
        mono_ns = int(mono_ns)
        if self._last_seen_mono_ns is not None and mono_ns == self._last_seen_mono_ns:
            self.prune()
            return False
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return False
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        self._items.append(FrameItem(mono_ns=mono_ns, frame=np.array(arr, copy=True)))
        self._last_seen_mono_ns = mono_ns
        self.prune(now_ns=mono_ns)
        return True

    def prune(self, *, now_ns: Optional[int] = None) -> None:
        if now_ns is None:
            now_ns = time.monotonic_ns()
        cutoff = int(now_ns) - self._window_ns
        while self._items and self._items[0].mono_ns < cutoff:
            self._items.popleft()

    def sample(self, max_frames: int) -> List[FrameItem]:
        self.prune()
        n = len(self._items)
        if n == 0:
            return []
        max_frames = max(1, int(max_frames))
        if n <= max_frames:
            return list(self._items)
        indices = _even_sample_indices(n, max_frames)
        seq: Sequence[FrameItem] = list(self._items)
        return [seq[i] for i in indices]

    def latest(self) -> Optional[FrameItem]:
        self.prune()
        if not self._items:
            return None
        return self._items[-1]


def _even_sample_indices(n: int, k: int) -> List[int]:
    if k <= 1:
        return [n - 1]
    if n <= k:
        return list(range(n))
    step = float(n - 1) / float(k - 1)
    out: List[int] = []
    for i in range(k):
        idx = int(round(i * step))
        if idx < 0:
            idx = 0
        if idx > (n - 1):
            idx = n - 1
        out.append(idx)
    out[-1] = n - 1
    deduped: List[int] = []
    for idx in out:
        if deduped and idx <= deduped[-1]:
            continue
        deduped.append(idx)
    if deduped[-1] != (n - 1):
        deduped[-1] = n - 1
    while len(deduped) < k:
        missing = [i for i in range(n) if i not in set(deduped)]
        if not missing:
            break
        deduped.insert(-1, missing[0])
    return deduped[:k]

