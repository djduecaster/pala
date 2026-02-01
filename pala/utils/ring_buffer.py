from __future__ import annotations

from typing import Generic, Optional, TypeVar
import threading

T = TypeVar("T")


class LatestValue(Generic[T]):
    def __init__(self):
        self._lock = threading.Lock()
        self._value: Optional[T] = None
        self._ts: Optional[float] = None

    def set(self, value: T, ts: float) -> None:
        with self._lock:
            self._value = value
            self._ts = ts

    def get(self) -> tuple[Optional[T], Optional[float]]:
        with self._lock:
            return self._value, self._ts
