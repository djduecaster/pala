from __future__ import annotations

import time


class RateLimiter:
    def __init__(self, hz: float):
        self._period = 1.0 / max(1e-6, hz)
        self._next = time.monotonic()

    def sleep(self) -> None:
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + self._period, time.monotonic())
