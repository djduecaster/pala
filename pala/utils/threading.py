from __future__ import annotations

import threading


def stop_event() -> threading.Event:
    return threading.Event()
