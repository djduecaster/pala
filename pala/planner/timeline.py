from __future__ import annotations

from dataclasses import dataclass
import json
import os
import threading
import time
from typing import Any


@dataclass
class TimelineConfig:
    enabled: bool = True
    jsonl_path: str = "logs/orchestrator_timeline.jsonl"


class TimelineWriter:
    def __init__(self, cfg: TimelineConfig):
        self._cfg = cfg
        self._lock = threading.Lock()

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self._cfg.enabled or not self._cfg.jsonl_path:
            return
        item = {
            "type": str(event_type),
            "ts_wall_s": time.time(),
            "ts_mono_s": time.monotonic(),
            "payload": payload,
        }
        directory = os.path.dirname(self._cfg.jsonl_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            with self._lock:
                with open(self._cfg.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, separators=(",", ":"), ensure_ascii=True))
                    f.write("\n")
        except OSError:
            # Timeline logging is best-effort and must not break control flow.
            return
