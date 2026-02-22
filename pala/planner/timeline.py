from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional


@dataclass
class TimelineConfig:
    enabled: bool = False
    jsonl_path: str = "logs/orchestrator_timeline.jsonl"


class TimelineWriter:
    def __init__(self, config: Optional[TimelineConfig] = None):
        self._cfg = config or TimelineConfig()

    def append(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self._cfg.enabled:
            return
        path = Path(self._cfg.jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        item = {"type": str(event_type), "ts_wall_s": time.time(), "payload": dict(payload)}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

