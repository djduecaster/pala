from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional


@dataclass
class MemoryManagerConfig:
    enabled: bool = False
    jsonl_path: str = "logs/orchestrator_memory.jsonl"
    recent_events: int = 20
    digest_items: int = 3
    distill_every_n_events: int = 20


class MemoryManager:
    """Small local event memory used by planner/orchestrator paths."""

    def __init__(self, config: Optional[MemoryManagerConfig] = None):
        self._cfg = config or MemoryManagerConfig()
        self._lock = threading.Lock()
        self._recent_events: List[Dict[str, Any]] = []
        self._digests: List[Dict[str, Any]] = []
        self._total_events = 0
        self._load_existing_if_enabled()

    def append_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self._cfg.enabled:
            return
        item = {
            "type": str(event_type),
            "ts_wall_s": time.time(),
            "payload": dict(payload),
        }
        with self._lock:
            self._recent_events.append(item)
            if len(self._recent_events) > max(1, int(self._cfg.recent_events)):
                self._recent_events = self._recent_events[-int(self._cfg.recent_events) :]
            self._total_events += 1
            self._append_jsonl_locked(item)

            n = max(0, int(self._cfg.distill_every_n_events))
            if n > 0 and (self._total_events % n) == 0:
                digest = self._distill_locked()
                if digest is not None:
                    self._digests.append(digest)
                    max_items = max(0, int(self._cfg.digest_items))
                    if max_items > 0 and len(self._digests) > max_items:
                        self._digests = self._digests[-max_items:]
                    self._append_jsonl_locked(digest)

    def context(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "recent_events": list(self._recent_events),
                "session_memory_digest": list(self._digests),
            }

    def _distill_locked(self) -> Optional[Dict[str, Any]]:
        if not self._recent_events:
            return None
        window = self._recent_events[-min(10, len(self._recent_events)) :]
        states = [str(evt.get("payload", {}).get("state")) for evt in window if evt.get("payload")]
        zones = [str(evt.get("payload", {}).get("zone_hint")) for evt in window if evt.get("payload")]
        primitives = [str(evt.get("payload", {}).get("primitive")) for evt in window if evt.get("payload")]
        payload = {
            "window_events": len(window),
            "highlights": [
                f"dominant_state={_dominant(states)}",
                f"dominant_zone={_dominant(zones)}",
                f"dominant_primitive={_dominant(primitives)}",
            ],
        }
        return {"type": "summary_event", "ts_wall_s": time.time(), "payload": payload}

    def _load_existing_if_enabled(self) -> None:
        if not self._cfg.enabled:
            return
        path = Path(self._cfg.jsonl_path)
        if not path.exists():
            return
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            self._total_events += 1
            self._recent_events.append(item)
        if self._recent_events:
            self._recent_events = self._recent_events[-max(1, int(self._cfg.recent_events)) :]

    def _append_jsonl_locked(self, item: Dict[str, Any]) -> None:
        path = Path(self._cfg.jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


def _dominant(values: List[str]) -> str:
    counts: Dict[str, int] = {}
    for value in values:
        token = value.strip().lower() if isinstance(value, str) else ""
        if not token or token in {"none", "null"}:
            continue
        counts[token] = counts.get(token, 0) + 1
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda item: item[1])[0]

