from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Any, Deque

logger = logging.getLogger(__name__)


@dataclass
class MemoryManagerConfig:
    enabled: bool = True
    jsonl_path: str = "logs/orchestrator_memory.jsonl"
    recent_events: int = 10
    digest_items: int = 3
    distill_every_n_events: int = 20


class MemoryManager:
    def __init__(self, cfg: MemoryManagerConfig):
        self._cfg = cfg
        self._recent: Deque[dict[str, Any]] = deque(maxlen=max(1, int(cfg.recent_events)))
        self._digests: Deque[dict[str, Any]] = deque(maxlen=max(1, int(cfg.digest_items)))
        self._events_since_distill = 0

        if self._cfg.enabled:
            self._load_existing()

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "type": str(event_type),
            "ts_wall_s": time.time(),
            "payload": payload,
        }
        self._recent.append(event)
        if self._cfg.enabled:
            self._append_jsonl(event)

        self._events_since_distill += 1
        if self._events_since_distill >= max(1, int(self._cfg.distill_every_n_events)):
            digest = self._distill_event_window()
            if digest is not None:
                self._digests.append(digest)
                if self._cfg.enabled:
                    self._append_jsonl({"type": "summary_event", "ts_wall_s": time.time(), "payload": digest})
            self._events_since_distill = 0

    def context(self) -> dict[str, Any]:
        return {
            "recent_events": list(self._recent),
            "session_memory_digest": list(self._digests),
        }

    def stats(self) -> dict[str, int]:
        return {
            "recent_events": len(self._recent),
            "digest_items": len(self._digests),
        }

    def _load_existing(self) -> None:
        path = self._cfg.jsonl_path
        if not path or not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(item, dict):
                        continue
                    kind = item.get("type")
                    if kind == "summary_event":
                        self._digests.append(item.get("payload") if isinstance(item.get("payload"), dict) else {})
                    else:
                        self._recent.append(item)
        except OSError as exc:
            logger.warning("memory manager load failed: %s", exc)

    def _append_jsonl(self, item: dict[str, Any]) -> None:
        path = self._cfg.jsonl_path
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, separators=(",", ":"), ensure_ascii=True))
                f.write("\n")
        except OSError as exc:
            logger.warning("memory manager append failed: %s", exc)

    def _distill_event_window(self) -> dict[str, Any] | None:
        if not self._recent:
            return None

        state_counts: Counter[str] = Counter()
        zone_counts: Counter[str] = Counter()
        primitive_counts: Counter[str] = Counter()
        for event in self._recent:
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            state = payload.get("state")
            if isinstance(state, str) and state:
                state_counts[state] += 1
            zone = payload.get("zone_hint")
            if isinstance(zone, str) and zone:
                zone_counts[zone] += 1
            primitive = payload.get("primitive")
            if isinstance(primitive, str) and primitive:
                primitive_counts[primitive] += 1

        top_state = state_counts.most_common(1)[0][0] if state_counts else None
        top_zone = zone_counts.most_common(1)[0][0] if zone_counts else None
        top_primitive = primitive_counts.most_common(1)[0][0] if primitive_counts else None
        highlights = []
        if top_state is not None:
            highlights.append(f"dominant_state={top_state}")
        if top_zone is not None:
            highlights.append(f"dominant_zone={top_zone}")
        if top_primitive is not None:
            highlights.append(f"dominant_primitive={top_primitive}")
        if not highlights:
            highlights.append("limited_signal")

        return {
            "window_events": len(self._recent),
            "highlights": highlights,
        }
