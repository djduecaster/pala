from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import logging
import os
import threading
import time
from typing import Any, Deque, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryManagerConfig:
    enabled: bool = True
    jsonl_path: str = "logs/orchestrator_memory.jsonl"
    recent_events: int = 200
    # Kept for compatibility with existing config; not used by current implementation.
    digest_items: int = 0
    distill_every_n_events: int = 0


class MemoryManager:
    """Canonical append-only memory stream with bounded in-memory cache."""

    def __init__(self, cfg: MemoryManagerConfig):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._recent: Deque[dict[str, Any]] = deque(maxlen=max(16, int(cfg.recent_events)))
        self._digests: Deque[dict[str, Any]] = deque(maxlen=max(1, int(cfg.digest_items or 0)))
        self._events_since_distill = 0
        if self._cfg.enabled:
            self._load_existing()

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        item = {
            "type": str(event_type),
            "ts_wall_s": time.time(),
            "payload": payload if isinstance(payload, dict) else {},
        }
        new_digest: Optional[dict[str, Any]] = None
        with self._lock:
            self._recent.append(item)
            self._events_since_distill += 1
            if self._cfg.distill_every_n_events > 0 and self._events_since_distill >= self._cfg.distill_every_n_events:
                digest = _distill(self._recent)
                self._events_since_distill = 0
                if digest is not None:
                    self._digests.append(digest)
                    new_digest = digest
        if self._cfg.enabled:
            self._append_jsonl(item)
            if new_digest is not None:
                self._append_jsonl(
                    {
                        "type": "summary_event",
                        "ts_wall_s": time.time(),
                        "payload": new_digest,
                    }
                )

    def recent_events(self, *, event_type: Optional[str] = None, limit: int = 10) -> list[dict[str, Any]]:
        n = max(0, int(limit))
        if n == 0:
            return []
        with self._lock:
            rows = list(self._recent)
        if event_type is not None:
            rows = [row for row in rows if row.get("type") == event_type]
        return rows[-n:]

    def recent_payloads(self, event_type: str, limit: int) -> list[dict[str, Any]]:
        rows = self.recent_events(event_type=event_type, limit=limit)
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = row.get("payload")
            if isinstance(payload, dict):
                out.append(dict(payload))
        return out

    def recent_lines(
        self,
        *,
        event_types: Iterable[str],
        limit: int,
        max_chars: int,
    ) -> list[str]:
        allowed = {str(t) for t in event_types}
        n = max(0, int(limit))
        if n == 0 or max_chars <= 0:
            return []
        with self._lock:
            rows = [row for row in self._recent if str(row.get("type")) in allowed]
        rows = rows[-n:]
        lines: list[str] = []
        for row in rows:
            ts = row.get("ts_wall_s")
            ts_str = time.strftime("%H:%M:%S", time.localtime(ts)) if isinstance(ts, (int, float)) else "00:00:00"
            event_type = str(row.get("type", "event"))
            payload = row.get("payload")
            if isinstance(payload, dict):
                compact = _compact_payload(payload)
            else:
                compact = "{}"
            lines.append(f"{ts_str} {event_type}: {compact}")
        while lines and sum(len(line) + 1 for line in lines) > max_chars:
            lines.pop(0)
        return lines

    def context(self) -> dict[str, Any]:
        with self._lock:
            digests = list(self._digests)
        return {
            "recent_events": self.recent_events(limit=min(50, len(self._recent))),
            "session_memory_digest": digests,
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
                    if not isinstance(item.get("payload"), dict):
                        item["payload"] = {}
                    if item.get("type") == "summary_event":
                        self._digests.append(item.get("payload", {}))
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


def _compact_payload(payload: dict[str, Any]) -> str:
    preview: dict[str, Any] = {}
    for key in (
        "scene_state",
        "state",
        "intent",
        "primitive",
        "primitive_hint",
        "target_zone",
        "zone_hint",
        "confidence",
        "source",
    ):
        if key in payload:
            preview[key] = payload[key]
    if not preview:
        preview = payload
    try:
        return json.dumps(preview, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return "{}"


def _distill(events: Iterable[dict[str, Any]]) -> Optional[dict[str, Any]]:
    state_counts: dict[str, int] = {}
    zone_counts: dict[str, int] = {}
    primitive_counts: dict[str, int] = {}
    count = 0
    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        count += 1
        state = payload.get("state")
        if isinstance(state, str) and state:
            state_counts[state] = state_counts.get(state, 0) + 1
        zone = payload.get("zone_hint")
        if isinstance(zone, str) and zone:
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        primitive = payload.get("primitive")
        if isinstance(primitive, str) and primitive:
            primitive_counts[primitive] = primitive_counts.get(primitive, 0) + 1
    if count == 0:
        return None

    def _top(counter: dict[str, int]) -> Optional[str]:
        if not counter:
            return None
        return max(counter.items(), key=lambda kv: kv[1])[0]

    highlights: list[str] = []
    top_state = _top(state_counts)
    top_zone = _top(zone_counts)
    top_primitive = _top(primitive_counts)
    if top_state is not None:
        highlights.append(f"dominant_state={top_state}")
    if top_zone is not None:
        highlights.append(f"dominant_zone={top_zone}")
    if top_primitive is not None:
        highlights.append(f"dominant_primitive={top_primitive}")
    if not highlights:
        highlights.append("limited_signal")
    return {
        "window_events": count,
        "highlights": highlights,
    }
