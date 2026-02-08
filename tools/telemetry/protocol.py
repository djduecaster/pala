from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional


def monotonic_s() -> float:
    return time.monotonic()


def wall_s() -> float:
    return time.time()


def event(
    *,
    source: str,
    payload: Dict[str, Any],
    level: str = "info",
    msg_type: str = "event",
    ts_mono_s: Optional[float] = None,
    ts_wall_s: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "type": msg_type,
        "source": source,
        "level": level,
        "ts_mono_s": monotonic_s() if ts_mono_s is None else float(ts_mono_s),
        "ts_wall_s": wall_s() if ts_wall_s is None else float(ts_wall_s),
        "payload": payload,
    }


def encode_message(msg: Dict[str, Any]) -> str:
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=True)


def decode_message(line: str) -> Optional[Dict[str, Any]]:
    text = line.strip()
    if not text:
        return None
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded

