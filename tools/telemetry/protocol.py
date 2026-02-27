from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

PROTOCOL_NAME = "pala.telemetry"
PROTOCOL_VERSION = 1


def event(
    *,
    source: str,
    payload: Dict[str, Any],
    level: str = "info",
    msg_type: str = "event",
    ts_mono_s: Optional[float] = None,
    ts_wall_s: Optional[float] = None,
    seq: Optional[int] = None,
) -> Dict[str, Any]:
    msg: Dict[str, Any] = {
        "proto": PROTOCOL_NAME,
        "proto_version": PROTOCOL_VERSION,
        "type": msg_type,
        "source": source,
        "level": level,
        "ts_mono_s": time.monotonic() if ts_mono_s is None else float(ts_mono_s),
        "ts_wall_s": time.time() if ts_wall_s is None else float(ts_wall_s),
        "payload": payload,
    }
    if seq is not None:
        msg["seq"] = int(seq)
    return msg


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
    proto = decoded.get("proto")
    if proto is not None:
        if proto != PROTOCOL_NAME:
            return None
        try:
            version = int(decoded.get("proto_version", -1))
        except (TypeError, ValueError):
            return None
        if version != PROTOCOL_VERSION:
            return None
    else:
        # Backward-compatibility for older untagged telemetry events.
        if not {"type", "source", "payload"}.issubset(decoded):
            return None

    msg_type = decoded.get("type")
    source = decoded.get("source")
    payload = decoded.get("payload")
    if not isinstance(msg_type, str):
        return None
    if not isinstance(source, str):
        return None
    if not isinstance(payload, dict):
        return None
    return decoded
