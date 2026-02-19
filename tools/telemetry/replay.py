from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Iterator, Optional, Tuple


class SessionReplayReader:
    def __init__(self, session_dir: str) -> None:
        self._session_dir = str(session_dir)
        self._session_root = os.path.realpath(self._session_dir)
        self._events_path = os.path.join(self._session_dir, "events.jsonl")
        self._manifest_path = os.path.join(self._session_dir, "manifest.json")
        self._manifest: Optional[Dict[str, Any]] = None
        if os.path.exists(self._manifest_path):
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    self._manifest = loaded
            except Exception:
                self._manifest = None

    @property
    def manifest(self) -> Optional[Dict[str, Any]]:
        return self._manifest

    def _hydrate_frame(self, payload: Dict[str, Any]) -> None:
        if "bytes_b64" in payload:
            return
        frame_ref = payload.get("frame_ref")
        if not isinstance(frame_ref, str) or not frame_ref:
            return
        abs_path = os.path.realpath(os.path.join(self._session_dir, frame_ref))
        try:
            if os.path.commonpath([self._session_root, abs_path]) != self._session_root:
                return
        except ValueError:
            return
        if not os.path.exists(abs_path):
            return
        try:
            with open(abs_path, "rb") as fh:
                data = fh.read()
        except OSError:
            return
        payload["bytes_b64"] = base64.b64encode(data).decode("ascii")

    def iter_events(self) -> Iterator[Tuple[Dict[str, Any], Optional[float]]]:
        if not os.path.exists(self._events_path):
            raise FileNotFoundError(f"capture events file missing: {self._events_path}")

        prev_ts: Optional[float] = None
        with open(self._events_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                payload = msg.get("payload")
                if isinstance(payload, dict) and msg.get("source") == "video_frame":
                    self._hydrate_frame(payload)

                ts_wall_s = msg.get("ts_wall_s")
                delay_s: Optional[float] = None
                if isinstance(ts_wall_s, (int, float)):
                    ts = float(ts_wall_s)
                    if prev_ts is not None:
                        delay_s = max(0.0, ts - prev_ts)
                    prev_ts = ts
                yield msg, delay_s
