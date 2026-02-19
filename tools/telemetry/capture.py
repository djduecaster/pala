from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import os
import socket
import time
from typing import Any, Dict, List, Optional


MANIFEST_SCHEMA_VERSION = 1


@dataclass
class CaptureConfig:
    directory: str
    frames_mode: str = "off"  # off | keyframes | all
    max_seconds: float = 0.0
    manifest_version: int = MANIFEST_SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionCaptureWriter:
    def __init__(self, cfg: CaptureConfig) -> None:
        directory = str(cfg.directory).strip()
        if not directory:
            raise ValueError("capture directory is required")
        self._cfg = cfg
        self._dir = directory
        self._events_path = os.path.join(self._dir, "events.jsonl")
        self._manifest_path = os.path.join(self._dir, "manifest.json")
        self._index_path = os.path.join(self._dir, "index.json")
        self._frames_dir = os.path.join(self._dir, "frames")
        self._events_fh = None
        self._start_wall_s = time.time()
        self._event_count = 0
        self._frame_count = 0
        self._bytes_written = 0
        self._index: List[Dict[str, Any]] = []
        self._closed = False

        os.makedirs(self._dir, exist_ok=True)
        os.makedirs(self._frames_dir, exist_ok=True)
        self._events_fh = open(self._events_path, "w", encoding="utf-8")

    def _should_store_frame(self, frame_id: Optional[int]) -> bool:
        mode = str(self._cfg.frames_mode or "off").strip().lower()
        if mode == "off":
            return False
        if mode == "all":
            return True
        if mode == "keyframes":
            if frame_id is None:
                return self._frame_count == 0
            return frame_id <= 0 or (frame_id % 10) == 0
        return False

    def _store_frame(self, event_idx: int, payload: Dict[str, Any]) -> None:
        frame_id_obj = payload.get("frame_id")
        frame_id = frame_id_obj if isinstance(frame_id_obj, int) else None
        if not self._should_store_frame(frame_id):
            payload.pop("bytes_b64", None)
            return

        b64 = payload.get("bytes_b64")
        if not isinstance(b64, str):
            return
        try:
            data = base64.b64decode(b64, validate=True)
        except Exception:
            return
        file_name = f"frame_{event_idx:08d}_{(frame_id or 0):08d}.jpg"
        abs_path = os.path.join(self._frames_dir, file_name)
        with open(abs_path, "wb") as fh:
            fh.write(data)
        self._bytes_written += len(data)
        payload.pop("bytes_b64", None)
        payload["frame_ref"] = f"frames/{file_name}"
        payload["frame_bytes"] = len(data)
        self._index.append(
            {
                "event_index": event_idx,
                "frame_id": frame_id,
                "frame_ref": payload["frame_ref"],
                "pts_ns": payload.get("pts_ns"),
            }
        )
        self._frame_count += 1

    def write(self, msg: Dict[str, Any]) -> bool:
        if self._closed:
            return False

        max_s = max(0.0, float(self._cfg.max_seconds))
        if max_s > 0.0 and (time.time() - self._start_wall_s) > max_s:
            return False

        line_obj = json.loads(json.dumps(msg, ensure_ascii=True))
        payload = line_obj.get("payload")
        if isinstance(payload, dict) and line_obj.get("source") == "video_frame":
            self._store_frame(self._event_count, payload)

        assert self._events_fh is not None
        encoded = json.dumps(line_obj, separators=(",", ":"), ensure_ascii=True)
        self._events_fh.write(encoded)
        self._events_fh.write("\n")
        self._event_count += 1
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._events_fh is not None:
            self._events_fh.flush()
            self._events_fh.close()
            self._events_fh = None

        with open(self._index_path, "w", encoding="utf-8") as fh:
            json.dump({"frames": self._index}, fh, separators=(",", ":"), ensure_ascii=True)

        manifest = {
            "schema_version": int(self._cfg.manifest_version),
            "created_at_wall_s": self._start_wall_s,
            "closed_at_wall_s": time.time(),
            "host": socket.gethostname(),
            "events_path": "events.jsonl",
            "index_path": "index.json",
            "frames_dir": "frames",
            "event_count": self._event_count,
            "frame_count": self._frame_count,
            "stored_frame_bytes": self._bytes_written,
            "frames_mode": str(self._cfg.frames_mode),
            "metadata": self._cfg.metadata,
        }
        with open(self._manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, separators=(",", ":"), ensure_ascii=True)

