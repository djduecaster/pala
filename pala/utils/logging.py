from __future__ import annotations

from typing import Optional
import os
import threading

from ..types import to_json_line


class JsonlLogger:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, obj) -> None:
        line = to_json_line(obj)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def maybe_logger(path: Optional[str]) -> Optional[JsonlLogger]:
    if not path:
        return None
    return JsonlLogger(path)
