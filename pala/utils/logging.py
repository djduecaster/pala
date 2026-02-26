from __future__ import annotations

import re
from typing import Optional
import os
import threading

from ..types import to_json_line


_DATA_IMAGE_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


def _redact_data_urls(value):
    if isinstance(value, str):
        if _DATA_IMAGE_URL_RE.match(value.strip()):
            return f"<image_data_url chars={len(value)}>"
        return value
    if isinstance(value, list):
        return [_redact_data_urls(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_data_urls(item) for key, item in value.items()}
    return value


class JsonlLogger:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, obj) -> None:
        line = to_json_line(_redact_data_urls(obj))
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
