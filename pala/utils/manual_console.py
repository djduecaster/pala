"""Nonblocking line input polled by the runtime supervisor (no extra loop thread)."""
from __future__ import annotations

import os
import select
from typing import TextIO


class ManualConsole:
    def __init__(self, stream: TextIO) -> None:
        self.fd = stream.fileno()
        self.buffer = b""
        self.eof = False

    def poll(self) -> list[str]:
        if self.eof or not select.select([self.fd], [], [], 0)[0]:
            return []
        chunk = os.read(self.fd, 4096)
        if not chunk:
            self.eof = True
            lines = [self.buffer.decode().strip()] if self.buffer else []
            self.buffer = b""
            return lines + ["shutdown"]
        self.buffer += chunk
        if len(self.buffer) > 8192:
            raise ValueError("Manual command exceeds input limit")
        pieces = self.buffer.split(b"\n")
        self.buffer = pieces.pop()
        return [line.decode().strip() for line in pieces]
