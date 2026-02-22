from __future__ import annotations

from typing import Any, Mapping, Optional


class TraceBus:
    """Best-effort structured trace emitter."""

    def __init__(self, logger_obj: Optional[Any]) -> None:
        self._logger_obj = logger_obj

    def emit(self, payload: Mapping[str, Any]) -> None:
        if self._logger_obj is None:
            return
        try:
            self._logger_obj.write(dict(payload))
        except Exception:  # noqa: BLE001
            return

    def close(self) -> None:
        if self._logger_obj is None:
            return
        try:
            self._logger_obj.close()
        except Exception:  # noqa: BLE001
            return
