from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _resample_filter() -> int:
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.BILINEAR)
    return int(Image.BILINEAR)


class PreviewTapWriter:
    """Writes a throttled latest-frame JPEG + metadata sidecar for telemetry tools."""

    def __init__(
        self,
        *,
        enabled: bool,
        jpeg_path: str,
        meta_path: str,
        max_hz: float,
        max_width: int,
        max_height: int,
        jpeg_quality: int,
    ) -> None:
        self._enabled = bool(enabled)
        self._jpeg_path = str(jpeg_path)
        self._meta_path = str(meta_path)
        self._max_width = max(16, int(max_width))
        self._max_height = max(16, int(max_height))
        self._jpeg_quality = max(30, min(95, int(jpeg_quality)))
        self._period_ns = int((1.0 / max(0.1, float(max_hz))) * 1_000_000_000.0)
        self._last_emit_ns: Optional[int] = None
        self._frame_id = 0
        self._last_warn_s = 0.0

        if self._enabled:
            for path in (self._jpeg_path, self._meta_path):
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)

    def write(self, frame: object, *, mono_ns: int, pts_ns: Optional[int]) -> None:
        self.write_with_extra(frame, mono_ns=mono_ns, pts_ns=pts_ns, extra=None)

    def write_with_extra(
        self,
        frame: object,
        *,
        mono_ns: int,
        pts_ns: Optional[int],
        extra: Optional[Dict[str, Any]],
    ) -> None:
        if not self._enabled:
            return
        mono_ns = int(mono_ns)
        if self._last_emit_ns is not None and (mono_ns - self._last_emit_ns) < self._period_ns:
            return

        tmp_jpeg = f"{self._jpeg_path}.tmp.{os.getpid()}"
        tmp_meta = f"{self._meta_path}.tmp.{os.getpid()}"
        try:
            arr = np.asarray(frame)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"expected HxWx3 frame, got {arr.shape!r}")
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)

            image = Image.fromarray(arr, mode="RGB")
            if image.width > self._max_width or image.height > self._max_height:
                image.thumbnail((self._max_width, self._max_height), resample=_resample_filter())

            image.save(tmp_jpeg, format="JPEG", quality=self._jpeg_quality, optimize=False)
            payload = {
                "frame_id": self._frame_id,
                "width": int(image.width),
                "height": int(image.height),
                "mono_ns": mono_ns,
                "pts_ns": None if pts_ns is None else int(pts_ns),
                "ts_wall_s": time.time(),
            }
            if isinstance(extra, dict) and extra:
                payload["extra"] = extra
            with open(tmp_meta, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=True, separators=(",", ":"))

            os.replace(tmp_jpeg, self._jpeg_path)
            os.replace(tmp_meta, self._meta_path)
            self._frame_id += 1
            self._last_emit_ns = mono_ns
        except Exception as exc:
            now = time.monotonic()
            # Keep failures visible but avoid flooding logs in the perception loop.
            if (now - self._last_warn_s) >= 5.0:
                logger.warning("telemetry preview tap write failed: %r", exc)
                self._last_warn_s = now
            for path in (tmp_jpeg, tmp_meta):
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except OSError:
                    continue

    def close(self) -> None:
        return None
