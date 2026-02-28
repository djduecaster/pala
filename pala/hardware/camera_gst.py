"""GStreamer camera backend for Jetson."""
from __future__ import annotations

from typing import Optional, Tuple
import time

import numpy as np


class GStreamerCamera:
    def __init__(
        self,
        *,
        device: str,
        width: int,
        height: int,
        fps: int,
        pipeline: Optional[str] = None,
    ) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        self._gst = Gst
        self._device = device
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)

        if pipeline is None:
            pipeline = (
                f"v4l2src device={self._device} do-timestamp=true ! "
                f"video/x-raw,format=YUY2,width={self._width},height={self._height},framerate={self._fps}/1 ! "
                "videoconvert ! video/x-raw,format=RGB ! "
                "appsink name=appsink emit-signals=true max-buffers=2 drop=true sync=false"
            )
        elif "v4l2src" not in pipeline:
            pipeline = f"v4l2src device={self._device} do-timestamp=true ! {pipeline}"

        Gst.init(None)
        self._pipeline = Gst.parse_launch(pipeline)
        self._pipeline.set_state(Gst.State.PLAYING)
        self._appsink = self._pipeline.get_by_name("appsink")
        if self._appsink is None:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer pipeline missing appsink element named 'appsink'")

    def get_frame(self) -> Tuple[np.ndarray, Optional[int], int]:
        sample = self._appsink.emit("pull-sample")
        if not sample:
            raise RuntimeError("Failed to capture frame")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        h = caps.get_structure(0).get_value("height")
        w = caps.get_structure(0).get_value("width")

        ok, mapinfo = buf.map(self._gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("Buffer map failed")
        try:
            arr = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h, w, 3).copy()
        finally:
            buf.unmap(mapinfo)

        pts_ns = int(buf.pts) if buf.pts != self._gst.CLOCK_TIME_NONE else None
        mono_ns = time.monotonic_ns()
        return arr, pts_ns, mono_ns

    def shutdown(self) -> None:
        self._pipeline.set_state(self._gst.State.NULL)
