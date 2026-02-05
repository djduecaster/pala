from __future__ import annotations

import time
from collections import deque
from typing import Optional

from ..types import PerceptionState, BBoxNorm, PointNorm
from .frame_source import FrameSource, DummyFrameSource, FramePacket
from .detector import DetectorInterface, DummyDetector, Detection


class PerceptionNode:
    """Dummy perception node. Produces a moving bbox in normalized coords."""

    def __init__(self, source: Optional[FrameSource] = None, detector: Optional[DetectorInterface] = None):
        self.source = source or DummyFrameSource()
        self.detector = detector or DummyDetector()
        self._last_ts = None
        self._fps = None
        self._last_packet: Optional[FramePacket] = None
        self._frame_times = deque(maxlen=30)

    def step(self) -> PerceptionState:
        packet, is_new = self._acquire_packet()
        ts_wall = time.time()

        if packet is None:
            return PerceptionState(
                timestamp_monotonic_s=time.monotonic(),
                timestamp_wall_s=ts_wall,
                fps=self._fps,
                latency_ms=None,
                primary_person=None,
                primary_person_conf=None,
                pointing_target=None,
                pointing_conf=None,
                debug={"no_frame": True},
            )

        ts_mono = packet.mono_ns / 1_000_000_000.0
        if is_new:
            self._frame_times.append(packet.mono_ns)
            self._fps = _fps_from_window(self._frame_times)

        primary_bbox = None
        primary_conf = None
        primary = self._detect_primary(packet)
        if primary is not None:
            primary_bbox, primary_conf = primary

        if primary_bbox is None:
            if isinstance(self.source, DummyFrameSource):
                cx = self.source.dummy_position()
            else:
                cx = 0.5
            primary_bbox = BBoxNorm(cx=cx, cy=0.5, w=0.2, h=0.4)
            primary_conf = 0.5

        # Optional pointing target: when near right, point to top-right
        pointing = None
        pointing_conf = None
        cx = primary_bbox.cx
        if cx > 0.7:
            pointing = PointNorm(x=0.85, y=0.2)
            pointing_conf = 0.6

        debug = {"zone_hint": _zone_from_cx(cx)}
        if not is_new:
            debug["stale_frame"] = True
            debug["frame_age_ms"] = (time.monotonic_ns() - packet.mono_ns) / 1_000_000.0

        return PerceptionState(
            timestamp_monotonic_s=ts_mono,
            timestamp_wall_s=ts_wall,
            fps=self._fps,
            latency_ms=5.0,
            primary_person=primary_bbox,
            primary_person_conf=primary_conf,
            pointing_target=pointing,
            pointing_conf=pointing_conf,
            debug=debug,
        )

    def shutdown(self) -> None:
        self.source.shutdown()

    def _acquire_packet(self) -> tuple[Optional[FramePacket], bool]:
        packet = None
        if hasattr(self.source, "get_latest"):
            packet = self.source.get_latest(timeout_s=0.01)
        if packet is None and hasattr(self.source, "get_packet"):
            packet = self.source.get_packet()

        if packet is None:
            if self._last_packet is None:
                return None, False
            return self._last_packet, False

        self._last_packet = packet
        return packet, True

    def _detect_primary(self, packet: FramePacket) -> Optional[tuple[BBoxNorm, float]]:
        frame = packet.frame
        if frame is None:
            return None
        detections = self.detector.detect(frame)
        if not detections:
            return None
        best = _pick_primary_detection(detections)
        if best is None:
            return None
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return None
        x1, y1, x2, y2 = best.bbox_xyxy_px
        cx = ((x1 + x2) * 0.5) / w
        cy = ((y1 + y2) * 0.5) / h
        bw = max(0.0, (x2 - x1) / w)
        bh = max(0.0, (y2 - y1) / h)
        return BBoxNorm(cx=cx, cy=cy, w=bw, h=bh), float(best.conf)


def _fps_from_window(times_ns: deque) -> Optional[float]:
    if len(times_ns) < 2:
        return None
    dt_ns = times_ns[-1] - times_ns[0]
    if dt_ns <= 0:
        return None
    return (len(times_ns) - 1) / (dt_ns / 1_000_000_000.0)


def _pick_primary_detection(dets: list[Detection]) -> Optional[Detection]:
    best = None
    for det in dets:
        x1, y1, x2, y2 = det.bbox_xyxy_px
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        key = (float(det.conf), area)
        if best is None or key > best[0]:
            best = (key, det)
    return None if best is None else best[1]


def _zone_from_cx(cx: float) -> str:
    if cx < 0.33:
        return "left"
    if cx < 0.66:
        return "center"
    return "right"
