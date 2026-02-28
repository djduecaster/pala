import time
import numpy as np

from pala.perception.node import PerceptionNode
from pala.perception.frame_source import FramePacket
from pala.perception.detector.interface import Detection


class _NoFrameSource:
    def get_latest(self, timeout_s: float = 0.01):
        time.sleep(min(0.001, timeout_s))
        return None

    def shutdown(self) -> None:
        return None


class _OneFrameSource:
    def __init__(self):
        self._sent = False

    def get_latest(self, timeout_s: float = 0.01):
        if self._sent:
            time.sleep(min(0.001, timeout_s))
            return None
        self._sent = True
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        return FramePacket(frame=frame, pts_ns=123, mono_ns=time.monotonic_ns())

    def shutdown(self) -> None:
        return None


def test_perception_no_frame_returns_quickly():
    node = PerceptionNode(source=_NoFrameSource())
    st = node.step()
    assert st.debug.get("no_frame") is True
    assert st.debug.get("num_detections") == 0
    assert st.debug.get("detector_alive") is True
    assert st.debug.get("used_fallback_bbox") is False


def test_perception_stale_frame_fallback():
    node = PerceptionNode(source=_OneFrameSource())
    first = node.step()
    assert first.debug.get("no_frame") is None
    assert first.debug.get("num_detections") == 1
    assert first.debug.get("detector_alive") is True
    assert first.debug.get("used_fallback_bbox") is False

    second = node.step()
    assert second.debug.get("stale_frame") is True
    assert second.debug.get("num_detections") == 1
    assert second.debug.get("detector_alive") is True
    assert second.debug.get("used_fallback_bbox") is False


class _TwoFrameSource:
    def __init__(self):
        self._calls = 0
        self._base = 1_000_000_000
        self._delta = 50_000_000  # 50ms
        self._frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def get_latest(self, timeout_s: float = 0.01):
        if self._calls == 0:
            self._calls += 1
            return FramePacket(frame=self._frame, pts_ns=1, mono_ns=self._base)
        if self._calls == 1:
            self._calls += 1
            return FramePacket(frame=self._frame, pts_ns=2, mono_ns=self._base + self._delta)
        time.sleep(min(0.001, timeout_s))
        return None

    def shutdown(self) -> None:
        return None


def test_perception_fps_from_two_frames():
    node = PerceptionNode(source=_TwoFrameSource())
    first = node.step()
    assert first.fps is None
    second = node.step()
    assert second.fps is not None
    assert abs(second.fps - 20.0) < 1.0


def test_perception_zone_changes_in_dev(monkeypatch):
    class _Det:
        def __init__(self):
            self._calls = 0

        def detect(self, frame):
            h, w = frame.shape[:2]
            if self._calls == 0:
                self._calls += 1
                return [Detection(bbox_xyxy_px=(0, 0, w * 0.2, h * 0.4), conf=0.9, cls=0)]
            self._calls += 1
            return [Detection(bbox_xyxy_px=(w * 0.8, 0, w * 1.0, h * 0.4), conf=0.9, cls=0)]

    node = PerceptionNode(detector=_Det())
    first = node.step()
    second = node.step()

    assert first.primary_person is not None
    assert second.primary_person is not None
    assert first.debug.get("zone_hint") != second.debug.get("zone_hint")


def test_perception_detector_error_sets_debug_and_uses_fallback():
    class _ErrDet:
        def detect(self, _frame):
            raise RuntimeError("detector down")

    node = PerceptionNode(detector=_ErrDet())
    st = node.step()
    assert st.debug.get("detector_alive") is False
    assert st.debug.get("num_detections") == 0
    assert st.debug.get("used_fallback_bbox") is True
    assert "detector_error" in st.debug


def test_perception_real_source_no_detection_reports_no_person():
    class _NoDet:
        def detect(self, _frame):
            return []

    node = PerceptionNode(source=_OneFrameSource(), detector=_NoDet())
    st = node.step()
    assert st.primary_person is None
    assert st.primary_person_conf is None
    assert st.debug.get("zone_hint") is None
    assert st.debug.get("num_detections") == 0
    assert st.debug.get("used_fallback_bbox") is False
    assert st.latency_ms is not None
    assert st.latency_ms >= 0.0


def test_perception_source_error_is_reported_without_crashing():
    class _ErrSource:
        def get_packet(self):
            raise RuntimeError("camera unavailable")

        def shutdown(self) -> None:
            return None

    node = PerceptionNode(source=_ErrSource())
    st = node.step()
    assert st.debug.get("no_frame") is True
    assert st.debug.get("source_alive") is False
    assert "source_error" in st.debug
    assert "camera unavailable" in str(st.debug.get("source_error"))


def test_perception_shutdown_calls_detector_and_source():
    calls = {"source": 0, "detector": 0}

    class _Source:
        def shutdown(self) -> None:
            calls["source"] += 1

    class _Detector:
        def detect(self, _frame):
            return []

        def shutdown(self) -> None:
            calls["detector"] += 1

    node = PerceptionNode(source=_Source(), detector=_Detector())
    node.shutdown()
    assert calls == {"source": 1, "detector": 1}


def test_perception_shutdown_still_closes_source_if_detector_shutdown_fails():
    calls = {"source": 0, "detector": 0}

    class _Source:
        def shutdown(self) -> None:
            calls["source"] += 1

    class _Detector:
        def detect(self, _frame):
            return []

        def shutdown(self) -> None:
            calls["detector"] += 1
            raise RuntimeError("detector shutdown failure")

    node = PerceptionNode(source=_Source(), detector=_Detector())
    node.shutdown()
    assert calls == {"source": 1, "detector": 1}
