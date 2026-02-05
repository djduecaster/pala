import time

import numpy as np
import pytest

from pala.perception.detector.deepstream_backend import DeepStreamDetector
from pala.perception.detector.interface import Detection


class _TestDetector(DeepStreamDetector):
    def __init__(self):
        super().__init__(config_path="dummy")
        self._calls = 0

    def _infer(self, frame: np.ndarray):
        self._calls += 1
        h, w = frame.shape[:2]
        return [Detection(bbox_xyxy_px=(0, 0, w * 0.5, h * 0.5), conf=0.9, cls=0)]


def test_deepstream_detector_async_returns_latest():
    det = _TestDetector()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    try:
        det.detect(frame)
        time.sleep(0.01)
        out = det.detect(frame)
        assert out
        assert isinstance(out[0], Detection)
    finally:
        det.shutdown()


def test_deepstream_detector_requires_config_path():
    det = DeepStreamDetector(config_path=None)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="config_path"):
        det.detect(frame)
    det.shutdown()
