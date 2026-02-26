from __future__ import annotations

import json

import numpy as np
import pytest

from pala.perception.frame_cache import LatestFrameCache
from pala.perception.detector.jetson_backend import JetsonDetector
from pala.utils import LatestValue, maybe_logger, stop_event
import pala.utils.timing as timing


def test_latest_frame_cache_returns_none_when_empty():
    cache = LatestFrameCache()
    assert cache.get() is None


def test_latest_frame_cache_respects_max_age(monkeypatch):
    cache = LatestFrameCache()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    cache.set(frame, mono_ns=1_000_000_000, pts_ns=42)

    monkeypatch.setattr("pala.perception.frame_cache.time.monotonic_ns", lambda: 1_200_000_000)
    fresh = cache.get(max_age_ms=250.0)
    assert fresh is not None
    assert fresh.frame is frame
    assert fresh.pts_ns == 42

    monkeypatch.setattr("pala.perception.frame_cache.time.monotonic_ns", lambda: 1_500_100_000)
    stale = cache.get(max_age_ms=500.0)
    assert stale is None


def test_rate_limiter_sleeps_until_next_tick(monkeypatch):
    now = {"value": 100.0}
    sleeps: list[float] = []

    def _monotonic() -> float:
        return now["value"]

    def _sleep(duration: float) -> None:
        sleeps.append(duration)
        now["value"] += duration

    monkeypatch.setattr(timing.time, "monotonic", _monotonic)
    monkeypatch.setattr(timing.time, "sleep", _sleep)

    limiter = timing.RateLimiter(2.0)  # period = 0.5s
    now["value"] = 100.1
    limiter.sleep()
    assert sleeps == []

    now["value"] = 100.2
    limiter.sleep()
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.3, abs=1e-6)


def test_latest_value_round_trip():
    latest = LatestValue[int]()
    assert latest.get() == (None, None)
    latest.set(7, 12.5)
    assert latest.get() == (7, 12.5)


def test_stop_event_returns_new_unset_event():
    ev1 = stop_event()
    ev2 = stop_event()
    assert ev1.is_set() is False
    assert ev2.is_set() is False
    assert ev1 is not ev2


def test_maybe_logger_writes_jsonl(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    logger = maybe_logger(str(path))
    assert logger is not None
    logger.write({"ok": True, "n": 1})
    logger.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"ok": True, "n": 1}
    assert maybe_logger(None) is None


def test_maybe_logger_redacts_data_image_urls(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    logger = maybe_logger(str(path))
    assert logger is not None
    logger.write(
        {
            "payload": {
                "a": "data:image/jpeg;base64,ABCDEF",
                "b": ["ok", "data:image/png;base64,XYZ"],
                "c": {"url": "http://example.com/image.jpg"},
            }
        }
    )
    logger.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["payload"]["a"] == "<image_data_url chars=29>"
    assert row["payload"]["b"][1] == "<image_data_url chars=25>"
    assert row["payload"]["c"]["url"] == "http://example.com/image.jpg"


def test_jetson_detector_stub_raises_not_implemented():
    detector = JetsonDetector()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError, match="not implemented"):
        detector.detect(frame)
