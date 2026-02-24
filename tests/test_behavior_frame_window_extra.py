from __future__ import annotations

import builtins

import numpy as np

from pala.behavior.frame_window import RollingFrameWindow, _even_sample_indices


def test_frame_window_sample_and_latest_when_empty():
    window = RollingFrameWindow(window_s=1.0)
    assert window.sample(max_frames=3) == []
    assert window.latest() is None


def test_even_sample_indices_defensive_clamps_and_fill(monkeypatch):
    original_round = builtins.round
    calls = {"n": 0}

    def _fake_round(value):
        calls["n"] += 1
        if calls["n"] == 1:
            return -10  # hit idx < 0 clamp
        if calls["n"] == 2:
            return 999  # hit idx > n-1 clamp
        return 0  # force duplicates to exercise dedupe/fill path

    monkeypatch.setattr("builtins.round", _fake_round)
    try:
        indices = _even_sample_indices(5, 4)
    finally:
        monkeypatch.setattr("builtins.round", original_round)

    assert len(indices) == 4
    assert indices[-1] == 4
    assert indices == sorted(indices)
    assert set(indices).issubset(set(range(5)))


def test_frame_window_add_frame_rejects_bad_shape_without_updating_state():
    window = RollingFrameWindow(window_s=1.0)
    ok = window.add_frame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10)
    assert ok is True
    last_seen_before = window._last_seen_mono_ns  # noqa: SLF001

    bad = window.add_frame(np.zeros((4, 4), dtype=np.uint8), mono_ns=11)
    assert bad is False
    assert window._last_seen_mono_ns == last_seen_before  # noqa: SLF001
    assert len(window._items) == 1  # noqa: SLF001
