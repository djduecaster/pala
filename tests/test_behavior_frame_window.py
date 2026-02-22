from __future__ import annotations

import time

import numpy as np

from pala.behavior.frame_window import RollingFrameWindow, _even_sample_indices


def test_frame_window_rejects_non_rgb_inputs():
    window = RollingFrameWindow(window_s=2.0)
    base = time.monotonic_ns()
    assert window.add_frame(np.zeros((4, 4), dtype=np.uint8), mono_ns=base) is False
    assert window.add_frame(np.zeros((4, 4, 4), dtype=np.uint8), mono_ns=base + 1) is False
    assert len(window._items) == 0


def test_frame_window_converts_dtype_and_copies_input():
    window = RollingFrameWindow(window_s=2.0)
    base = time.monotonic_ns()
    src = np.full((3, 3, 3), 7.0, dtype=np.float32)
    assert window.add_frame(src, mono_ns=base) is True
    src[:, :, :] = 100.0

    latest = window.latest()
    assert latest is not None
    assert latest.frame.dtype == np.uint8
    assert int(latest.frame[0, 0, 0]) == 7


def test_frame_window_ignores_duplicate_mono_ns():
    window = RollingFrameWindow(window_s=2.0)
    base = time.monotonic_ns()
    frame_a = np.zeros((2, 2, 3), dtype=np.uint8)
    frame_b = np.ones((2, 2, 3), dtype=np.uint8)

    assert window.add_frame(frame_a, mono_ns=base) is True
    assert window.add_frame(frame_b, mono_ns=base) is False
    assert len(window._items) == 1
    assert int(window._items[0].frame[0, 0, 0]) == 0


def test_frame_window_prune_drops_old_items():
    window = RollingFrameWindow(window_s=0.2)
    now = time.monotonic_ns()
    window.add_frame(np.full((2, 2, 3), 1, dtype=np.uint8), mono_ns=now - 300_000_000)
    window.add_frame(np.full((2, 2, 3), 2, dtype=np.uint8), mono_ns=now - 100_000_000)
    window.add_frame(np.full((2, 2, 3), 3, dtype=np.uint8), mono_ns=now - 20_000_000)
    window.prune(now_ns=now)

    assert len(window._items) == 2
    assert int(window._items[0].frame[0, 0, 0]) == 2
    assert int(window._items[-1].frame[0, 0, 0]) == 3


def test_frame_window_sample_even_spacing_includes_newest():
    window = RollingFrameWindow(window_s=2.0)
    base = time.monotonic_ns()
    for i in range(10):
        window.add_frame(np.full((2, 2, 3), i, dtype=np.uint8), mono_ns=base + i * 1_000_000)

    sampled = window.sample(max_frames=4)
    values = [int(item.frame[0, 0, 0]) for item in sampled]
    assert values == [0, 3, 6, 9]


def test_even_sample_indices_edge_cases():
    assert _even_sample_indices(10, 1) == [9]
    assert _even_sample_indices(3, 5) == [0, 1, 2]
    assert _even_sample_indices(7, 3) == [0, 3, 6]
