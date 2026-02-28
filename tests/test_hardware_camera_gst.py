from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from pala.hardware.camera_gst import GStreamerCamera


def test_gstreamer_camera_requires_named_appsink(monkeypatch):
    class _FakePipeline:
        def __init__(self):
            self.states = []

        def set_state(self, state):
            self.states.append(state)

        def get_by_name(self, _name):
            return None

    fake_pipeline = _FakePipeline()

    class _FakeGst:
        class State:
            PLAYING = "PLAYING"
            NULL = "NULL"

        @staticmethod
        def init(_arg):
            return None

        @staticmethod
        def parse_launch(_pipeline):
            return fake_pipeline

    fake_gi = SimpleNamespace(require_version=lambda *_args, **_kwargs: None)
    fake_repository = SimpleNamespace(Gst=_FakeGst)
    monkeypatch.setitem(sys.modules, "gi", fake_gi)
    monkeypatch.setitem(sys.modules, "gi.repository", fake_repository)

    with pytest.raises(RuntimeError, match="missing appsink"):
        GStreamerCamera(device="/dev/video0", width=640, height=480, fps=30, pipeline="videotestsrc ! fakesink")

    assert fake_pipeline.states == ["PLAYING", "NULL"]
