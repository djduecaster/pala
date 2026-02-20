import time

import numpy as np

from pala.perception.frame_cache import LatestFrameCache
from pala.planner.scene_summarizer import AsyncSceneSummarizer
from pala.types import BBoxNorm, PerceptionState


def test_scene_summarizer_local_mode_emits_summary():
    cache = LatestFrameCache()
    summarizer = AsyncSceneSummarizer(
        frame_cache=cache,
        provider="local",
        summarizer_hz=20.0,
        summary_max_frames=2,
    )
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            cache.set(np.full((8, 8, 3), 2, dtype=np.uint8), mono_ns=time.monotonic_ns(), pts_ns=None)
            summarizer.observe(
                PerceptionState(
                    timestamp_monotonic_s=time.monotonic(),
                    primary_person=BBoxNorm(cx=0.5, cy=0.5, w=0.2, h=0.4),
                    primary_person_conf=0.8,
                    debug={"zone_hint": "center"},
                )
            )
            latest = summarizer.latest_summary()
            if latest is not None:
                break
            time.sleep(0.01)
        latest = summarizer.latest_summary()
        assert latest is not None
        assert latest.person_present is True
        assert latest.zone_hint == "center"
    finally:
        summarizer.shutdown()
