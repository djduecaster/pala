import json
import shutil
import threading
import time

import pytest
from PIL import Image

from pala.perception.frame_source import DummyFrameSource, ThreadedFrameSource
from tools.record_intro import main, record


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='requires ffmpeg')
@pytest.mark.parametrize("seconds", [10, None])
def test_camera_only_recording_still_and_console_stop(tmp_path, seconds):
    class Console:
        def __init__(self):
            self.started = time.monotonic()
            self.saved = False

        def poll(self):
            elapsed = time.monotonic() - self.started
            if elapsed > .5:
                return ['stop']
            if elapsed > .2 and not self.saved:
                self.saved = True
                return ['snap']
            return []

    source = ThreadedFrameSource(DummyFrameSource(), min_interval_s=.01)
    output = tmp_path / 'intro'
    try:
        assert record(source, output, seconds=seconds, stop=threading.Event(), console=Console()) == 0
    finally:
        source.shutdown()
    result = json.loads((output / 'result.json').read_text())
    assert result['error'] is None and 0 < result['video_seconds'] < 3
    assert (output / 'pov.mp4').stat().st_size > 0
    data = (output / 'pov.mp4').read_bytes()
    assert data.index(b'moov') < data.index(b'mdat')
    assert b'moof' in data
    session = json.loads((output / 'session.json').read_text())
    assert session['fragmented'] is True and session['max_lag_s'] is None
    with Image.open(output / 'still_0001.jpg') as frame:
        assert frame.size == (160, 90)


def test_invalid_duration_rejected_before_camera_open():
    with pytest.raises(SystemExit) as exc:
        main(['--mode', 'jetson_full', '--seconds', '0'])
    assert exc.value.code == 2
