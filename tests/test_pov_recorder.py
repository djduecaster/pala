import json
import shutil
import subprocess
import time

import numpy as np
import pytest

from pala.perception.frame_cache import LatestFrameCache
from tools.pov_recorder import PovRecorder


def test_missing_encoder_is_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, 'which', lambda _: None)
    recorder = PovRecorder(LatestFrameCache(), tmp_path / 'pov')
    recorder.start(); recorder.worker.join(2); recorder.close()
    assert 'ffmpeg is required' in recorder.error


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='requires video tools')
def test_real_mp4_dimensions_duration_and_frame_timing(tmp_path):
    cache = LatestFrameCache()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    cache.set(frame, mono_ns=time.monotonic_ns(), pts_ns=None)
    recorder = PovRecorder(cache, tmp_path / 'pov', fps=10, max_seconds=.6)
    recorder.start(); recorder.worker.join(5); recorder.close()
    assert recorder.error is None
    path = recorder.directory / 'pov.mp4'
    result = subprocess.run(['ffprobe','-v','error','-show_streams','-of','json',str(path)], capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)['streams'][0]
    assert (stream['width'], stream['height']) == (64, 48)
    assert stream['codec_name'] == 'h264'
    assert float(stream['duration']) == pytest.approx(.6, abs=.05)
    rows = [json.loads(line) for line in (recorder.directory/'frames.jsonl').read_text().splitlines()]
    assert len(rows) == 6 and rows[1]['duplicate']
    decoded = subprocess.run(['ffmpeg','-v','error','-i',str(path),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'], capture_output=True, check=True)
    pixels = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(48,64,3)
    assert pixels[:,:,0].mean() > 240 and pixels[:,:,2].mean() < 10


def test_close_without_camera_frame_does_not_hang(tmp_path):
    recorder = PovRecorder(LatestFrameCache(), tmp_path / 'pov')
    recorder.start(); recorder.close()
    assert not recorder.worker.is_alive()
    assert recorder.frames == 0
