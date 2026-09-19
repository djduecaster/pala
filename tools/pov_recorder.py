"""Optional video sidecar consuming the runtime's latest RGB frame cache."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import subprocess
import threading
import time

from PIL import Image

from pala.perception.frame_cache import LatestFrameCache

LOG = logging.getLogger(__name__)


class PovRecorder:
    """Bounded-memory, best-effort recording; never runs work in a control loop.

    Output uses a constant frame rate. Duplicate source frames retain real-time
    pacing; timing JSONL identifies these and any stale camera frames. If encoding
    cannot keep pace, stop instead of silently producing a sped-up demo.
    """

    def __init__(self, cache: LatestFrameCache, directory: Path, *, fps: int = 20,
                 max_seconds: float | None = 600, fragmented: bool = False,
                 max_lag_s: float | None = 1.0) -> None:
        self.cache = cache
        self.directory = Path(directory)
        self.fps = fps
        self.max_seconds = max_seconds
        self.fragmented = fragmented
        self.max_lag_s = max_lag_s
        if not 1 <= fps <= 30 or (max_seconds is not None and not 0 < max_seconds <= 3600):
            raise ValueError('Invalid recording rate or duration')
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None
        self.process: subprocess.Popen | None = None
        self.frames = 0
        self.error: str | None = None

    def start(self) -> None:
        self.worker = threading.Thread(target=self._run, name='pov-recorder', daemon=True)
        self.worker.start()

    def _run(self) -> None:
        created_directory = False
        try:
            encoder = shutil.which('ffmpeg')
            if encoder is None:
                raise RuntimeError('ffmpeg is required for --record-pov')
            self.directory.mkdir(parents=True, exist_ok=False)
            created_directory = True
            with (self.directory / 'encoder.log').open('w') as errors, (self.directory / 'frames.jsonl').open('w') as timing:
                snapshot = None
                while not self.stop.is_set():
                    snapshot = self.cache.get(max_age_ms=500)
                    if snapshot is not None:
                        break
                    self.stop.wait(.05)
                if snapshot is None or self.stop.is_set():
                    return
                height, width = snapshot.frame.shape[:2]
                scale = min(1, 1280 / width, 720 / height)
                size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
                command = [encoder, '-nostdin', '-n', '-loglevel', 'warning',
                    '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{size[0]}x{size[1]}',
                    '-framerate', str(self.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                    '-preset', 'ultrafast', '-crf', '20', '-threads', '2', '-pix_fmt', 'yuv420p',
                    '-movflags', '+frag_keyframe+empty_moov+default_base_moof' if self.fragmented else '+faststart',
                    *(['-frag_duration', '1000000'] if self.fragmented else []),
                    str(self.directory / 'pov.mp4')]
                self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
                started = time.monotonic()
                (self.directory / 'session.json').write_text(json.dumps(dict(
                    fps=self.fps, width=size[0], height=size[1], start_monotonic_s=started,
                    max_seconds=self.max_seconds, fragmented=self.fragmented, max_lag_s=self.max_lag_s,
                    audio=False, source='runtime_latest_rgb'), indent=2))
                LOG.info('POV recording started: %s', self.directory / 'pov.mp4')
                previous = None
                while not self.stop.is_set():
                    due = started + self.frames / self.fps
                    if self.max_seconds is not None and self.frames / self.fps >= self.max_seconds:
                        break
                    if self.stop.wait(max(0, due - time.monotonic())):
                        break
                    now = time.monotonic()
                    if self.max_lag_s is not None and now - due > self.max_lag_s:
                        raise RuntimeError('Encoder fell over one second behind; recording stopped')
                    snapshot = self.cache.get()
                    if snapshot is None:
                        raise RuntimeError('Camera frame cache lost')
                    frame = Image.fromarray(snapshot.frame).convert('RGB')
                    if frame.size != size:
                        frame = frame.resize(size, Image.Resampling.BILINEAR)
                    self.process.stdin.write(frame.tobytes())
                    timing.write(json.dumps(dict(frame=self.frames, video_s=self.frames / self.fps,
                        sample_monotonic_s=now, source_mono_ns=snapshot.mono_ns,
                        duplicate=snapshot.mono_ns == previous,
                        stale=now - snapshot.mono_ns / 1e9 > .5)) + '\n')
                    previous = snapshot.mono_ns
                    self.frames += 1
                self.process.stdin.close()
                if self.process.wait(timeout=10) != 0:
                    raise RuntimeError('ffmpeg failed; see encoder.log')
        except Exception as exc:
            self.error = str(exc)
            LOG.warning('POV recording stopped: %s', exc)
        finally:
            if self.process is not None and self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            if created_directory:
                try:
                    (self.directory / 'result.json').write_text(json.dumps(dict(
                        frames=self.frames, video_seconds=self.frames / self.fps, error=self.error), indent=2))
                except OSError:
                    LOG.warning('Could not write POV recording result', exc_info=True)
            LOG.info('POV recording finished: frames=%s error=%s', self.frames, self.error)

    def close(self) -> None:
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=12)
            if self.worker.is_alive() and self.process is not None:
                self.process.kill()
                self.worker.join(timeout=2)
