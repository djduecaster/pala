"""Record a stationary lamp POV without starting servo control or Gemini."""
from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import signal
import sys
import threading
import time
from uuid import uuid4

from PIL import Image

from pala.config import load_config
from pala.perception.frame_cache import LatestFrameCache
from pala.perception.frame_source import CameraFrameSource, DummyFrameSource, ThreadedFrameSource
from pala.utils.manual_console import ManualConsole
from tools.pov_recorder import PovRecorder

LOG = logging.getLogger(__name__)
HELP = 'snap: save a full-resolution still | status | help | stop/q/shutdown: finalize video and exit'


def record(source: ThreadedFrameSource, directory: Path, *, seconds: float | None,
           stop: threading.Event, console: ManualConsole | None = None) -> int:
    """Own camera capture and video only; the caller owns source cleanup."""
    cache = LatestFrameCache()
    recorder = PovRecorder(cache, directory, max_seconds=seconds, fragmented=True, max_lag_s=None)
    still_count = 0
    last_frame_at = time.monotonic()
    failed = False
    recorder.start()
    try:
        while not stop.is_set() and recorder.worker.is_alive():
            packet = source.get_latest(timeout_s=.05)
            if packet is not None:
                cache.set(packet.frame, mono_ns=packet.mono_ns, pts_ns=packet.pts_ns)
                last_frame_at = time.monotonic()
            if time.monotonic() - last_frame_at > 5:
                LOG.error('No camera frames for five seconds; stopping recording')
                failed = True
                break
            for token in console.poll() if console else []:
                token = token.lower()
                if token in {'stop', 'q', 'shutdown'}:
                    stop.set()
                elif token == 'snap':
                    snapshot = cache.get(max_age_ms=500)
                    if snapshot is None or not directory.is_dir():
                        LOG.warning('No fresh frame available for a still')
                        continue
                    still_count += 1
                    path = directory / f'still_{still_count:04d}.jpg'
                    Image.fromarray(snapshot.frame).save(path, quality=95)
                    LOG.info('Still saved: %s', path)
                elif token == 'status':
                    LOG.info('Recorded %.1fs; stills=%s; error=%s',
                             recorder.frames / recorder.fps, still_count, recorder.error)
                else:
                    LOG.info(HELP)
    finally:
        recorder.close()
    return 1 if failed or recorder.error or recorder.frames == 0 else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['dev', 'jetson_full'], default='dev',
                        help='dev uses dummy frames; jetson_full opens only the camera')
    parser.add_argument('--config', default='config/robot.yaml')
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--until-stopped', action='store_true',
                          help='No automatic duration limit; stop explicitly to finalize')
    duration.add_argument('--seconds', type=float, default=600,
                        help='Maximum video duration, 1–3600 seconds (default: 600)')
    parser.add_argument('--output-root', type=Path, default=Path('logs/intro'))
    parser.add_argument('--no-console', action='store_true', help='Use duration or signals to stop')
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 3600:
        parser.error('--seconds must be between 1 and 3600')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    directory = args.output_root / f'{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}'
    LOG.info('Camera-only recording: NO servo commands, NO Gemini. Establish zero before launching; this tool does not move or actively hold the lamp.')
    LOG.info('Stop other camera owners before launching. Output: %s', directory)
    LOG.info(HELP)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    source = None
    try:
        if args.mode == 'dev':
            inner = DummyFrameSource()
        else:
            from pala.hardware.camera_gst import GStreamerCamera
            cfg = load_config(args.config)
            inner = CameraFrameSource(GStreamerCamera(
                device=cfg.camera.device, width=cfg.camera.width, height=cfg.camera.height,
                fps=cfg.camera.fps, pipeline=cfg.camera.pipeline))
        source = ThreadedFrameSource(inner, min_interval_s=.01)
        console = None if args.no_console else ManualConsole(sys.stdin)
        return record(source, directory, seconds=None if args.until_stopped else args.seconds, stop=stop, console=console)
    finally:
        if source is not None:
            source.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
