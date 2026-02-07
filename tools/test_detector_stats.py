#!/usr/bin/env python3
"""Manual detector sanity tool (no control/hardware actuation)."""
from __future__ import annotations

import argparse
import time

from pala.config import load_config
from pala.hardware.camera_gst import GStreamerCamera
from pala.perception.detector import DeepStreamDetector, DummyDetector, JetsonDetector
from pala.perception.frame_source import CameraFrameSource, DummyFrameSource


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0, help="Run duration")
    parser.add_argument("--mode", type=str, default=None, help="Override config mode")
    parser.add_argument("--detector", type=str, default=None, help="Override detector backend")
    parser.add_argument("--force", action="store_true", help="Run even if mode != jetson_full")
    args = parser.parse_args()

    cfg = load_config("config/robot.yaml")
    mode = args.mode or cfg.mode
    detector_name = args.detector or cfg.detector

    if mode != "jetson_full" and not args.force:
        print("detector_stats: mode is not jetson_full; use --force to run anyway.")
        return 2

    if mode == "jetson_full":
        camera = GStreamerCamera(
            device=cfg.camera.device,
            width=cfg.camera.width,
            height=cfg.camera.height,
            fps=cfg.camera.fps,
            pipeline=cfg.camera.pipeline,
        )
        source = CameraFrameSource(camera)
    else:
        source = DummyFrameSource()

    if detector_name == "deepstream":
        detector = DeepStreamDetector(
            config_path=cfg.deepstream.config_path,
            person_class_id=cfg.deepstream.person_class_id,
            conf_threshold=cfg.deepstream.conf_threshold,
        )
    elif detector_name == "jetson":
        detector = JetsonDetector()
    elif detector_name == "dummy":
        detector = DummyDetector()
    else:
        raise ValueError(f"Unknown detector backend: {detector_name}")

    t_start = time.monotonic()
    t_end = t_start + max(0.1, float(args.seconds))
    interval_start = t_start
    interval_frames = 0
    interval_detected = 0
    interval_top_conf = 0.0
    last_error = None

    try:
        while time.monotonic() < t_end:
            packet = source.get_packet()
            interval_frames += 1
            try:
                dets = detector.detect(packet.frame)
            except Exception as exc:
                last_error = repr(exc)
                print(f"detector_error={last_error}")
                break

            if dets:
                interval_detected += 1
                top = max(float(d.conf) for d in dets)
                if top > interval_top_conf:
                    interval_top_conf = top

            now = time.monotonic()
            if now - interval_start >= 1.0:
                elapsed = max(1e-6, now - interval_start)
                fps = interval_frames / elapsed
                detected_ratio = interval_detected / max(1, interval_frames)
                print(
                    f"detector={detector_name} fps={fps:.2f} "
                    f"detected_frames={interval_detected}/{interval_frames} "
                    f"detected_ratio={detected_ratio:.2f} top_conf={interval_top_conf:.3f}"
                )
                interval_start = now
                interval_frames = 0
                interval_detected = 0
                interval_top_conf = 0.0
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(detector, "shutdown"):
            detector.shutdown()
        source.shutdown()

    if last_error is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
