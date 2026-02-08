from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pala.perception.preview_tap import PreviewTapWriter
from tools.telemetry.protocol import decode_message, encode_message, event
from tools.telemetry.jetson_agent import _TapVideoSource, _encode_jpeg_frame, _parse_tegrastats
from tools.telemetry.mac_viewer import (
    _build_parser,
    _build_remote_agent_command,
    _fit_image_to_window,
    _normalize_jetson_dir,
)


def test_protocol_roundtrip():
    msg = event(source="test", payload={"ok": True})
    line = encode_message(msg)
    decoded = decode_message(line)
    assert decoded is not None
    assert decoded["source"] == "test"
    assert decoded["payload"]["ok"] is True


def test_protocol_decode_invalid():
    assert decode_message("") is None
    assert decode_message("not json") is None
    assert decode_message("[1,2,3]") is None


def test_parse_tegrastats_extracts_core_fields():
    line = (
        "RAM 2233/7802MB (lfb 314x4MB) SWAP 0/3901MB "
        "CPU [11%@1728,off,7%@1728,9%@1728,10%@1728,off] "
        "EMC_FREQ 12%@1600 GR3D_FREQ 35%@624 "
        "CPU@47.5C GPU@45.0C Tdiode@49.2C"
    )
    parsed = _parse_tegrastats(line)
    assert parsed["ram_used_mb"] == 2233
    assert parsed["ram_total_mb"] == 7802
    assert parsed["gpu_util_pct"] == 35
    assert parsed["emc_util_pct"] == 12
    assert parsed["cpu_cores_online"] == 4
    assert parsed["cpu_util_avg_pct"] == 9.2
    assert parsed["temp_max_c"] == 49.2


def test_encode_jpeg_frame_scales_and_encodes():
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[:, :, 1] = 200
    jpeg, width, height = _encode_jpeg_frame(
        frame,
        max_width=32,
        max_height=24,
        quality=80,
    )
    assert jpeg[:2] == b"\xff\xd8"
    assert width <= 32
    assert height <= 24


def test_normalize_jetson_dir_maps_mac_absolute():
    mapped, note = _normalize_jetson_dir("/Users/djduecaster/development/pala")
    assert mapped == "~/pala"
    assert note is not None


def test_normalize_jetson_dir_keeps_linux_absolute():
    mapped, note = _normalize_jetson_dir("/home/dylan/pala")
    assert mapped == "/home/dylan/pala"
    assert note is None


def test_normalize_jetson_dir_keeps_tilde_path():
    mapped, note = _normalize_jetson_dir("~/pala")
    assert mapped == "~/pala"
    assert note is None


def test_remote_command_expands_tilde_dir():
    args = _build_parser().parse_args([])
    args.jetson_dir = "~/pala"
    cmd = _build_remote_agent_command(args)
    assert "PALA_TELEMETRY_JETSON_DIR" in cmd
    assert "PALA_TELEMETRY_JETSON_DIR#\\~/" in cmd
    assert 'cd "$PALA_TELEMETRY_JETSON_DIR"' in cmd
    assert "tools.telemetry.jetson_agent" in cmd
    assert "&& --perception-log" not in cmd
    assert "--video-source" in cmd
    assert "--video-tap-jpeg" in cmd
    assert "--video-tap-meta" in cmd


def test_preview_tap_writer_emits_files_and_throttles(tmp_path):
    jpeg_path = tmp_path / "latest.jpg"
    meta_path = tmp_path / "latest.json"
    writer = PreviewTapWriter(
        enabled=True,
        jpeg_path=str(jpeg_path),
        meta_path=str(meta_path),
        max_hz=2.0,
        max_width=40,
        max_height=30,
        jpeg_quality=70,
    )

    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[:, :, 2] = 255
    writer.write(frame, mono_ns=1_000_000_000, pts_ns=123)
    first_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert first_meta["frame_id"] == 0
    assert first_meta["pts_ns"] == 123
    assert first_meta["width"] <= 40
    assert first_meta["height"] <= 30

    # 0.2s delta < 0.5s period, so this should be skipped.
    writer.write(frame, mono_ns=1_200_000_000, pts_ns=124)
    second_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert second_meta["frame_id"] == 0
    assert second_meta["pts_ns"] == 123

    writer.write(frame, mono_ns=1_700_000_000, pts_ns=125)
    third_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert third_meta["frame_id"] == 1
    assert third_meta["pts_ns"] == 125


def test_tap_video_source_reads_written_preview(tmp_path):
    jpeg_path = tmp_path / "latest.jpg"
    meta_path = tmp_path / "latest.json"

    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[:, :, 0] = 180
    Image.fromarray(frame).save(jpeg_path, format="JPEG", quality=75)
    meta_path.write_text(
        json.dumps({"frame_id": 7, "mono_ns": 555_000_000, "pts_ns": 444_000_000}),
        encoding="utf-8",
    )

    src = _TapVideoSource(jpeg_path=str(jpeg_path), meta_path=str(meta_path))
    out_frame, pts_ns, mono_ns = src.get_frame()
    assert out_frame.shape == (24, 32, 3)
    assert pts_ns == 444_000_000
    assert mono_ns == 555_000_000


def test_fit_image_to_window_returns_target_size():
    image = Image.new("RGB", (320, 180), (5, 10, 15))
    fitted = _fit_image_to_window(image, target_w=200, target_h=200)
    assert fitted.size == (200, 200)
