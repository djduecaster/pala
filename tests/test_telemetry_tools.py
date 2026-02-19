from __future__ import annotations

import json
import pathlib
import sys
import base64
import time

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pala.perception.preview_tap import PreviewTapWriter
from tools.telemetry.lamp_viz import draw_lamp_panel
from tools.telemetry.protocol import decode_message, encode_message, event
from tools.telemetry.jetson_agent import _TapVideoSource, _encode_jpeg_frame, _parse_tegrastats
from tools.telemetry.filters import parse_field_filter, matches_field_filters
from tools.telemetry.mac_viewer import (
    DashboardState,
    _build_parser,
    _build_remote_agent_command,
    _fit_image_to_window,
    _normalize_jetson_dir,
)
from tools.telemetry.packs import resolve_packs, apply_pack_overrides
from tools.telemetry.capture import CaptureConfig, SessionCaptureWriter
from tools.telemetry.replay import SessionReplayReader


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


def test_remote_command_forwards_pack_and_filters():
    args = _build_parser().parse_args([])
    args.pack = ["runtime_core", "memory_debug"]
    args.field_filter = ["actions_log.data.confidence<0.5"]
    cmd = _build_remote_agent_command(args)
    assert "--pack runtime_core" in cmd
    assert "--pack memory_debug" in cmd
    assert "--field-filter" in cmd
    assert "confidence<0.5" in cmd


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

    writer.write_with_extra(
        frame,
        mono_ns=2_300_000_000,
        pts_ns=126,
        extra={"command": {"joint_names": ["yaw"], "joint_angles_rad": [0.25], "enable": True}},
    )
    fourth_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert fourth_meta["frame_id"] == 2
    assert fourth_meta.get("extra", {}).get("command", {}).get("joint_names") == ["yaw"]


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


def test_dashboard_state_updates_command_from_video_tap_extra():
    state = DashboardState(host="jetson")
    state.apply(
        {
            "source": "video_frame",
            "payload": {
                "bytes_b64": base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii"),
                "frame_id": 1,
                "tap_extra": {
                    "command": {
                        "joint_names": ["yaw", "pitch1"],
                        "joint_angles_rad": [0.1, -0.2],
                        "enable": True,
                    }
                },
            },
        }
    )
    assert state.command is not None
    assert state.command["joint_names"] == ["yaw", "pitch1"]


def test_dashboard_state_updates_memory_and_timeline():
    state = DashboardState(host="jetson")
    state.apply({"source": "memory_log", "payload": {"data": {"type": "summary_event", "payload": {"highlights": ["a"]}}}})
    state.apply({"source": "timeline_log", "payload": {"data": {"type": "req_start", "payload": {"id": 3}}}})
    assert state.memory is not None
    assert state.timeline is not None


def test_field_filter_parsing_and_match():
    flt = parse_field_filter("actions_log.data.confidence<0.5")
    msg_ok = {"source": "actions_log", "payload": {"data": {"confidence": 0.4}}}
    msg_bad = {"source": "actions_log", "payload": {"data": {"confidence": 0.9}}}
    assert matches_field_filters(msg_ok, [flt]) is True
    assert matches_field_filters(msg_bad, [flt]) is False


def test_signal_pack_resolution_and_override():
    resolved = resolve_packs(["runtime_core", "memory_debug"])
    assert "perception_log" in resolved.sources
    assert "memory_log" in resolved.sources
    updated = apply_pack_overrides(resolved, ["exclude_sources=video_frame", "include_sources=journal"])
    assert "video_frame" not in updated.sources
    assert "journal" in updated.sources


def test_capture_and_replay_roundtrip(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(
        CaptureConfig(directory=str(session_dir), frames_mode="all", max_seconds=0.0, metadata={"test": True})
    )
    msg1 = {
        "type": "event",
        "source": "actions_log",
        "ts_wall_s": time.time(),
        "payload": {"data": {"primitive": "hold", "confidence": 0.7}},
    }
    frame = base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii")
    msg2 = {
        "type": "frame",
        "source": "video_frame",
        "ts_wall_s": time.time(),
        "payload": {"frame_id": 1, "bytes_b64": frame},
    }
    assert writer.write(msg1) is True
    assert writer.write(msg2) is True
    writer.close()

    reader = SessionReplayReader(str(session_dir))
    events = [msg for msg, _ in reader.iter_events()]
    assert len(events) == 2
    assert events[0]["source"] == "actions_log"
    assert events[1]["source"] == "video_frame"
    assert isinstance(events[1]["payload"].get("bytes_b64"), str)


def test_draw_lamp_panel_with_command_data():
    panel = draw_lamp_panel(
        height=240,
        width=260,
        command={
            "joint_names": ["yaw", "pitch1", "pitch2", "roll", "pitch3"],
            "joint_angles_rad": [0.2, -0.4, 0.3, 0.1, -0.2],
            "enable": True,
        },
    )
    assert panel.size == (260, 240)
