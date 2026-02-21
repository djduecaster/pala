from __future__ import annotations

import json
import pathlib
import sys
import base64
import time
import io
import contextlib

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
from tools.telemetry.insights import build_improvement_report, load_improvement_report
from tools.telemetry.mac_viewer import (
    _apply_alert_policy,
    _apply_panel_preset,
    _build_in_memory_query_rows,
    _can_preload_trace_index,
    _write_query_export,
    _write_query_slice_export,
    DashboardState,
    _build_parser,
    _build_remote_agent_command,
    _fit_image_to_window,
    _normalize_jetson_dir,
)
from tools.telemetry.packs import resolve_packs, apply_pack_overrides
from tools.telemetry.capture import CaptureConfig, SessionCaptureWriter
from tools.telemetry.compare import compare_sessions
from tools.telemetry.dataset_export import export_dataset_rows
from tools.telemetry.doctor import build_doctor_report, load_doctor_report
from tools.telemetry.incident import build_incident_report, load_incident_report, render_incident_markdown
from tools.telemetry.replay import SessionReplayReader
from tools.telemetry.reasoning import format_reasoning_snippet, normalize_reasoning_message, redact_reasoning_text
from tools.telemetry.quality import evaluate_quality_gate, load_quality_report
from tools.telemetry.labels import load_labels_jsonl
from tools.telemetry.schema_v3 import TELEMETRY_SCHEMA_VERSION_V3
from tools.telemetry.scoreboard import load_scoreboard, summarize_scoreboard
from tools.telemetry.storage_sqlite import build_session_db, query_session_db, resolve_session_db_path
from tools.telemetry import telemetry as telemetry_cli
from tools.telemetry.telemetry import main as telemetry_main
from tools.telemetry.trace_graph import TraceGraphBuilder, TraceRecord, load_trace_index
from tools.telemetry.watchdog import resolve_candidate_sessions, run_watchdog


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


def test_remote_command_forwards_capture_metadata_tags():
    args = _build_parser().parse_args([])
    args.agent_capture_dir = "~/captures/session1"
    args.capture_frames = "keyframes"
    args.capture_max_seconds = 30.0
    args.scenario_tag = ["kitchen", "occlusion"]
    args.goal_tag = ["post_training"]
    args.runbook = "night-lighting"
    args.golden_session = ["/tmp/golden_a"]
    args.scoreboard_path = "logs/telemetry/scoreboard.json"
    args.no_scoreboard_update = True
    cmd = _build_remote_agent_command(args)
    assert "--capture-scenario-tag kitchen" in cmd
    assert "--capture-goal-tag post_training" in cmd
    assert "--capture-runbook night-lighting" in cmd
    assert "--capture-golden-session /tmp/golden_a" in cmd
    assert "--no-capture-scoreboard" in cmd


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
    reasoning_index = json.loads((session_dir / "reasoning_index.json").read_text(encoding="utf-8"))
    assert isinstance(reasoning_index.get("events"), list)
    trace_index = json.loads((session_dir / "trace_index.json").read_text(encoding="utf-8"))
    assert isinstance(trace_index.get("traces"), list)
    loaded_traces = load_trace_index(str(session_dir / "trace_index.json"))
    assert isinstance(loaded_traces, list)
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("schema_version") == TELEMETRY_SCHEMA_VERSION_V3
    assert manifest.get("trace_index_path") == "trace_index.json"
    assert manifest.get("doctor_report_path") == "doctor_report.json"
    assert manifest.get("incident_report_path") == "incident_report.json"
    assert isinstance(manifest.get("trace_count"), int)
    assert (session_dir / "session.db").exists()
    assert (session_dir / "quality_report.json").exists()
    assert (session_dir / "improvement_report.json").exists()
    assert (session_dir / "doctor_report.json").exists()
    assert (session_dir / "incident_report.json").exists()
    assert (session_dir / "labels.weak.jsonl").exists()


def test_session_db_query_and_quality_gate(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {
                "data": {
                    "type": "req_end",
                    "payload": {
                        "request_id": 42,
                        "status": "parse_fail",
                        "latency_ms": 3200,
                        "reasoning": "parse failed after timeout",
                    },
                }
            },
        }
    )
    writer.close()

    db_path = resolve_session_db_path(str(session_dir))
    out = query_session_db(db_path, query="status:parse_fail req:42", limit=10)
    assert out["events"]
    assert out["events"][0]["req_id"] == 42

    quality = load_quality_report(str(session_dir))
    assert quality is not None
    passed_warn, _ = evaluate_quality_gate(quality, "warn")
    assert isinstance(passed_warn, bool)


def test_dataset_export_from_weak_labels(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 7, "status": "parse_fail"}}},
        }
    )
    writer.close()

    labels = load_labels_jsonl(str(session_dir / "labels.weak.jsonl"))
    assert labels

    exported = export_dataset_rows(str(session_dir), include_unlabeled=False, min_label_confidence=0.5)
    assert exported["row_count"] >= 1
    assert pathlib.Path(exported["output_path"]).exists()


def test_query_trace_filter_scopes_event_results(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "journal",
            "ts_wall_s": time.time(),
            "payload": {"line": "generic warning without req id"},
            "level": "warning",
        }
    )
    writer.close()
    out = query_session_db(str(session_dir), query="trace:req:999", limit=10)
    assert out["events"] == []


def test_query_supports_since_and_quoted_phrase(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    now = time.time()
    writer.write(
        {
            "type": "event",
            "source": "journal",
            "ts_wall_s": now - 600.0,
            "payload": {"line": "camera timeout historical"},
            "level": "warning",
        }
    )
    writer.write(
        {
            "type": "event",
            "source": "journal",
            "ts_wall_s": now,
            "payload": {"line": "camera timeout current"},
            "level": "warning",
        }
    )
    writer.close()

    out = query_session_db(str(session_dir), query='since:5m "camera timeout"', limit=10)
    assert len(out["events"]) == 1
    ts_wall = float(out["events"][0].get("ts_wall_s", 0.0))
    assert ts_wall >= (now - 60.0)
    counts = out.get("counts")
    assert isinstance(counts, dict)
    assert int(counts.get("events", 0)) == 1


def test_query_supports_kind_filter(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_start", "payload": {"request_id": 5}}},
        }
    )
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 5, "status": "ok"}}},
        }
    )
    writer.close()

    trace_only = query_session_db(str(session_dir), query="kind:trace", limit=10)
    assert trace_only["events"] == []
    assert trace_only["reasoning"] == []
    assert len(trace_only["traces"]) >= 1

    reasoning_only = query_session_db(str(session_dir), query="kind:reasoning", limit=10)
    assert reasoning_only["events"] == []
    assert reasoning_only["traces"] == []
    assert len(reasoning_only["reasoning"]) >= 1


def test_in_memory_query_rows_support_event_kind():
    state = DashboardState(host="jetson")
    state.apply(
        {
            "source": "journal",
            "ts_wall_s": time.time(),
            "level": "warning",
            "payload": {"line": "camera timeout detected"},
        }
    )
    state.apply(
        {
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 3, "status": "ok", "reasoning": "done"}}},
        }
    )
    state.query_text = 'kind:event source:journal "camera timeout"'
    rows = _build_in_memory_query_rows(state, limit=10, now_wall_s=time.time())
    assert len(rows) == 1
    assert rows[0].get("kind") == "event"
    assert rows[0].get("source") == "journal"


def test_query_export_writer(tmp_path):
    out_path = tmp_path / "query_export.json"
    _write_query_export(
        str(out_path),
        query="status:error",
        note="test",
        rows=[{"kind": "event", "id": 1}],
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload.get("query") == "status:error"
    assert payload.get("row_count") == 1


def test_query_slice_export_writer(tmp_path):
    out_path = tmp_path / "query_slice.jsonl"
    count = _write_query_slice_export(
        str(out_path),
        query="status:timeout",
        rows=[{"kind": "trace", "id": "req:7", "trace_id": "req:7", "summary": "planner timeout"}],
    )
    assert count == 1
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[0])
    assert payload.get("label") == "query_slice"
    assert payload.get("query") == "status:timeout"


def test_apply_alert_policy_sets_thresholds():
    args = _build_parser().parse_args(["--alert-policy", "demo"])
    _apply_alert_policy(args)
    assert args.alert_stale_s <= 5.0
    assert args.alert_video_idle_s <= 4.0
    assert args.alert_dropped_events == 1


def test_improvement_report_builder(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 9, "status": "parse_fail", "latency_ms": 2500}}},
        }
    )
    writer.close()

    report = build_improvement_report(str(session_dir))
    assert isinstance(report.get("recommendations"), list)
    assert report.get("summary", {}).get("parse_fail_count", 0) >= 1
    loaded = load_improvement_report(str(session_dir))
    assert loaded is not None
    assert "summary" in loaded


def test_capture_writer_updates_scoreboard(tmp_path):
    session_dir = tmp_path / "session"
    board_path = tmp_path / "scoreboard.json"
    writer = SessionCaptureWriter(
        CaptureConfig(
            directory=str(session_dir),
            frames_mode="off",
            max_seconds=0.0,
            scenario_tags=["lab"],
            goal_tags=["ptx"],
            runbook="runbook-a",
            scoreboard_path=str(board_path),
        )
    )
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 1, "status": "ok"}}},
        }
    )
    writer.close()
    board = load_scoreboard(str(board_path))
    assert isinstance(board.get("sessions"), list)
    assert len(board["sessions"]) == 1
    assert board["sessions"][0].get("scenario_tags") == ["lab"]


def test_doctor_report_builder_and_loader(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 4, "status": "parse_fail"}}},
        }
    )
    writer.close()

    report = build_doctor_report(str(session_dir))
    assert isinstance(report.get("readiness"), dict)
    loaded = load_doctor_report(str(session_dir))
    assert loaded is not None
    assert "checks" in loaded


def test_compare_sessions_reports_delta(tmp_path):
    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"

    writer_a = SessionCaptureWriter(CaptureConfig(directory=str(base_dir), frames_mode="off", max_seconds=0.0))
    writer_a.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 10, "status": "ok"}}},
        }
    )
    writer_a.close()

    writer_b = SessionCaptureWriter(CaptureConfig(directory=str(cand_dir), frames_mode="off", max_seconds=0.0))
    writer_b.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 11, "status": "parse_fail", "latency_ms": 3200}}},
        }
    )
    writer_b.close()

    out = compare_sessions(str(base_dir), str(cand_dir), parse_fail_increase_tol=0.0, timeout_increase_tol=0.0)
    assert out.get("verdict") in {"warn", "fail", "pass"}
    assert isinstance(out.get("delta"), dict)


def test_incident_report_builder_and_markdown(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 21, "status": "parse_fail"}}},
        }
    )
    writer.close()

    report = build_incident_report(str(session_dir), limit=4)
    assert report.get("severity") in {"critical", "high", "medium", "low"}
    loaded = load_incident_report(str(session_dir))
    assert loaded is not None
    md = render_incident_markdown(report)
    assert "## Summary" in md


def test_scoreboard_summary_leaderboards(tmp_path):
    board_path = tmp_path / "scoreboard.json"
    first = SessionCaptureWriter(
        CaptureConfig(
            directory=str(tmp_path / "s1"),
            frames_mode="off",
            max_seconds=0.0,
            scenario_tags=["desk"],
            goal_tags=["pt"],
            scoreboard_path=str(board_path),
        )
    )
    first.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 1, "status": "ok"}}},
        }
    )
    first.close()
    second = SessionCaptureWriter(
        CaptureConfig(
            directory=str(tmp_path / "s2"),
            frames_mode="off",
            max_seconds=0.0,
            scenario_tags=["desk", "night"],
            goal_tags=["pt"],
            scoreboard_path=str(board_path),
        )
    )
    second.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 2, "status": "ok"}}},
        }
    )
    second.close()
    board = load_scoreboard(str(board_path))
    summary = summarize_scoreboard(board, min_sessions=1, top_n=5)
    scenarios = summary.get("scenario_leaderboard")
    assert isinstance(scenarios, list)
    assert any(row.get("tag") == "desk" for row in scenarios)


def test_watchdog_resolve_and_run(tmp_path):
    baseline = tmp_path / "baseline"
    cand_root = tmp_path / "cands"
    cand_root.mkdir(parents=True, exist_ok=True)
    cand_a = cand_root / "cand_a"
    cand_b = cand_root / "cand_b"

    writer_base = SessionCaptureWriter(CaptureConfig(directory=str(baseline), frames_mode="off", max_seconds=0.0))
    writer_base.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 31, "status": "ok"}}},
        }
    )
    writer_base.close()

    writer_a = SessionCaptureWriter(CaptureConfig(directory=str(cand_a), frames_mode="off", max_seconds=0.0))
    writer_a.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 32, "status": "ok"}}},
        }
    )
    writer_a.close()
    writer_b = SessionCaptureWriter(CaptureConfig(directory=str(cand_b), frames_mode="off", max_seconds=0.0))
    writer_b.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 33, "status": "parse_fail"}}},
        }
    )
    writer_b.close()

    resolved = resolve_candidate_sessions([str(cand_root)], discover=True)
    assert len(resolved) == 2
    report = run_watchdog(str(baseline), resolved, parse_fail_increase_tol=0.0, timeout_increase_tol=0.0)
    assert report.get("candidate_count") == 2
    assert report.get("overall_verdict") in {"pass", "warn", "fail"}


def test_in_memory_query_rows_support_trace_kind():
    state = DashboardState(host="jetson")
    state.set_traces(
        [
            TraceRecord(
                trace_id="req:7",
                req_id=7,
                start_ts_wall_s=time.time() - 0.2,
                end_ts_wall_s=time.time() - 0.1,
                duration_ms=100.0,
                status="timeout",
                severity="error",
                summary="planner timeout",
                event_refs=tuple(),
            )
        ]
    )
    state.query_text = "kind:trace status:timeout"
    rows = _build_in_memory_query_rows(state, limit=10, now_wall_s=time.time())
    assert len(rows) == 1
    assert rows[0].get("kind") == "trace"


def test_capture_reasoning_index_includes_confidence(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "actions_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"primitive": "hold", "confidence": 0.2}},
        }
    )
    writer.close()
    reasoning_index = json.loads((session_dir / "reasoning_index.json").read_text(encoding="utf-8"))
    assert reasoning_index["events"][0].get("confidence") == 0.2


def test_manifest_trace_index_override_used_for_index_and_export(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    event = {
        "type": "event",
        "source": "timeline_log",
        "ts_wall_s": 1.0,
        "payload": {"data": {"type": "req_end", "payload": {"status": "ok", "reasoning": "done"}}},
    }
    (session_dir / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    (session_dir / "reasoning_index.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_index": 0,
                        "source": "timeline_log",
                        "ts_wall_s": 1.0,
                        "req_id": 99,
                        "phase": "req_end",
                        "status": "ok",
                        "severity": "info",
                        "snippet": "done",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "labels.weak.jsonl").write_text(
        json.dumps({"event_index": 0, "label": "trace_failure", "confidence": 0.7}) + "\n",
        encoding="utf-8",
    )
    (session_dir / "manifest.json").write_text(
        json.dumps({"trace_index_path": "alt_trace.json", "event_count": 1}),
        encoding="utf-8",
    )
    (session_dir / "alt_trace.json").write_text(
        json.dumps(
            {
                "version": 1,
                "traces": [
                    {
                        "trace_id": "req:99",
                        "req_id": 99,
                        "start_ts_wall_s": 1.0,
                        "end_ts_wall_s": 1.1,
                        "duration_ms": 100.0,
                        "status": "ok",
                        "severity": "info",
                        "summary": "manual trace",
                        "event_refs": [
                            {
                                "event_index": 0,
                                "source": "timeline_log",
                                "ts_wall_s": 1.0,
                                "req_id": 99,
                                "phase": "req_end",
                                "status": "ok",
                                "latency_ms": None,
                                "severity": "info",
                                "summary": "manual trace",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_session_db(str(session_dir))
    assert summary["trace_count"] == 1
    out = query_session_db(str(session_dir), query="trace:req:99", limit=10)
    assert len(out["events"]) == 1
    assert out["events"][0].get("trace_id") == "req:99"

    exported = export_dataset_rows(str(session_dir), include_unlabeled=False, min_label_confidence=0.6)
    lines = (session_dir / "dataset_rows.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert exported["row_count"] == 1
    row = json.loads(lines[0])
    assert row.get("trace_id") == "req:99"


def test_build_parser_accepts_alert_panel_and_thresholds():
    args = _build_parser().parse_args(
        [
            "--panel",
            "alerts",
            "--panel",
            "throughput",
            "--panel",
            "insights",
            "--panel",
            "story",
            "--panel",
            "scoreboard",
            "--panel",
            "doctor",
            "--panel",
            "incident",
            "--alert-stale-s",
            "7.5",
            "--alert-policy",
            "training",
            "--index-refresh-s",
            "9",
            "--query-export",
            "out.json",
            "--query-slice-export",
            "slice.jsonl",
            "--insight-max-recommendations",
            "3",
            "--doctor-gate",
            "warn",
        ]
    )
    assert "alerts" in args.panel
    assert "throughput" in args.panel
    assert "insights" in args.panel
    assert "story" in args.panel
    assert "scoreboard" in args.panel
    assert "doctor" in args.panel
    assert "incident" in args.panel
    assert args.alert_stale_s == 7.5
    assert args.alert_policy == "training"
    assert args.index_refresh_s == 9
    assert args.query_export == "out.json"
    assert args.query_slice_export == "slice.jsonl"
    assert args.insight_max_recommendations == 3
    assert args.doctor_gate == "warn"


def test_replay_rejects_frame_ref_path_traversal(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"\xff\xd8\xff\xd9")

    event = {
        "type": "frame",
        "source": "video_frame",
        "ts_wall_s": time.time(),
        "payload": {"frame_ref": "../outside.jpg"},
    }
    (session_dir / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    reader = SessionReplayReader(str(session_dir))
    events = [msg for msg, _ in reader.iter_events()]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload.get("frame_ref") == "../outside.jpg"
    assert "bytes_b64" not in payload


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


def test_reasoning_normalization_timeline_event():
    msg = {
        "source": "timeline_log",
        "ts_wall_s": 123.4,
        "payload": {
            "data": {
                "type": "req_end",
                "payload": {
                    "id": 7,
                    "status": "parse_fail",
                    "latency_ms": 321.0,
                    "reasoning": "token=abc123 parse failed",
                },
            }
        },
    }
    out = normalize_reasoning_message(msg)
    assert out is not None
    assert out.req_id == 7
    assert out.phase == "req_end"
    assert out.status == "parse_fail"
    assert out.latency_ms == 321.0
    assert out.severity == "error"


def test_reasoning_redaction_and_snippet():
    raw = "authorization=Bearer abc.def.ghi token=secretvalue1234567890"
    redacted = redact_reasoning_text(raw)
    assert "secretvalue1234567890" not in redacted
    snippet = format_reasoning_snippet(raw, max_chars=24, redact=True)
    assert len(snippet) <= 24


def test_dashboard_reasoning_filters_and_selection():
    state = DashboardState(host="jetson")
    state.configure_panels(["summary", "reasoning_stream", "request_detail"], focus_panel="reasoning_stream")
    state.apply(
        {
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"id": 1, "status": "ok", "latency_ms": 3000}}},
        }
    )
    state.apply(
        {
            "source": "timeline_log",
            "ts_wall_s": time.time(),
            "payload": {"data": {"type": "req_end", "payload": {"id": 2, "status": "parse_fail", "latency_ms": 100}}},
        }
    )
    all_events = state.filtered_reasoning_events(slow_ms=2000)
    assert len(all_events) == 2
    state.reasoning_filter_mode = "errors"
    err_events = state.filtered_reasoning_events(slow_ms=2000)
    assert len(err_events) == 1
    state.reasoning_filter_mode = "slow"
    slow_events = state.filtered_reasoning_events(slow_ms=2000)
    assert len(slow_events) == 1
    state.move_reasoning_selection(delta=-1, slow_ms=2000)
    assert state.reasoning_selected_seq is not None


def test_panel_preset_changes_active_panels():
    args = _build_parser().parse_args([])
    state = DashboardState(host="jetson")
    state.configure_panels(["summary"], focus_panel="summary")
    _apply_panel_preset(state, args, "1")
    assert "reasoning_stream" in args.panel
    assert "request_detail" in args.panel
    assert "trace_list" in args.panel
    assert "trace_detail" in args.panel


def test_dashboard_trace_selection_and_pin():
    state = DashboardState(host="jetson")
    traces = [
        TraceRecord(
            trace_id="req:1",
            req_id=1,
            start_ts_wall_s=1.0,
            end_ts_wall_s=1.5,
            duration_ms=500.0,
            status="ok",
            severity="info",
            summary="ok",
            event_refs=tuple(),
        ),
        TraceRecord(
            trace_id="time:1000-1",
            req_id=None,
            start_ts_wall_s=2.0,
            end_ts_wall_s=2.1,
            duration_ms=100.0,
            status="error",
            severity="error",
            summary="error",
            event_refs=tuple(),
        ),
    ]
    state.set_traces(traces)
    assert state.selected_trace() is not None
    state.move_trace_selection(1)
    selected = state.selected_trace()
    assert selected is not None
    assert selected.trace_id == "time:1000-1"
    state.toggle_trace_pin()
    assert state.trace_pinned_id == "time:1000-1"
    state.toggle_trace_pin()
    assert state.trace_pinned_id is None


def test_trace_graph_correlates_by_req_id():
    b = TraceGraphBuilder(match_window_s=2.0, max_events=100)
    b.ingest(
        {
            "seq": 1,
            "source": "timeline_log",
            "ts_wall_s": 10.0,
            "payload": {"data": {"type": "request_start", "payload": {"request_id": 42}}},
        }
    )
    b.ingest(
        {
            "seq": 2,
            "source": "timeline_log",
            "ts_wall_s": 10.4,
            "payload": {"data": {"type": "request_end", "payload": {"request_id": 42, "status": "parse_fail"}}},
        }
    )
    traces = b.traces()
    assert len(traces) == 1
    assert traces[0].req_id == 42
    assert traces[0].status == "parse_fail"


def test_trace_graph_temporal_fallback_without_req_id():
    b = TraceGraphBuilder(match_window_s=2.0, max_events=100)
    b.ingest(
        {
            "seq": 1,
            "source": "timeline_log",
            "ts_wall_s": 20.0,
            "payload": {"data": {"type": "request_start", "payload": {"request_id": 8}}},
        }
    )
    b.ingest(
        {
            "seq": 2,
            "source": "journal",
            "ts_wall_s": 20.8,
            "payload": {"line": "orchestrator warning timeout waiting for response"},
            "level": "warning",
        }
    )
    traces = b.traces()
    assert len(traces) == 1
    assert len(traces[0].event_refs) == 2


def test_trace_graph_splits_far_events():
    b = TraceGraphBuilder(match_window_s=2.0, max_events=100)
    b.ingest({"seq": 1, "source": "journal", "ts_wall_s": 1.0, "payload": {"line": "error first failure"}, "level": "warning"})
    b.ingest({"seq": 2, "source": "journal", "ts_wall_s": 10.0, "payload": {"line": "error second failure"}, "level": "warning"})
    traces = b.traces()
    assert len(traces) == 2


def test_trace_graph_extracts_id_from_orchestrator_text():
    b = TraceGraphBuilder(match_window_s=2.0, max_events=100)
    b.ingest(
        {
            "seq": 1,
            "source": "journal",
            "ts_wall_s": 1.0,
            "payload": {"line": "orchestrator timeout id=77 while waiting"},
            "level": "warning",
        }
    )
    traces = b.traces()
    assert len(traces) == 1
    assert traces[0].req_id == 77


def test_can_preload_trace_index_requires_no_field_filters():
    assert _can_preload_trace_index(field_filters=[]) is True
    assert _can_preload_trace_index(field_filters=[object()]) is False


def test_unified_telemetry_cli_lists_packs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["telemetry", "packs"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = telemetry_main()
    out = buf.getvalue()
    assert rc == 0
    assert "reasoning_live" in out


def test_unified_telemetry_cli_viewer_allows_double_dash(monkeypatch):
    captured = {}

    def _fake_run(cmd, check=False):
        captured["cmd"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(telemetry_cli.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", ["telemetry", "viewer", "--", "--jetson-host", "jetson"])
    rc = telemetry_main()
    assert rc == 0
    cmd = captured.get("cmd", [])
    assert "--" not in cmd
    assert "--jetson-host" in cmd


def test_unified_telemetry_cli_dispatches_doctor(monkeypatch):
    captured = {}

    def _fake_run(cmd, check=False):
        captured["cmd"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(telemetry_cli.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", ["telemetry", "doctor", "--", "logs/telemetry/session_001"])
    rc = telemetry_main()
    assert rc == 0
    cmd = captured.get("cmd", [])
    assert "tools.telemetry.doctor" in cmd


def test_unified_telemetry_cli_dispatches_compare(monkeypatch):
    captured = {}

    def _fake_run(cmd, check=False):
        captured["cmd"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(telemetry_cli.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", ["telemetry", "compare", "--", "a", "b"])
    rc = telemetry_main()
    assert rc == 0
    cmd = captured.get("cmd", [])
    assert "tools.telemetry.compare" in cmd


def test_unified_telemetry_cli_dispatches_incident(monkeypatch):
    captured = {}

    def _fake_run(cmd, check=False):
        captured["cmd"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(telemetry_cli.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", ["telemetry", "incident", "--", "s"])
    rc = telemetry_main()
    assert rc == 0
    cmd = captured.get("cmd", [])
    assert "tools.telemetry.incident" in cmd


def test_unified_telemetry_cli_dispatches_scoreboard(monkeypatch):
    captured = {}

    def _fake_run(cmd, check=False):
        captured["cmd"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(telemetry_cli.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", ["telemetry", "scoreboard"])
    rc = telemetry_main()
    assert rc == 0
    cmd = captured.get("cmd", [])
    assert "tools.telemetry.scoreboard" in cmd


def test_unified_telemetry_cli_dispatches_watchdog(monkeypatch):
    captured = {}

    def _fake_run(cmd, check=False):
        captured["cmd"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(telemetry_cli.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", ["telemetry", "watchdog", "--", "b", "c"])
    rc = telemetry_main()
    assert rc == 0
    cmd = captured.get("cmd", [])
    assert "tools.telemetry.watchdog" in cmd
