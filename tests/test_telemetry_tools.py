from __future__ import annotations

import json
import pathlib
import sys
import base64
import time
import re

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
    _apply_mode_defaults,
    _append_viewer_run,
    _build_viewer_summary,
    load_timeline_rows,
    DashboardState,
    _build_parser,
    _build_remote_agent_command,
    _fit_image_to_window,
    _normalize_jetson_dir,
    _paths_equivalent,
    _resolve_live_save_session_dir,
    _run_curation_export,
    _write_viewer_summary,
)
from tools.telemetry.capture import CaptureConfig, SessionCaptureWriter
from tools.telemetry.dataset_export import export_dataset_rows
from tools.telemetry.annotations import annotation_key, append_annotation, load_annotations
from tools.telemetry.doctor import _check_session_dir
from tools.telemetry.integrity import build_integrity_report, verify_integrity_report, write_integrity_report
from tools.telemetry.replay import SessionReplayReader
from tools.telemetry.reasoning import format_reasoning_snippet, normalize_reasoning_message, redact_reasoning_text
from tools.telemetry.quality import evaluate_quality_gate, load_quality_report
from tools.telemetry.run_report import build_run_report
from tools.telemetry.pipeline import _build_parser as _build_pipeline_parser
from tools.telemetry.labels import load_labels_jsonl
from tools.telemetry.schema_v3 import REASONING_TRACE_INDEX_PATH, TELEMETRY_SCHEMA_VERSION_V3
from tools.telemetry.storage_sqlite import (
    build_session_db,
    query_case_detail_db,
    query_cases_db,
    query_session_db,
    resolve_session_db_path,
    review_case,
)
from tools.telemetry.trace_join import load_reasoning_trace_index
from tools.telemetry.trace_graph import TraceEventRef, TraceGraphBuilder, TraceRecord, load_trace_index


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
    assert "--behavior-trace-log" in cmd


def test_mode_defaults_apply_live_baseline():
    parser = _build_parser()
    args = parser.parse_args([])
    notes = _apply_mode_defaults(args, parser)
    assert notes == []
    assert args.mode == "live"
    assert args.pack == ["reasoning_live"]
    assert "reasoning_stream" in args.panel


def test_mode_defaults_apply_curate_profile():
    parser = _build_parser()
    args = parser.parse_args(["--mode", "curate", "--replay", "logs/telemetry/session"])
    notes = _apply_mode_defaults(args, parser)
    assert notes == []
    assert args.curate_on_exit is True
    assert args.no_video is True
    assert args.quality_gate == "strict"
    assert args.index_mode == "sqlite"
    assert args.pack == ["behavior_v2_debug"]
    assert "annotations" in args.panel
    assert "status:parse_fail" in args.query


def test_pipeline_parser_compile_command():
    parser = _build_pipeline_parser()
    args = parser.parse_args(["compile", "logs/telemetry/session"])
    assert args.cmd == "compile"
    assert args.session_dir == "logs/telemetry/session"


def test_pipeline_parser_capture_requires_save_session():
    parser = _build_pipeline_parser()
    args = parser.parse_args(["capture", "--save-session", "logs/telemetry/session_v4"])
    assert args.cmd == "capture"
    assert args.save_session == "logs/telemetry/session_v4"


def test_pipeline_parser_report_supports_strict():
    parser = _build_pipeline_parser()
    args = parser.parse_args(["report", "--root", "logs/telemetry", "--strict"])
    assert args.cmd == "report"
    assert args.strict is True


def test_pipeline_report_strict_exit_when_alerts(tmp_path):
    session_ok = tmp_path / "session_ok"
    session_empty = tmp_path / "session_empty"
    session_ok.mkdir(parents=True, exist_ok=True)
    session_empty.mkdir(parents=True, exist_ok=True)
    (session_ok / "viewer_summary.json").write_text(
        json.dumps({"run_id": "ok-1", "mode": "live", "exit_code": 0}),
        encoding="utf-8",
    )
    (session_empty / "manifest.json").write_text(json.dumps({"schema_version": 3}), encoding="utf-8")

    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.telemetry.pipeline",
            "report",
            "--root",
            str(tmp_path),
            "--strict",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "sessions_missing_runs:1" in proc.stdout


def test_remote_command_forwards_pack_and_trace_limits():
    args = _build_parser().parse_args([])
    args.pack = ["runtime_core", "memory_debug"]
    args.trace_max_events = 4096
    cmd = _build_remote_agent_command(args)
    assert "--pack runtime_core" in cmd
    assert "--pack memory_debug" in cmd
    assert "--trace-max-events" in cmd
    assert "4096" in cmd


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


def test_dashboard_state_tracks_memory_and_timeline_event_counts():
    state = DashboardState(host="jetson")
    state.apply({"source": "memory_log", "payload": {"data": {"type": "summary_event", "payload": {"highlights": ["a"]}}}})
    state.apply({"source": "timeline_log", "payload": {"data": {"type": "req_start", "payload": {"id": 3}}}})
    assert state.event_counts.get("memory_log") == 1
    assert state.event_counts.get("timeline_log") == 1


def test_field_filter_parsing_and_match():
    flt = parse_field_filter("actions_log.data.confidence<0.5")
    msg_ok = {"source": "actions_log", "payload": {"data": {"confidence": 0.4}}}
    msg_bad = {"source": "actions_log", "payload": {"data": {"confidence": 0.9}}}
    assert matches_field_filters(msg_ok, [flt]) is True
    assert matches_field_filters(msg_bad, [flt]) is False


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
    assert isinstance(manifest.get("trace_count"), int)
    assert manifest.get("reasoning_trace_index_path") == REASONING_TRACE_INDEX_PATH
    assert (session_dir / "session.db").exists()
    assert (session_dir / "quality_report.json").exists()
    assert (session_dir / "labels.weak.jsonl").exists()
    assert (session_dir / REASONING_TRACE_INDEX_PATH).exists()


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
    assert out["counts"]["events"] >= 1
    assert "joined" in out["counts"]

    quality = load_quality_report(str(session_dir))
    assert quality is not None
    passed_warn, _ = evaluate_quality_gate(quality, "warn")
    assert isinstance(passed_warn, bool)


def test_case_query_and_review_roundtrip(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": 1.0,
            "payload": {
                "data": {
                    "type": "req_end",
                    "payload": {
                        "request_id": 42,
                        "status": "parse_fail",
                        "latency_ms": 2500,
                        "reasoning": "parse failed after timeout",
                    },
                }
            },
        }
    )
    writer.close()

    db_path = resolve_session_db_path(str(session_dir))
    queried = query_cases_db(db_path, query="status:parse_fail", limit=10)
    assert queried["total_count"] >= 1
    assert queried["cases"]
    case_id = str(queried["cases"][0].get("case_id") or "")
    assert case_id.startswith("case:")
    assert queried["cases"][0].get("source") == "sqlite.cases.v4"

    updated = review_case(db_path, case_id=case_id, decision="accept", reviewer="pytest")
    assert updated["decision"] == "accept"
    detail = query_case_detail_db(db_path, case_id=case_id, event_limit=20)
    assert detail["case"] is not None
    assert detail["review"] is not None
    assert detail["review"]["decision"] == "accept"
    assert isinstance(detail.get("contexts"), dict)


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


def test_query_trace_filter_matches_journal_trace_without_reasoning_rows(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "journal",
            "ts_wall_s": time.time(),
            "payload": {"line": "planner warning request_id=42 timeout while waiting"},
            "level": "warning",
        }
    )
    writer.close()
    out = query_session_db(str(session_dir), query="trace:req:42", limit=10)
    assert len(out["events"]) == 1
    assert out["events"][0].get("trace_id") == "req:42"


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


def test_behavior_reasoning_normalization_and_joined_index(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "perception_log",
            "ts_wall_s": 10.0,
            "payload": {"data": {"primary_person_conf": 0.91, "debug": {"zone_hint": "desk"}, "frame_id": 77}},
        }
    )
    writer.write(
        {
            "type": "event",
            "source": "behavior_env_log",
            "ts_wall_s": 10.1,
            "payload": {
                "data": {
                    "request_id": 7,
                    "phase": "env_processor",
                    "status": "ok",
                    "delta_score": 0.78,
                    "summary": "person reached toward lamp",
                    "latency_ms": 130.0,
                }
            },
        }
    )
    writer.write(
        {
            "type": "event",
            "source": "behavior_planner_log",
            "ts_wall_s": 10.2,
            "payload": {
                "data": {
                    "request_id": 7,
                    "phase": "planner",
                    "status": "ok",
                    "decision_json": {"primitive": "hold", "confidence": 0.66, "command": {"target_zone": "desk"}},
                    "rationale_short": "maintain gentle focus",
                    "latency_ms": 220.0,
                }
            },
        }
    )
    writer.close()

    rows = load_reasoning_trace_index(str(session_dir))
    assert rows
    env_rows = [row for row in rows if row.get("component") == "env_processor"]
    planner_rows = [row for row in rows if row.get("component") == "planner"]
    assert env_rows
    assert planner_rows
    assert env_rows[0].get("perception_zone_hint") == "desk"
    assert planner_rows[0].get("trace_id") == "req:7"


def test_query_session_db_supports_joined_kind_and_component(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "behavior_env_log",
            "ts_wall_s": 20.0,
            "payload": {"data": {"request_id": 5, "phase": "env_processor", "status": "ok", "summary": "desk state stable"}},
        }
    )
    writer.close()

    out = query_session_db(str(session_dir), query="kind:joined component:env_processor", limit=5)
    joined = out.get("joined")
    assert isinstance(joined, list)
    assert len(joined) >= 1
    assert joined[0].get("component") == "env_processor"
    assert out.get("events") == []


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


def test_reasoning_normalization_behavior_trace_event():
    msg = {
        "source": "behavior_trace_log",
        "ts_wall_s": 123.9,
        "payload": {
            "mode": "engage",
            "decision": {"committed": False, "reason": "planner_timeout"},
            "signals": {"zone_hint": "left", "env_delta": 0.72},
            "current_action": {"primitive": "hold", "confidence": 0.61},
            "top_candidates": [{"primitive": "glance", "intent": "scan", "utility": 0.44}],
        },
    }
    out = normalize_reasoning_message(msg)
    assert out is not None
    assert out.source == "behavior_trace_log"
    assert out.phase == "engage"
    assert out.status == "no_commit"
    assert out.component == "arbiter"
    assert out.target_zone == "left"
    assert out.delta_score == 0.72
    assert out.primitive == "hold"


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


def test_trace_graph_ingests_behavior_trace_events():
    b = TraceGraphBuilder(match_window_s=2.0, max_events=100)
    b.ingest(
        {
            "seq": 1,
            "source": "behavior_trace_log",
            "ts_wall_s": 30.0,
            "payload": {
                "mode": "engage",
                "decision": {"committed": True, "reason": "commit_remote"},
                "top_candidates": [{"primitive": "move_to", "intent": "orient", "utility": 0.88}],
            },
        }
    )
    traces = b.traces()
    assert traces
    ref = traces[0].event_refs[0]
    assert ref.source == "behavior_trace_log"
    assert ref.status == "committed"


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


def test_trace_graph_roundtrip_preserves_event_ref_metadata(tmp_path):
    b = TraceGraphBuilder(match_window_s=2.0, max_events=100)
    b.ingest(
        {
            "seq": 1,
            "source": "journal",
            "ts_wall_s": 1.0,
            "payload": {"line": "orchestrator timeout request_id=77 error path"},
            "level": "warning",
        }
    )
    path = tmp_path / "trace_index.json"
    path.write_text(json.dumps(b.build_trace_index()), encoding="utf-8")
    traces = load_trace_index(str(path))
    assert traces
    ref = traces[0].event_refs[0]
    assert ref.req_id == 77
    assert ref.severity == "error"
    assert ref.summary


def test_capture_rebuilds_trace_index_from_full_session_events(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0, trace_max_events=64))
    for req_id in range(70):
        writer.write(
            {
                "type": "event",
                "source": "timeline_log",
                "ts_wall_s": float(req_id),
                "payload": {"data": {"type": "req_end", "payload": {"request_id": req_id, "status": "ok"}}},
            }
        )
    writer.close()
    traces = load_trace_index(str(session_dir / "trace_index.json"))
    req_ids = {trace.req_id for trace in traces}
    assert 0 in req_ids
    assert 69 in req_ids
    assert len(req_ids) == 70


def test_query_session_db_supports_latency_ts_and_sort(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": 1.0,
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 1, "status": "ok", "latency_ms": 150}}},
        }
    )
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": 2.5,
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 2, "status": "parse_fail", "latency_ms": 3100}}},
        }
    )
    writer.close()

    out = query_session_db(str(session_dir), query="kind:event latency_ms>1000 sort:latency order:desc", limit=10)
    assert out["events"]
    assert out["events"][0]["req_id"] == 2
    out_ts = query_session_db(str(session_dir), query="kind:event ts:[2.0,3.0]", limit=10)
    assert out_ts["events"]
    assert out_ts["events"][0]["req_id"] == 2
    out_or = query_session_db(str(session_dir), query="kind:event status:ok|parse_fail", limit=10)
    assert len(out_or["events"]) == 2


def test_annotations_append_and_load(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    append_annotation(str(session_dir), {"tag": "bookmark", "trace_id": "req:7", "req_id": 7, "note": "interesting failure"})
    rows = load_annotations(str(session_dir))
    assert rows
    assert rows[-1]["trace_id"] == "req:7"
    all_rows = load_annotations(str(session_dir), limit=0)
    assert len(all_rows) >= 1


def test_annotation_key_includes_event_index():
    a = {"tag": "bookmark", "trace_id": "req:1", "req_id": 1, "event_index": 11, "created_at_wall_s": 1.0}
    b = {"tag": "bookmark", "trace_id": "req:1", "req_id": 1, "event_index": 12, "created_at_wall_s": 1.0}
    assert annotation_key(a) != annotation_key(b)


def test_dashboard_state_annotation_dedup():
    state = DashboardState(host="jetson")
    row = {
        "tag": "bookmark",
        "trace_id": "req:9",
        "req_id": 9,
        "note": "keep this",
        "created_at_wall_s": 100.0,
    }
    assert state.add_annotation(dict(row)) is True
    assert state.add_annotation(dict(row)) is False
    assert len(state.annotations) == 1


def test_dashboard_state_drops_monotonic_from_transport_and_agent():
    state = DashboardState(host="jetson")
    state.apply({"source": "transport_stats", "payload": {"dropped_events": 7}})
    assert state.dropped_events_reported == 7
    state.apply({"source": "agent", "payload": {"dropped_events": 3}})
    assert state.dropped_events_reported == 7
    state.apply({"source": "agent", "payload": {"dropped_events": 11}})
    assert state.dropped_events_reported == 11


def test_dashboard_state_tracks_transport_queue_metrics():
    state = DashboardState(host="jetson")
    state.apply({"source": "transport_stats", "payload": {"queue_depth": 90, "queue_capacity": 100, "dropped_events": 4}})
    assert state.transport_queue_depth == 90
    assert state.transport_queue_capacity == 100
    assert state.transport_queue_utilization == 0.9
    assert state.transport_queue_utilization_peak == 0.9
    assert state.dropped_events_reported == 4
    state.apply({"source": "transport_stats", "payload": {"queue_depth": 20, "queue_capacity": 100, "dropped_events": 2}})
    assert state.transport_queue_utilization == 0.2
    assert state.transport_queue_utilization_peak == 0.9
    assert state.dropped_events_reported == 4
    assert state.rx_rate_5s > 0.0


def test_load_timeline_rows_requires_v3_schema(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    out = load_timeline_rows(
        session_root=str(session_dir),
        session_db_path=str(session_dir / "session.db"),
        query_text="",
        selected_trace_id="",
        limit=20,
        index_mode="sqlite",
    )
    assert out.source == "unavailable"
    assert out.unavailable_reason == "schema_version=2"
    assert "migrate_session" in out.note


def test_load_timeline_rows_from_sqlite_joined(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": 1.0,
            "payload": {
                "data": {
                    "type": "req_end",
                    "payload": {
                        "request_id": 7,
                        "status": "parse_fail",
                        "latency_ms": 2400,
                        "reasoning": "planner timed out while parsing",
                    },
                }
            },
        }
    )
    writer.close()
    out = load_timeline_rows(
        session_root=str(session_dir),
        session_db_path=resolve_session_db_path(str(session_dir)),
        query_text="status:parse_fail",
        selected_trace_id="",
        limit=20,
        index_mode="sqlite",
    )
    assert out.source == "sqlite.cases.v4"
    assert out.total_count >= len(out.rows)
    assert len(out.rows) >= 1
    assert out.rows[0].provenance == "sqlite.cases.v4"


def test_integrity_report_write_and_verify(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "events.jsonl").write_text("{\"source\":\"x\"}\n", encoding="utf-8")
    report = build_integrity_report(str(session_dir), files=["events.jsonl"])
    write_integrity_report(str(session_dir), report)
    verify_ok = verify_integrity_report(str(session_dir))
    assert verify_ok["ok"] is True
    (session_dir / "events.jsonl").write_text("{\"source\":\"y\"}\n", encoding="utf-8")
    verify_bad = verify_integrity_report(str(session_dir))
    assert verify_bad["ok"] is False
    assert verify_bad["mismatch"] == ["events.jsonl"]


def test_dataset_export_profiles_emit_manifest(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": 10.0,
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 1, "status": "ok", "latency_ms": 150}}},
        }
    )
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": 11.0,
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 2, "status": "parse_fail", "latency_ms": 3200}}},
        }
    )
    writer.close()

    queried = query_cases_db(str(session_dir), query="", limit=10)
    case_id = str(queried["cases"][0].get("case_id") or "")
    review_case(str(session_dir), case_id=case_id, decision="accept", reviewer="pytest")

    out = export_dataset_rows(str(session_dir), profile="strict")
    assert out["row_count"] >= 1
    manifest_path = pathlib.Path(out["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile"] == "strict"
    assert manifest["row_granularity"] == "case"


def test_dataset_export_includes_annotation_linked_rows(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": 1.0,
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 55, "status": "ok", "latency_ms": 120}}},
        }
    )
    writer.close()
    append_annotation(
        str(session_dir),
        {
            "tag": "bookmark",
            "trace_id": "req:55",
            "req_id": 55,
            "note": "good reference",
        },
    )

    out = export_dataset_rows(str(session_dir), profile="hard_cases", include_unlabeled=False)
    assert out["row_count"] == 1
    row_lines = (session_dir / "dataset_rows.jsonl").read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(row_lines[0])
    assert row["trace_id"] == "req:55"
    assert row["case_id"].startswith("case:")
    assert row["source"] == "sqlite.cases.v4"
    assert isinstance(row["provenance_refs"].get("event_indices"), list)
    assert isinstance(row["reasoning_context"], list)
    assert row["annotations"]
    assert row["annotations"][0]["tag"] == "bookmark"
    assert "annotation" in row["inclusion_reasons"]
    manifest = json.loads((session_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["annotated_row_count"] == 1
    assert manifest["annotation_count"] >= 1
    assert manifest["annotation_coverage_ratio"] > 0.0
    assert manifest["source_counts"].get("timeline_log") == 1
    assert manifest["status_counts"].get("ok") == 1
    assert manifest["inclusion_reason_counts"].get("annotation") == 1


def test_dataset_export_strict_requires_review_even_with_annotation(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": 1.0,
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 88, "status": "ok", "latency_ms": 120}}},
        }
    )
    writer.close()
    append_annotation(
        str(session_dir),
        {
            "tag": "bookmark",
            "trace_id": "req:88",
            "req_id": 88,
            "note": "include this in strict curation",
        },
    )

    out = export_dataset_rows(str(session_dir), profile="strict", include_unlabeled=False)
    assert out["row_count"] == 0

    queried = query_cases_db(str(session_dir), query="", limit=10)
    case_id = str(queried["cases"][0].get("case_id") or "")
    review_case(str(session_dir), case_id=case_id, decision="accept", reviewer="pytest")

    out_reviewed = export_dataset_rows(str(session_dir), profile="strict", include_unlabeled=False)
    assert out_reviewed["row_count"] == 1
    row = json.loads((session_dir / "dataset_rows.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
    assert row["review_decision"] == "accept"
    assert "annotation" in row["inclusion_reasons"]
    manifest = json.loads((session_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["annotated_row_count"] == 1
    assert manifest["reviewed_row_count"] == 1


def test_run_curation_export_helper(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": 1.0,
            "payload": {"data": {"type": "req_end", "payload": {"request_id": 9, "status": "parse_fail", "latency_ms": 2800}}},
        }
    )
    writer.close()
    out = _run_curation_export(session_dir=str(session_dir), profile="hard_cases")
    assert out["ok"] is True
    assert out["row_count"] >= 1
    assert pathlib.Path(out["output_path"]).exists()


def test_run_curation_export_helper_missing_dir(tmp_path):
    out = _run_curation_export(session_dir=str(tmp_path / "missing"), profile="hard_cases")
    assert out["ok"] is False
    assert "session directory unavailable" in str(out["error"])


def test_run_curation_export_helper_zero_rows_fails(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    out = _run_curation_export(session_dir=str(session_dir), profile="hard_cases")
    assert out["ok"] is False
    assert "session db missing" in str(out["error"])


def test_build_and_write_viewer_summary(tmp_path):
    state = DashboardState(host="jetson")
    state.started_wall_s = max(0.0, time.time() - 2.0)
    state.event_counts["timeline_log"] = 3
    state.dropped_events_reported = 2
    state.local_dropped_events = 1
    state.transport_queue_utilization_peak = 0.82
    state.transport_queue_alerts = 3
    state.local_queue_utilization_peak = 0.33
    state.local_queue_alerts = 1
    state.rx_rate_peak_5s = 17.25
    state.reconnect_total = 4
    state.reconnect_stale = 2
    state.reconnect_disconnect = 1
    state.reconnect_start_fail = 1
    state.quality_report = {"grade": "pass", "score": 91.5}
    state.quality_gate_passed = True
    state.quality_gate_note = "strict pass"
    state.timeline_source = "sqlite.cases.v4"
    state.timeline_total_rows = 8
    summary = _build_viewer_summary(
        mode="curate",
        state=state,
        query_text="status:parse_fail",
        quality_gate="strict",
        curation_result={"ok": True, "row_count": 4},
        exit_code=0,
    )
    path = _write_viewer_summary(session_dir=str(tmp_path / "session"), summary=summary)
    assert path.endswith("viewer_summary.json")
    loaded = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    assert loaded["mode"] == "curate"
    assert loaded["version"] == 4
    assert loaded["schema_version"] == 3
    assert loaded["run_id"]
    assert loaded["event_counts"]["timeline_log"] == 3
    assert loaded["event_count_total"] == 3
    assert loaded["session_duration_s"] >= 1.0
    assert loaded["dropped_events_agent"] == 2
    assert loaded["dropped_events_local"] == 1
    assert loaded["transport_queue_peak_utilization"] == 0.82
    assert loaded["transport_queue_alert_count"] == 3
    assert loaded["local_queue_peak_utilization"] == 0.33
    assert loaded["local_queue_alert_count"] == 1
    assert loaded["rx_rate_peak_5s"] == 17.25
    assert loaded["reconnect_total"] == 4
    assert loaded["reconnect_stale"] == 2
    assert loaded["reconnect_disconnect"] == 1
    assert loaded["reconnect_start_fail"] == 1
    assert loaded["quality_grade"] == "pass"
    assert loaded["quality_score"] == 91.5
    assert loaded["case_source"] == "sqlite.cases.v4"
    assert loaded["case_rows_total"] == 8
    assert loaded["case_rows_visible"] == 0
    assert loaded["case_unavailable_reason"] == ""
    assert loaded["exit_code"] == 0


def test_append_viewer_run_history(tmp_path):
    session_dir = tmp_path / "session"
    first = {"run_id": "run-1", "mode": "live", "exit_code": 0}
    second = {"run_id": "run-2", "mode": "curate", "exit_code": 2}
    path = _append_viewer_run(session_dir=str(session_dir), summary=first)
    assert path.endswith("viewer_runs.jsonl")
    _append_viewer_run(session_dir=str(session_dir), summary=second)
    rows = [
        json.loads(line)
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert rows[0]["run_id"] == "run-1"
    assert rows[1]["run_id"] == "run-2"


def test_resolve_live_save_session_dir_auto_on_curate():
    path, note = _resolve_live_save_session_dir(
        replay_mode=False,
        save_session_dir="",
        curate_on_exit=True,
        now_wall_s=1700000000.0,
    )
    assert path.startswith("logs/telemetry/session_curate_")
    assert "auto save_session" in note


def test_resolve_live_save_session_dir_expands_user_path():
    home = pathlib.Path.home()
    path, note = _resolve_live_save_session_dir(
        replay_mode=False,
        save_session_dir="~/pala_telemetry_test",
        curate_on_exit=False,
        now_wall_s=1700000000.0,
    )
    assert path.startswith(str(home))
    assert note == ""


def test_resolve_live_save_session_dir_preserves_existing():
    path, note = _resolve_live_save_session_dir(
        replay_mode=False,
        save_session_dir="logs/telemetry/custom",
        curate_on_exit=True,
        now_wall_s=1700000000.0,
    )
    assert path == "logs/telemetry/custom"
    assert note == ""


def test_paths_equivalent_detects_same_directory(tmp_path):
    target = tmp_path / "session"
    target.mkdir(parents=True, exist_ok=True)
    left = str(target)
    right = str(target / ".." / "session")
    assert _paths_equivalent(left, right) is True


def test_doctor_session_dir_flags_missing_v3_artifacts(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "events.jsonl").write_text("{}", encoding="utf-8")
    (session_dir / "manifest.json").write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    checks = _check_session_dir(str(session_dir))
    summary = {check.name: check for check in checks}
    assert summary["session:v3_artifacts"].status == "fail"


def test_doctor_session_dir_invalid_schema_value_warns(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "events.jsonl").write_text("{}", encoding="utf-8")
    (session_dir / "manifest.json").write_text(json.dumps({"schema_version": "nope"}), encoding="utf-8")
    checks = _check_session_dir(str(session_dir))
    summary = {check.name: check for check in checks}
    assert summary["session:schema_version"].status == "warn"


def test_doctor_session_dir_invalid_viewer_runs_fails(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "events.jsonl").write_text("{}", encoding="utf-8")
    (session_dir / "manifest.json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    (session_dir / "viewer_runs.jsonl").write_text("{\"run_id\":\"ok\"}\nnot-json\n", encoding="utf-8")
    checks = _check_session_dir(str(session_dir))
    summary = {check.name: check for check in checks}
    assert summary["session:viewer_runs.parse"].status == "fail"
    assert summary["session:case_explorer.source"].status == "warn"
    assert summary["session:case_explorer.ready"].status == "warn"
    assert summary["session:viewer_runs.health"].status == "pass"


def test_doctor_session_dir_viewer_latest_mismatch_warns(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "events.jsonl").write_text("{}", encoding="utf-8")
    (session_dir / "manifest.json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    (session_dir / "viewer_summary.json").write_text(json.dumps({"run_id": "summary-run"}), encoding="utf-8")
    (session_dir / "viewer_runs.jsonl").write_text("{\"run_id\":\"other-run\"}\n", encoding="utf-8")
    checks = _check_session_dir(str(session_dir))
    summary = {check.name: check for check in checks}
    assert summary["session:viewer_summary.parse"].status == "pass"
    assert summary["session:viewer_runs.parse"].status == "pass"
    assert summary["session:viewer_runs.latest_match"].status == "warn"
    assert summary["session:case_explorer.source"].status == "warn"
    assert summary["session:case_explorer.ready"].status == "warn"
    assert summary["session:viewer_runs.health"].status == "pass"


def test_doctor_session_dir_stream_checks_warn_on_pressure(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "events.jsonl").write_text("{}", encoding="utf-8")
    (session_dir / "manifest.json").write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    (session_dir / "viewer_runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "mode": "live",
                "exit_code": 0,
                "case_source": "sqlite.cases.v4",
                "transport_queue_peak_utilization": 0.95,
                "reconnect_total": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checks = _check_session_dir(str(session_dir))
    summary = {check.name: check for check in checks}
    assert summary["session:stream.queue_peak"].status == "warn"
    assert summary["session:stream.reconnects"].status == "warn"


def test_doctor_session_dir_case_explorer_passes_with_sqlite_source(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "events.jsonl").write_text("{}", encoding="utf-8")
    (session_dir / "manifest.json").write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    (session_dir / "viewer_runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "mode": "replay",
                "exit_code": 0,
                "case_source": "sqlite.cases.v4",
                "case_unavailable_reason": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checks = _check_session_dir(str(session_dir))
    summary = {check.name: check for check in checks}
    assert summary["session:case_explorer.source"].status == "pass"
    assert summary["session:case_explorer.ready"].status == "pass"


def test_build_run_report_aggregates_runs_and_summary_fallback(tmp_path):
    session_a = tmp_path / "session_a"
    session_b = tmp_path / "session_b"
    session_a.mkdir(parents=True, exist_ok=True)
    session_b.mkdir(parents=True, exist_ok=True)
    (session_a / "viewer_runs.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "a1",
                        "mode": "live",
                        "exit_code": 0,
                        "session_duration_s": 5.0,
                        "quality_score": 92.0,
                        "quality_gate_passed": True,
                        "dropped_events_agent": 2,
                        "dropped_events_local": 1,
                        "case_source": "sqlite.cases.v4",
                    }
                ),
                json.dumps(
                    {
                        "run_id": "a2",
                        "mode": "curate",
                        "exit_code": 2,
                        "session_duration_s": 9.0,
                        "quality_score": 76.0,
                        "quality_gate_passed": False,
                        "dropped_events_agent": 5,
                        "dropped_events_local": 3,
                        "curation_result": {"ok": True},
                        "case_source": "sqlite.cases.v4",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (session_b / "viewer_summary.json").write_text(
        json.dumps(
            {
                "run_id": "b1",
                "mode": "replay",
                "exit_code": 0,
                "session_duration_s": 7.0,
                "quality_score": 88.0,
                "quality_gate_passed": True,
                "dropped_events_agent": 4,
                "dropped_events_local": 2,
                "case_source": "sqlite.cases.v4",
                "ended_at_wall_s": 100.0,
            }
        ),
        encoding="utf-8",
    )
    report = build_run_report(session_dirs=[str(session_a), str(session_b)])
    assert report["sessions_scanned"] == 2
    assert report["sessions_with_runs"] == 2
    assert report["runs_total"] == 3
    assert report["mode_counts"]["live"] == 1
    assert report["mode_counts"]["curate"] == 1
    assert report["mode_counts"]["replay"] == 1
    assert report["exit_code_counts"]["0"] == 2
    assert report["exit_code_counts"]["2"] == 1
    assert report["exit_code_sample_count"] == 3
    assert report["exit_nonzero_rate"] == (1.0 / 3.0)
    assert report["quality_gate_sample_count"] == 3
    assert report["quality_gate_pass_rate"] == (2.0 / 3.0)
    assert report["quality_gate_fail_rate"] == (1.0 / 3.0)
    assert report["curation_sample_count"] == 1
    assert report["curation_success_rate"] == 1.0
    assert report["curation_fail_rate"] == 0.0
    assert report["duration_s"]["p50"] == 7.0
    assert report["quality_score"]["p50"] == 88.0
    assert report["drops"]["agent_total"] == 11.0
    assert report["drops"]["local_total"] == 6.0
    assert report["cases"]["source_counts"]["sqlite.cases.v4"] == 3
    assert report["cases"]["unavailable_count"] == 0
    assert report["alerts"] == []
    assert report["alerts_count"] == 0
    assert report["health"] == "ok"
    assert report["latest_run"]["run_id"] == "b1"
    limited = build_run_report(session_dirs=[str(session_a), str(session_b)], limit=1)
    assert limited["runs_total"] == 1
    assert limited["latest_run"]["run_id"] == "b1"
    assert limited["mode_counts"] == {"replay": 1}
    assert limited["exit_nonzero_rate"] == 0.0
    curate_only = build_run_report(session_dirs=[str(session_a), str(session_b)], mode_filter="curate")
    assert curate_only["runs_total"] == 1
    assert curate_only["mode_counts"] == {"curate": 1}
    assert curate_only["quality_gate_fail_rate"] == 1.0


def test_build_run_report_alerts_stream_health_latest(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_id": "run-ok",
            "mode": "live",
            "exit_code": 0,
            "ended_at_wall_s": 1.0,
            "quality_score": 90.0,
            "quality_gate_passed": True,
            "dropped_events_agent": 0,
            "dropped_events_local": 0,
            "case_source": "sqlite.cases.v4",
            "transport_queue_peak_utilization": 0.2,
            "local_queue_peak_utilization": 0.1,
            "reconnect_total": 0,
            "reconnect_stale": 0,
            "rx_rate_peak_5s": 7.0,
        },
        {
            "run_id": "run-hot",
            "mode": "live",
            "exit_code": 0,
            "ended_at_wall_s": 2.0,
            "quality_score": 89.0,
            "quality_gate_passed": True,
            "dropped_events_agent": 1,
            "dropped_events_local": 1,
            "case_source": "sqlite.cases.v4",
            "transport_queue_peak_utilization": 0.92,
            "local_queue_peak_utilization": 0.87,
            "reconnect_total": 4,
            "reconnect_stale": 1,
            "reconnect_disconnect": 2,
            "reconnect_start_fail": 1,
            "rx_rate_peak_5s": 33.0,
        },
    ]
    (session_dir / "viewer_runs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = build_run_report(session_dirs=[str(session_dir)])
    alerts = report["alerts"]
    assert any(text.startswith("latest_transport_queue_pressure_high:") for text in alerts)
    assert any(text.startswith("latest_local_queue_pressure_high:") for text in alerts)
    assert "latest_reconnect_churn:4" in alerts
    assert "latest_stale_reconnects:1" in alerts
    assert report["stream_health"]["transport_queue_peak_max"] == 0.92
    assert report["stream_health"]["local_queue_peak_max"] == 0.87
    assert report["stream_health"]["reconnect_total_p95"] == 4.0
    assert report["stream_health"]["rx_rate_peak_max"] == 33.0


def test_build_run_report_alerts_live_low_activity(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_id": "run-0",
            "mode": "live",
            "exit_code": 0,
            "ended_at_wall_s": 1.0,
            "session_duration_s": 10.0,
            "quality_score": 90.0,
            "quality_gate_passed": True,
            "case_source": "sqlite.cases.v4",
            "event_count_total": 200,
            "rx_rate_peak_5s": 5.0,
        },
        {
            "run_id": "run-low",
            "mode": "live",
            "exit_code": 0,
            "ended_at_wall_s": 2.0,
            "session_duration_s": 60.0,
            "quality_score": 90.0,
            "quality_gate_passed": True,
            "case_source": "sqlite.cases.v4",
            "event_count_total": 8,
            "rx_rate_peak_5s": 0.2,
        },
    ]
    (session_dir / "viewer_runs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = build_run_report(session_dirs=[str(session_dir)])
    alerts = report["alerts"]
    assert any(text.startswith("latest_live_low_activity_events:") for text in alerts)
    assert any(text.startswith("latest_live_low_rx_peak:") for text in alerts)
    latest = report.get("latest_run")
    assert isinstance(latest, dict)
    assert latest.get("event_count_total") == 8


def test_build_run_report_detects_latest_regressions(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(4):
        rows.append(
            {
                "run_id": f"run-{idx}",
                "mode": "live",
                "exit_code": 0,
                "ended_at_wall_s": float(idx + 1),
                "quality_score": 92.0,
                "quality_gate_passed": True,
                "dropped_events_agent": 2,
                "dropped_events_local": 2,
                "case_source": "sqlite.cases.v4",
            }
        )
    rows.append(
        {
            "run_id": "run-bad",
            "mode": "live",
            "exit_code": 2,
            "ended_at_wall_s": 10.0,
            "quality_score": 65.0,
            "quality_gate_passed": False,
            "dropped_events_agent": 44,
            "dropped_events_local": 42,
            "case_source": "unavailable",
            "case_unavailable_reason": "session_db_missing",
        }
    )
    (session_dir / "viewer_runs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = build_run_report(session_dirs=[str(session_dir)])
    assert report["health"] == "warn"
    alerts = report["alerts"]
    assert "latest_exit_code_nonzero:2" in alerts
    assert "latest_quality_gate_failed" in alerts
    assert "latest_case_unavailable:session_db_missing" in alerts
    assert any(text.startswith("quality_score_regression:") for text in alerts)
    assert any(text.startswith("agent_drop_spike:") for text in alerts)
    assert any(text.startswith("local_drop_spike:") for text in alerts)
    assert report["alerts_count"] >= 5
    assert report["exit_nonzero_rate"] == (1.0 / 5.0)
    assert report["quality_gate_fail_rate"] == (1.0 / 5.0)


def test_build_run_report_alerts_no_run_artifacts(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    report = build_run_report(session_dirs=[str(session_dir)])
    assert report["runs_total"] == 0
    assert report["alerts"] == ["no_run_artifacts"]
    assert report["alerts_count"] == 1
    assert report["health"] == "warn"


def test_build_run_report_alerts_sessions_missing_runs(tmp_path):
    session_ok = tmp_path / "session_ok"
    session_empty = tmp_path / "session_empty"
    session_ok.mkdir(parents=True, exist_ok=True)
    session_empty.mkdir(parents=True, exist_ok=True)
    (session_ok / "viewer_summary.json").write_text(
        json.dumps({"run_id": "ok-1", "mode": "live", "exit_code": 0}),
        encoding="utf-8",
    )
    (session_empty / "manifest.json").write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    report = build_run_report(session_dirs=[str(session_ok), str(session_empty)])
    assert report["runs_total"] == 1
    assert report["sessions_without_runs_count"] == 1
    assert "sessions_missing_runs:1" in report["alerts"]
    assert report["health"] == "warn"


def test_build_run_report_tracks_sessions_without_runs(tmp_path):
    session_ok = tmp_path / "session_ok"
    session_empty = tmp_path / "session_empty"
    session_ok.mkdir(parents=True, exist_ok=True)
    session_empty.mkdir(parents=True, exist_ok=True)
    (session_ok / "viewer_summary.json").write_text(
        json.dumps({"run_id": "ok-1", "mode": "live", "exit_code": 0}),
        encoding="utf-8",
    )
    report = build_run_report(session_dirs=[str(session_ok), str(session_empty)])
    assert report["sessions_scanned"] == 2
    assert report["sessions_with_runs"] == 1
    assert report["sessions_without_runs_count"] == 1
    assert len(report["sessions_without_runs"]) == 1
    assert report["sessions_without_runs"][0] == str(session_empty)
    assert "sessions_missing_runs:1" in report["alerts"]


def test_build_run_report_includes_case_review_coverage_from_db(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": 1.0,
            "payload": {
                "data": {
                    "type": "req_end",
                    "payload": {
                        "request_id": 42,
                        "status": "parse_fail",
                        "latency_ms": 1200,
                        "reasoning": "planner parse_fail",
                    },
                }
            },
        }
    )
    writer.close()

    queried = query_cases_db(str(session_dir), query="", limit=10)
    assert queried["cases"]
    case_id = str(queried["cases"][0].get("case_id") or "")
    assert case_id
    review_case(str(session_dir), case_id=case_id, decision="accept", reviewer="pytest")

    (session_dir / "viewer_summary.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "mode": "curate",
                "exit_code": 0,
                "ended_at_wall_s": 10.0,
                "quality_score": 90.0,
                "quality_gate_passed": True,
                "case_source": "sqlite.cases.v4",
            }
        ),
        encoding="utf-8",
    )

    report = build_run_report(session_dirs=[str(session_dir)])
    assert report["cases"]["total_cases"] >= 1
    assert report["cases"]["reviewed_cases"] >= 1
    assert report["cases"]["decision_counts"].get("accept", 0) >= 1
    assert report["cases"]["review_coverage"] is not None
    assert report["latest_run"]["case_review_coverage"] is not None


def test_doctor_session_dir_case_review_coverage_passes(tmp_path):
    session_dir = tmp_path / "session"
    writer = SessionCaptureWriter(CaptureConfig(directory=str(session_dir), frames_mode="off", max_seconds=0.0))
    writer.write(
        {
            "type": "event",
            "source": "timeline_log",
            "ts_wall_s": 1.0,
            "payload": {
                "data": {
                    "type": "req_end",
                    "payload": {
                        "request_id": 99,
                        "status": "parse_fail",
                        "latency_ms": 900,
                    },
                }
            },
        }
    )
    writer.close()

    queried = query_cases_db(str(session_dir), query="", limit=10)
    case_id = str(queried["cases"][0].get("case_id") or "")
    review_case(str(session_dir), case_id=case_id, decision="accept", reviewer="pytest")

    checks = _check_session_dir(str(session_dir))
    summary = {item.name: item for item in checks}
    assert summary["session:case_reviews.coverage"].status == "pass"


def test_doctor_session_dir_run_report_alerts_warn(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "events.jsonl").write_text("{}", encoding="utf-8")
    (session_dir / "manifest.json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    rows = [
        {
            "run_id": "good",
            "mode": "live",
            "exit_code": 0,
            "ended_at_wall_s": 1.0,
            "quality_score": 93.0,
            "quality_gate_passed": True,
            "dropped_events_agent": 1,
            "dropped_events_local": 1,
            "case_source": "sqlite.cases.v4",
        },
        {
            "run_id": "bad",
            "mode": "live",
            "exit_code": 2,
            "ended_at_wall_s": 2.0,
            "quality_score": 60.0,
            "quality_gate_passed": False,
            "dropped_events_agent": 40,
            "dropped_events_local": 40,
            "case_source": "unavailable",
            "case_unavailable_reason": "schema_version=2",
        },
    ]
    (session_dir / "viewer_runs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    checks = _check_session_dir(str(session_dir))
    summary = {check.name: check for check in checks}
    assert summary["session:case_explorer.source"].status == "warn"
    assert summary["session:case_explorer.ready"].status == "warn"
    assert summary["session:viewer_runs.health"].status == "warn"
    assert summary["session:viewer_runs.alerts"].status == "warn"


def test_doctor_session_dir_live_activity_alerts_warn(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "events.jsonl").write_text("{}", encoding="utf-8")
    (session_dir / "manifest.json").write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    rows = [
        {
            "run_id": "run-ok",
            "mode": "live",
            "exit_code": 0,
            "ended_at_wall_s": 1.0,
            "session_duration_s": 10.0,
            "quality_score": 92.0,
            "quality_gate_passed": True,
            "case_source": "sqlite.cases.v4",
            "event_count_total": 120,
            "rx_rate_peak_5s": 4.0,
        },
        {
            "run_id": "run-low",
            "mode": "live",
            "exit_code": 0,
            "ended_at_wall_s": 2.0,
            "session_duration_s": 80.0,
            "quality_score": 90.0,
            "quality_gate_passed": True,
            "case_source": "sqlite.cases.v4",
            "event_count_total": 5,
            "rx_rate_peak_5s": 0.2,
        },
    ]
    (session_dir / "viewer_runs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    checks = _check_session_dir(str(session_dir))
    summary = {item.name: item for item in checks}
    assert summary["session:viewer_runs.health"].status == "warn"
    assert summary["session:viewer_runs.live_activity"].status == "warn"


def test_viewer_help_surface_is_compact():
    parser = _build_parser()
    help_text = parser.format_help()
    options = sorted(set(re.findall(r"--[a-z0-9][a-z0-9-]*", help_text)))
    assert len(options) <= 18
    assert "--mode" in options
    assert "--preset" not in options
