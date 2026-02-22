from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pala.behavior.world_state_store import DecisionSnapshot, WorldStateStore, WorldStateStoreConfig
from pala.perception.preview_tap import PreviewTapWriter


def test_world_state_store_write_failures_are_swallowed(monkeypatch, tmp_path, caplog):
    def _raise_write(self, text, encoding="utf-8"):  # noqa: ARG001
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _raise_write)
    store = WorldStateStore(
        WorldStateStoreConfig(
            identity_path=str(tmp_path / "identity.md"),
            world_state_path=str(tmp_path / "world_state.md"),
            session_digest_path=str(tmp_path / "session_digest.md"),
        )
    )

    with caplog.at_level(logging.WARNING):
        store.append_event("event")
        store.append_decision(
            DecisionSnapshot(
                primitive="hold",
                style="calm",
                confidence=0.2,
                rationale_short="safe",
            )
        )
        store.rewrite_session_digest("digest")
        store.persist()

    assert any("world_state_store write failed" in rec.message for rec in caplog.records)


def test_world_state_store_read_failures_fall_back_to_defaults(monkeypatch, tmp_path, caplog):
    def _raise_read(self, encoding="utf-8"):  # noqa: ARG001
        raise OSError("read blocked")

    monkeypatch.setattr(Path, "read_text", _raise_read)
    identity_path = tmp_path / "identity.md"
    identity_path.write_text("ignored", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        store = WorldStateStore(
            WorldStateStoreConfig(
                identity_path=str(identity_path),
                world_state_path=str(tmp_path / "world_state.md"),
                session_digest_path=str(tmp_path / "session_digest.md"),
            )
        )

    assert "You are PALA" in store.identity_core
    assert any("identity load failed" in rec.message for rec in caplog.records)


def test_preview_tap_logs_invalid_frames_with_warning_throttle(tmp_path, monkeypatch, caplog):
    writer = PreviewTapWriter(
        enabled=True,
        jpeg_path=str(tmp_path / "latest.jpg"),
        meta_path=str(tmp_path / "latest.json"),
        max_hz=30.0,
        max_width=64,
        max_height=64,
        jpeg_quality=70,
    )
    bad_frame = np.zeros((8, 8), dtype=np.uint8)
    times = iter([10.0, 12.0, 16.2])
    monkeypatch.setattr("pala.perception.preview_tap.time.monotonic", lambda: next(times))

    with caplog.at_level(logging.WARNING):
        writer.write(bad_frame, mono_ns=1, pts_ns=None)
        writer.write(bad_frame, mono_ns=2, pts_ns=None)
        writer.write(bad_frame, mono_ns=3, pts_ns=None)

    warns = [rec for rec in caplog.records if "preview tap write failed" in rec.message]
    assert len(warns) == 2
    assert (tmp_path / "latest.jpg").exists() is False
    assert (tmp_path / "latest.json").exists() is False


def test_preview_tap_cleans_temp_files_when_write_fails_midway(tmp_path, monkeypatch):
    writer = PreviewTapWriter(
        enabled=True,
        jpeg_path=str(tmp_path / "latest.jpg"),
        meta_path=str(tmp_path / "latest.json"),
        max_hz=30.0,
        max_width=64,
        max_height=64,
        jpeg_quality=70,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    def _raise_dump(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("metadata encode failed")

    monkeypatch.setattr("pala.perception.preview_tap.json.dump", _raise_dump)
    writer.write(frame, mono_ns=1, pts_ns=None)

    assert list(tmp_path.glob("*.tmp.*")) == []
    assert (tmp_path / "latest.jpg").exists() is False
    assert (tmp_path / "latest.json").exists() is False

