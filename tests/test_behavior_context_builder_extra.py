from __future__ import annotations

from pala.behavior.context_builder import ContextBuilder
from pala.types import ActionPlan, HoldCommand, PrimitiveKind


def _hold_action() -> ActionPlan:
    return ActionPlan(
        primitive=PrimitiveKind.HOLD,
        command=HoldCommand(),
        confidence=0.4,
        style="calm",
    )


def test_context_builder_uses_env_zone_hint_and_no_local_person_conf():
    builder = ContextBuilder()
    world = {
        "latest_env_snapshot": {
            "scene": "desk",
            "summary": "steady",
            "delta_score": 0.3,
            "features": {"zone_hint": "right", "person_present": True},
        }
    }

    ctx = builder.build_planner_context(
        st=None,
        world_snapshot=world,
        current_action=_hold_action(),
        planner_health={"state": "HEALTHY"},
        now_mono_s=10.0,
        last_commit_mono_s=8.0,
        no_commit_s=2.0,
    )
    assert ctx["signals"]["person_conf"] is None
    assert ctx["signals"]["zone_hint"] == "right"
    assert "env:zone:right" in ctx["evidence_index"]["available"]


def test_context_builder_handles_invalid_timestamps_and_format_helpers():
    builder = ContextBuilder()
    world = {
        "latest_env_snapshot": {"summary": "stable"},
        "event_tail": [
            {"timestamp_wall_s": 1_700_000_000.0, "summary": "good"},
            {"timestamp_wall_s": "not-a-float", "summary": "bad"},
        ],
    }

    env_ctx = builder.build_env_context(
        world_snapshot=world,
        current_action=_hold_action(),
        frame_timeline=[],
    )
    assert env_ctx["recent_env_events"][0]["t"] is not None
    assert env_ctx["recent_env_events"][1]["t"] is None
    assert builder._format_ts_seconds("still-bad") is None  # noqa: SLF001


def test_context_builder_command_digest_fallback_and_non_mapping(monkeypatch):
    class _BadCommand:
        def __str__(self) -> str:
            return "bad-command"

    monkeypatch.setattr("pala.behavior.context_builder.to_json_dict", lambda _command: [1, 2, 3])
    assert ContextBuilder._command_digest(object()) == [1, 2, 3]  # noqa: SLF001

    def _raise(_command):
        raise RuntimeError("boom")

    monkeypatch.setattr("pala.behavior.context_builder.to_json_dict", _raise)
    assert ContextBuilder._command_digest(_BadCommand()) == "bad-command"  # noqa: SLF001
