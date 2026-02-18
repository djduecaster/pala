from __future__ import annotations

from pala.planner.memory_manager import MemoryManager, MemoryManagerConfig


def test_memory_manager_persists_recent_and_digest(tmp_path):
    path = tmp_path / "memory.jsonl"
    mgr = MemoryManager(
        MemoryManagerConfig(
            enabled=True,
            jsonl_path=str(path),
            recent_events=5,
            digest_items=2,
            distill_every_n_events=3,
        )
    )
    for i in range(4):
        mgr.append_event(
            "decision_event",
            {
                "state": "tracking",
                "zone_hint": "left" if i % 2 == 0 else "right",
                "primitive": "orient_to_zone",
            },
        )

    ctx = mgr.context()
    assert len(ctx["recent_events"]) == 4
    assert len(ctx["session_memory_digest"]) >= 1
    assert path.exists()


def test_memory_manager_loads_existing_file(tmp_path):
    path = tmp_path / "memory.jsonl"
    mgr0 = MemoryManager(
        MemoryManagerConfig(
            enabled=True,
            jsonl_path=str(path),
            recent_events=5,
            digest_items=2,
            distill_every_n_events=2,
        )
    )
    mgr0.append_event("observation_event", {"state": "user_detected", "zone_hint": "center"})
    mgr0.append_event("decision_event", {"state": "engaging", "primitive": "nod"})

    mgr1 = MemoryManager(
        MemoryManagerConfig(
            enabled=True,
            jsonl_path=str(path),
            recent_events=5,
            digest_items=2,
            distill_every_n_events=2,
        )
    )
    ctx = mgr1.context()
    assert len(ctx["recent_events"]) >= 2
