# PALA TODO

Last updated: 2026-02-07

## Priority Roadmap
1. Servo calibration and safety envelope
   - Validate per-joint min/max, neutral offsets, reverse flags, and clamping.
   - Verify deadman + commanded disable behavior on real hardware.
   - Add/confirm safe fallback behavior for stale/no-perception conditions.

2. Perception capture truthfulness
   - Validate camera frame IDs, frame age, source health, and FPS on Jetson.
   - Keep local detection disabled until the direct vision-model loop is stable.
   - Use `pala/perception/DEEPSTREAM_REINTRODUCTION.md` when local facts are added later.

3. Control primitives and motion policy
   - Finalize primitive set for demo (`hold`, `glance`, `ack`, `breath`, optional `track`).
   - Add per-primitive constraints (rate/accel/safety gating).

4. Behavior/planner reset
   - Phase 1 and 2 complete: capture-only perception and V4 behavior removal.
   - TODO: define the Phase 3 model decision contract and execution semantics.

5. Cosmos integration via Brev
   - TODO: re-specify request/response contract after reset.
   - TODO: re-validate timeout/retry strategy after reset.

6. Telemetry and evaluation
   - Build reproducible eval harness (JSONL -> metrics summary).
   - Add optional live telemetry preview for debugging/demos.

## Additional Cross-Cutting Items
- Reproducibility checklist for Jetson environment (package versions, one-command verification).
- Performance profile checklist (`nvpmodel`, clocks, thermal notes).
- Tests for safety invariants and contract validation.

## Suggested Next Step
- Design Phase 3 before writing code: model authority, state ownership, decision
  schema, skill completion semantics, and the smallest vertical slice.
