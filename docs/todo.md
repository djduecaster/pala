# PALA TODO

Last updated: 2026-02-07

## Priority Roadmap
1. Servo calibration and safety envelope
   - Validate per-joint min/max, neutral offsets, reverse flags, and clamping.
   - Verify deadman + commanded disable behavior on real hardware.
   - Add/confirm safe fallback behavior for stale/no-perception conditions.

2. Perception truthfulness and Jetson detector tuning
   - Make perception output explicit about detector state (`num_detections`, `used_fallback_bbox`, detector health).
   - Tune DeepStream thresholds/FPS and document lighting sensitivity.
   - Define acceptance criteria (example: fallback ratio under sustained subject presence).

3. Control primitives and motion policy
   - Finalize primitive set for demo (`hold`, `glance`, `ack`, `breath`, optional `track`).
   - Add per-primitive constraints (rate/accel/safety gating).

4. Behavior layer refinement
   - Keep deterministic local policy as baseline.
   - Add confidence/freshness gating so weak perception degrades safely.

5. Cosmos integration via Brev
   - Define strict I/O contract from `PerceptionState` to planner requests.
   - Add timeout/retry and guaranteed local fallback path.

6. Telemetry and evaluation
   - Build reproducible eval harness (JSONL -> metrics summary).
   - Add optional live telemetry preview for debugging/demos.

## Additional Cross-Cutting Items
- Reproducibility checklist for Jetson environment (package versions, one-command verification).
- Performance profile checklist (`nvpmodel`, clocks, thermal notes).
- Tests for safety invariants and contract validation.

## Suggested Next Step
- Start with items 1 and 2 together: safe servo envelope + perception truthfulness.
