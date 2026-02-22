# Behavior V3 Tasks

## Completed in this overhaul
- [x] Remove V2 behavior-specific parser/translator modules.
- [x] Introduce strict JSON schema contracts for env and intent outputs.
- [x] Introduce deterministic arbitration stack (governor + arbiter + compiler).
- [x] Ensure decision memory stores committed actions only.
- [x] Add health manager and breaker-aware scheduling.
- [x] Add deterministic idle heartbeat engine.
- [x] Add per-step trace logging.

## Next iteration
- [ ] Add optional schema-side command `oneOf` tightening for future primitives (`move_to`, `gaze_to`) after field testing.
- [ ] Add offline evaluation script over `behavior_trace.jsonl` to score no-commit streaks and churn.
- [ ] Add dashboard aggregation for planner/env health and latency percentiles.
