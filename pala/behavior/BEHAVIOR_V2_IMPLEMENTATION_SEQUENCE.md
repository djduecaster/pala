# BehaviorV2 Implementation Sequence

## Phase 0: Skeleton and Contracts
1. Create behavior module skeleton files:
   - `world_state_store.py`
   - `env_processor.py`
   - `planner_client.py`
   - `action_translator.py`
   - `policy.py`
2. Define internal dataclasses for env/planner parsed payloads.
3. Add strict parser helpers for tagged text blocks and `decision_json`.

Exit criteria:
- modules import cleanly
- no side effects at import time

## Phase 1: World State First
1. Implement `WorldStateStore` runtime model with bounded tails.
2. Implement markdown persistence and file bootstrap.
3. Add session digest rewrite strategy (periodic overwrite).

Exit criteria:
- can update and persist state without remote dependencies
- markdown output is stable and compact

## Phase 2: Environment Processor Threadless Client
1. Implement request builder and media-first payload format.
2. Implement non-blocking scheduling:
   - one in-flight
   - latest-only pending
3. Parse tagged env response and write into world state.
4. Log reasoning text to debug file only.

Exit criteria:
- env processor can run independently and update world state
- parse errors do not crash behavior loop

## Phase 3: Planner Client and Translation
1. Implement planner request builder from world state and control state.
2. Implement strict planner response parsing.
3. Translate parsed decisions into typed `ActionPlan`.
4. Ensure action defaults:
   - `cancel_current=False`
   - fresh `action_id`

Exit criteria:
- planner emits valid `ActionPlan` objects
- invalid payloads are rejected safely

## Phase 4: BehaviorPolicy Orchestration
1. Wire frame ingestion, env cadence, planner cadence.
2. Add event-trigger logic from env `delta_score`.
3. Add fallback action policy when no new valid planner action exists.
4. Feed latest control state back into world state each tick.

Exit criteria:
- end-to-end behavior loop produces stable actions
- loop remains responsive under remote latency

## Phase 5: Runtime Integration
1. Switch `pala/main.py` behavior construction to `BehaviorPolicy`.
2. Keep control/hardware loops unchanged.
3. Confirm local dummy-mode startup works.

Exit criteria:
- `uv run python -m pala.main` starts and runs without planner/behavior import errors

## Phase 6: First Validation Pass
1. Run short local smoke.
2. Run Jetson loop test with logging enabled.
3. Evaluate:
   - parse-fail rate
   - latency
   - primitive diversity
   - memory file size/readability

Exit criteria:
- baseline metrics captured
- concrete tuning backlog generated from logs

## Open Questions to Revisit After Phase 1
- add decision TTL/staleness handling
- add stronger action diversity policy
- add localization adapter layer
- add long-horizon memory structure beyond compact digest
