# BehaviorV2 Tasks

## Track A: WorldStateStore and Persistence
- [ ] Implement `WorldStateStore` dataclass and in-memory state.
- [ ] Add markdown persistence writer:
  - `memory/world_state.md`
  - `memory/session_digest.md`
- [ ] Load `memory/identity.md` on startup.
- [ ] Add compaction method for `session_digest` rewrite.
- [ ] Add bounded tails (`event_tail`, `decision_tail`) with fixed caps.

Acceptance criteria:
- state updates are deterministic and bounded
- markdown files are rewritten without unbounded growth
- behavior loop can run with missing identity file (safe default)

## Track B: Environment Processor Client
- [ ] Implement `CosmosEnvProcessor` request builder for sampled frame windows.
- [ ] Enforce one in-flight request and latest-only scheduling.
- [ ] Parse tagged response blocks (`scene`, `events`, `hypotheses`, `opportunities`, `uncertainties`, `delta_score`, `summary`).
- [ ] Store raw reasoning into debug log only.
- [ ] Convert parsed output into world state updates.

Acceptance criteria:
- malformed responses fail closed (no crash, no invalid state overwrite)
- parser returns stable fields across prompt variants
- env loop cadence remains stable under remote latency

## Track C: Planner Client and Action Translation
- [ ] Implement `CosmosPlanner` request builder from world state + control state.
- [ ] Enforce one in-flight request and latest-only scheduling.
- [ ] Parse `<decision_json>` and `<rationale_short>`.
- [ ] Validate primitive and command payload via `ActionPlan` types.
- [ ] Always emit `cancel_current=False` in phase 1.
- [ ] Generate unique `action_id` for accepted decisions.

Acceptance criteria:
- strict parser rejects invalid decisions
- valid decisions translate to executable `ActionPlan`
- no duplicate terminal action-id suppression regressions

## Track D: BehaviorPolicy Loop Integration
- [ ] Implement `BehaviorPolicy.step(...)` orchestration:
  - frame ingestion
  - env tick scheduling
  - planner tick scheduling and event-trigger
  - world state updates
  - action emission
- [ ] Add fallback behavior when planner has no valid decision.
- [ ] Surface minimal runtime counters for visibility.

Acceptance criteria:
- behavior tick never blocks on remote calls
- no thread crashes on parse/network failures
- output remains valid `ActionPlan` every tick

## Track E: Prompt Packs and Policy Text
- [ ] Add env processor prompt template (observer-scoped identity).
- [ ] Add planner prompt template (full identity + capability policy).
- [ ] Keep system prompt minimal and user payload rich.
- [ ] Ensure media-first ordering in payload content.

Acceptance criteria:
- templates are readable/editable and versioned
- planner prompt contains primitive contract and strict response format
- env prompt explicitly forbids action planning

## Track F: Logging and Evaluation Hooks
- [ ] Add behavior logs:
  - `logs/behavior_env.jsonl`
  - `logs/behavior_planner.jsonl`
  - `logs/behavior_reasoning.jsonl`
- [ ] Log parse-fails, latency, response status, action diversity counters.
- [ ] Add lightweight report helper script under `tools/` (optional in this phase).

Acceptance criteria:
- one run yields enough data to evaluate collapse/latency/parse quality
- reasoning is present in debug logs but absent from planner memory inputs
