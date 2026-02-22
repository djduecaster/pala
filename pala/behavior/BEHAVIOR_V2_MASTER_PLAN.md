# BehaviorV2 Master Plan

## Goal
Build a Cosmos-first behavior layer that turns scene understanding into reliable, expressive `ActionPlan` outputs for control, while keeping memory lightweight and stable.

## Scope
- In scope: `pala/behavior` implementation and behavior-side data contracts.
- Out of scope for this phase: control-layer redesign, full localization subsystem, local detector-driven semantics.

## Locked Decisions
1. Two remote Cosmos threads:
   - `Environment Processor`
   - `Planner`
2. One local behavior memory system:
   - `WorldStateStore` (same concept as memory)
3. Frames-first input:
   - sampled frames over rolling 5-10s window
4. Concurrency:
   - max one in-flight request per Cosmos thread
   - latest-only semantics (drop stale pending work)
5. Planner output:
   - strict execution object for parser/translator
   - reasoning stays debug-only
6. Action semantics:
   - actions complete by default
   - no planner cancel semantics in phase 1
7. No decision TTL/expiry logic in phase 1.
8. Keep current primitive library unchanged.
9. Memory persistence:
   - markdown files for state and identity
   - jsonl debug logs for traces

## Architecture
- `BehaviorPolicy` (orchestrates behavior loop tick)
- `RollingFrameWindow` (behavior-owned sampled frame history)
- `CosmosEnvProcessor` (scene and event extraction)
- `WorldStateStore` (local, persistent, compact memory)
- `CosmosPlanner` (decision generation from world state)
- `ActionTranslator` (strict parse/validation to `ActionPlan`)

## Data Flow
1. Perception writes frames into behavior frame window.
2. Env processor runs at ~1.0 Hz:
   - reads sampled frame window
   - returns tagged rich text
   - updates world state snapshot and event tail
3. Planner runs at ~0.5 Hz plus event trigger:
   - reads world state, control state, primitive catalog, optional single frame
   - returns strict decision object
   - translator emits `ActionPlan`
4. Control executes action and exposes latest control state.
5. Behavior writes control state updates into world state.

## Contract: Environment Processor Output
Required tags:
- `<scene>...</scene>`
- `<events>...</events>`
- `<hypotheses>...</hypotheses>`
- `<opportunities>...</opportunities>`
- `<uncertainties>...</uncertainties>`
- `<delta_score>0.0-1.0</delta_score>`
- `<summary>...</summary>`

Optional tag:
- `<think>...</think>` (debug-only; never fed into planner memory)

## Contract: Planner Output
Required tags:
- `<decision_json>{...}</decision_json>`
- `<rationale_short>...</rationale_short>`

Optional tag:
- `<think>...</think>` (debug-only; never fed into planner memory)

`decision_json` fields:
- `act_now` (bool)
- `primitive` (existing primitive names only)
- `command` (primitive-specific object)
- `style` (string)
- `confidence` (0..1)

Translator behavior:
- validates primitive/command schema
- sets `cancel_current=False`
- generates fresh `action_id`

## WorldStateStore Footprint
Runtime memory (compact):
- `identity_core`
- `latest_env_snapshot` (overwrite)
- `event_tail` (last 8-12)
- `decision_tail` (last 6)
- `control_state_latest`
- `session_digest` (periodic rewrite)

Persistent files:
- `memory/identity.md`
- `memory/world_state.md`
- `memory/session_digest.md`

Debug logs (append-only, not planner inputs):
- `logs/behavior_env.jsonl`
- `logs/behavior_planner.jsonl`
- `logs/behavior_reasoning.jsonl`

## Triggering and Cadence
- Env processor: ~1.0 Hz.
- Planner: ~0.5 Hz plus event-trigger.
- Event-trigger condition from latest env update:
  - `delta_score >= 0.65` with local cooldown (~0.7s).

## Prompting Rules
- System prompt stays short: `You are a helpful assistant.`
- User message contains policy/capability/context.
- Media blocks appear before text blocks.
- Reasoning mode is enabled, but reasoning is debug-only.

## Metrics (Phase 1)
- planner parse-fail rate < 10%
- stable behavior loop cadence
- non-trivial primitive diversity (no single-action collapse)
- visible env/planner latency telemetry
- compact readable memory files after extended runs

## Phase 1 Non-Goals
- full localization graph
- long-term semantic/vector memory
- planner cancellation policy tuning
- expanded primitive library
