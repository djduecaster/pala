# PALA Architecture

## Runtime Modes
- **dev**: dummy perception + dummy hardware, safe on Mac.
- **jetson_perception**: camera/perception focused mode (still evolving).
- **jetson_full**: full camera + planner + control + actuation on Jetson.

## Four-Loop Runtime
1. **Perception Loop (15-30 Hz)**
   - Reads from frame source.
   - Runs detector.
   - Publishes `PerceptionState`.
   - Updates latest frame cache for remote planner image payloads.

2. **Behavior Loop (2-5 Hz)**
   - Calls planner and emits `ActionPlan`.
   - If planner declares `owns_semantic_behavior=True` (Cosmos orchestrator), behavior layer does not add extra local semantic triggers.

3. **Control Loop (50-100 Hz)**
   - Converts `ActionPlan` to smoothed joint trajectories in `HardwareCommand`.
   - Preserves last commanded joints unless new action changes target.

4. **Hardware Loop (50-200 Hz)**
   - Applies `HardwareCommand` to servo backend.
   - Enforces deadman timeout and safe disable on stale commands.

## Planner Stack (Current)
- **Heuristic planner**: local fallback when Cosmos is disabled.
- **AsyncOrchestratorPlanner (Cosmos)**: remote-first semantic planner.
  - Samples rolling frame window from latest-frame history.
  - Builds media-first chat payload (images before user text).
  - User text includes policy blocks, output contract, and compact context JSON.
  - Expects `<think>...</think>` plus JSON decision payload.
  - Parses plain JSON, fenced JSON, and mixed think+JSON responses.
  - Converts canonical decision to core primitives (`hold`, `breath`, `glance`, `nod`, `orient_to_zone`).

## Context Sent to Cosmos
Current request context JSON includes:
- `control_state` (active primitive, action age, latest accepted decision),
- `transcript_tail` (recent `decision` and `reasoning` lines only),
- `frame_meta` (frames sent and frame age).

Current design intentionally does **not** send local semantic summaries (for example scene summary or belief packets) to remote planning.

## Remote/Local Behavior Contract
- If Cosmos is enabled and reachable, planner runs in remote-first mode.
- No local semantic substitute decision is generated when a remote response fails parse/validation; system keeps prior action (or initial neutral hold).
- Optional reasoning probe runs in a separate low-rate thread for diagnostics; default is disabled.

## Data Contracts
- `PerceptionState` -> planner input.
- `ActionPlan` -> control input.
- `HardwareCommand` -> hardware input.

## Logging and Replay Artifacts
- `logs/perception.jsonl`: perception loop output.
- `logs/actions.jsonl`: behavior output actions.
- `logs/runtime_debug.log`: request lifecycle, parse failures, reasoning previews.
- `logs/orchestrator_timeline.jsonl`: structured orchestrator events:
  - `run_start`, `request_start`, `request_end`, `reasoning_event`, `decision_event`, `fallback_event`.
- `logs/telemetry/preview/latest.jpg` + `latest.json`: optional preview tap for sidecar telemetry tools.

## Forward Plan: Memory Layers
- Near-term planner remains transcript-first.
- Future memory layering plan (spatial/environmental + short-term + long-term) is documented in:
  - `docs/memory_architecture.md`
