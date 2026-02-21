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
   - Calls planner to get a candidate `ActionPlan`.
   - Runs a **scene interpreter** (`person_entered`, `person_exited`, `zone_changed`) over local perception.
   - Updates **scene memory** (rolling short-term history of events/zones/activities).
   - **Planner-owned semantics path (default with Cosmos)**:
     - planner output is treated as the semantic source of truth,
     - no local behavior-mode arbitration (`idle|engage|track|assist|reacquire`) is run,
     - local logic is limited to execution safety guardrails before control.
   - **Local semantic stack path (fallback/experiments)**:
     - optional `IntentPlanner` + `ActionRealizer` can be enabled only when planner does not own semantics.
     - this path provides local mode selection + expressive realization heuristics for non-Cosmos runs.
   - Runs an **action governor** that keeps behavior interruptible and avoids persistent-action lockups:
     - Forces cancellation when switching away from persistent primitives (`hold`/`breath`),
     - Injects zone-tracking nudges on meaningful zone transitions,
     - Refreshes prolonged `hold`/`breath` into finite tracking motion when a person is present.
   - Arbitration keeps stable `action_id` only for persistent primitives (`hold`/`breath`); terminating
     primitives receive fresh IDs so they can be legitimately re-triggered.
   - Emits the governed `ActionPlan` to control.

3. **Control Loop (50-100 Hz)**
   - Converts `ActionPlan` to smoothed joint trajectories in `HardwareCommand`.
   - Preserves last commanded joints unless new action changes target.

4. **Hardware Loop (50-200 Hz)**
   - Applies `HardwareCommand` to servo backend.
   - Enforces deadman timeout and safe disable on stale commands.

## Planner Stack (Current)
- **Heuristic planner**: local fallback when Cosmos is disabled.
- **AsyncOrchestratorPlanner (Cosmos)**: remote-first semantic planner.
  - Ingests identity + summary memory + control state first.
  - Can include live frame images on first-pass planning request for direct VLM grounding (configurable).
  - Uses strict action schema parsing for decision output, with tolerant normalization for
    common non-canonical response shapes (`selected_action`, `prediction`, `action_details`).
  - Decision request is JSON-only (no think tags) to reduce parse failures.
  - Includes a self-critique replan pass (optional):
    - if a decision appears stale/repetitive/low-confidence relative to current perception,
      planner asks Cosmos for a better alternative in the same cycle,
    - candidate decision is accepted only if it scores materially better for current context.
  - Decision contract now supports utility-forward metadata:
    - `user_utility_goal`, `why_now`, `success_criteria`, `commit_s`
    - these are logged in transcript/memory and used to shape commitment timing.
  - Supports optional `frame_fetch` tool request when additional visual detail is needed.
  - Converts canonical decision to core primitives (`hold`, `breath`, `glance`, `nod`, `orient_to_zone`).
- **AsyncSceneSummarizer (Cosmos)**: low-rate scene understanding side loop.
  - Samples rolling frame window and emits compact `SceneSummary`.
  - Writes `summary_event` records into canonical memory stream.
  - Runs independently from high-level planner cadence.

## Context Sent to Cosmos
Current request context JSON includes:
- `control_state` (active primitive, action age, latest accepted decision),
- `summary_memory.latest_summary` + `summary_memory.recent_summaries`,
- `memory.recent_decisions` + `memory.recent_reasoning`,
- `memory.active_commitment` + `memory.transcript_tail`,
- `frame_meta` (frames available, whether images were included, optional frame-fetch reason).

Planner context intentionally avoids feeding raw local detector belief packets directly into remote decisions.

## Remote/Local Behavior Contract
- If Cosmos is enabled and reachable, planner runs in remote-first mode.
- Behavior layer remains active for timing + safety execution logic.
- In planner-owned semantics mode (default):
  - remote decisions are passed through directly to governor/control,
  - no local semantic mode rewrite is applied,
  - local behavior logic is safety-oriented (interruptibility, stale persistent refresh, zone nudge), not intent substitution.
- No local semantic substitute decision is generated when a remote response fails parse/validation; system keeps prior action (or initial neutral hold).
- Decision prompt enforces a stepwise decision tree (safety -> change -> opportunity -> selection).
- Planner may request `frame_fetch` before returning a strict action JSON.
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
  - `run_start`, `request_start`, `request_end`, `reasoning_event`, `decision_event`, `fallback_event`,
  - `summary_request_start`, `summary_request_end`, `summary_event`, `frame_fetch_event`.
- `logs/orchestrator_memory.jsonl`: canonical append-only memory events (`summary_event`, `decision_event`, `reasoning_event`).
- `logs/telemetry/preview/latest.jpg` + `latest.json`: optional preview tap for sidecar telemetry tools.

## Forward Plan: Memory Layers
- Near-term planner remains transcript-first.
- Future memory layering plan (spatial/environmental + short-term + long-term) is documented in:
  - `docs/memory_architecture.md`
