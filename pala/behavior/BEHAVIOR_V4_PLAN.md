# Behavior V4 Plan (Single-Call, Mode-Driven)

## Why V4
Behavior V3 proved the transport and control scaffolding, but the dual env/planner path and utility-heavy arbitration added complexity and reduced clarity during demo tuning.

Behavior V4 simplifies to one model decision call per cycle with a strict deterministic action guard.

## Design Targets
- Keep Cosmos Reason 2 central in decision-making.
- Keep the 4-loop runtime model intact (perception, behavior, control, hardware).
- Keep control primitives deterministic and safe.
- Remove dead code/fallback branches that are no longer part of the target design.
- Optimize for a 3-minute live demo, not general autonomy.

## Scope Boundaries
- In scope: `pala/behavior/*` refactor and tests.
- Keep: model clients, JSON parse helper, control layer, hardware layer, main loop wiring.
- Out of scope for initial V4 cut: new local detector semantics, 3D localization, major control primitive expansion.

## V4 Runtime Shape
1. `BehaviorPolicyV4.step()` runs at behavior cadence.
2. Build compact context packet:
   - current macro mode
   - current skill (if any)
   - current action summary
   - cooldown/dwell counters
   - recent transcript summary (optional, bounded)
   - latest image frame
3. Send one Cosmos call (`response_format=json_schema`) for a decision.
4. Parse + validate decision JSON.
5. `ActionGuard` validates action against deterministic rules.
6. If valid and committable, compile to `ActionPlan` and commit.
7. If invalid/stale/missing, execute deterministic fallback for current mode.

## Macro Modes (Tier 1)
1. `BOOT_AWAKEN`
   - Purpose: startup expression + camera settle to observation pose.
   - Exit: sequence complete or timeout.
2. `IDLE_PRESENCE`
   - Purpose: calm aliveness (breath + occasional glance).
   - Exit: person/activity trigger or command trigger.
3. `SOCIAL_INTERACT`
   - Purpose: greeting/engagement behaviors with mood/style variation.
   - Exit: task request trigger, disengage trigger, or timeout to idle.
4. `SEARCH_ASSIST`
   - Purpose: user-requested search/point flow.
   - Exit: target confirmed, cancel, or timeout to social/idle.
5. `TASK_LIGHTING`
   - Purpose: task-oriented positioning (call/read lighting).
   - Exit: task no longer active or explicit home command.
6. `RECOVER_RESET`
   - Purpose: degraded/failure-safe behavior.
   - Exit: health restored and minimum settle met.

## Skill Layer (Tier 2)
Skills are mode-scoped templates built from existing primitives:
- `wake_sequence`
- `observe_settle`
- `greet_user`
- `social_ack`
- `expressive_search`
- `point_and_hold`
- `task_light_adjust`
- `return_home`

Each skill has:
- allowed primitives
- max dwell
- completion criteria
- timeout fallback

## Decision Contract (Model Output)
Schema target: `pala.behavior_decision.v1`

Required fields:
- `schema_version`
- `mode` (chosen or keep)
- `mood` (`calm|curious|excited|focused|neutral`)
- `skill`
- `action` (`primitive`, `command`, `style`)
- `confidence` (0..1)
- `rationale_short`
- `mode_transition` (`stay|to_<mode>`)

Optional:
- `alternatives` (top 2 concise options for trace/debug)

## Deterministic ActionGuard (Replaces Governor+Arbiter)
ActionGuard must stay small and strict:
- Primitive allowed in current mode/skill.
- Command ranges clamped to safe envelopes.
- Cooldown checks (per primitive).
- Min dwell checks (avoid thrash).
- Staleness checks (drop old model outputs).
- Health gate checks (degraded/open breaker behavior).

Outputs:
- `accept` with normalized action
- `reject` with explicit reason code
- `fallback` action for current mode

## Health and Failure Policy
- One in-flight model request, latest-only pending.
- Request watchdog timeout and stale result handling.
- Parse/schema failures counted; no silent coercion.
- On repeated failures: mode -> `RECOVER_RESET`, continue deterministic behavior.
- No writes of non-committed decisions to world memory.

## Prompt Strategy (Mode-Aware)
- One base system contract: strict JSON, no markdown/fences.
- Mode-specific user instructions appended per tick.
- Keep prompts short and egocentric ("camera view is my view").
- Provide only mode-relevant skill/action options to reduce ambiguity.

## Implementation Phases

### Phase A - Skeleton and Contracts
- Add V4 modules:
  - `behavior/policy_v4.py`
  - `behavior/action_guard.py`
  - `behavior/mode_fsm_v4.py`
  - `behavior/decision_schema_v4.py`
  - `behavior/skills_v4.py`
- Add `pala.behavior_decision.v1` schema and parser.
- Add minimal tests for parse/validation/guard.

### Phase B - Single-Call Integration
- Wire policy to one model call path.
- Remove env/planner dual-call usage from active V4 path.
- Add trace fields for mode, skill, guard decision, reject reasons.

### Phase C - Demo Skills
- Implement `wake_sequence` + `observe_settle`.
- Implement `greet_user`, `expressive_search`, `task_light_adjust`.
- Add mode transition guards and timeout exits.

### Phase D - Hard Cutover and Cleanup
- Switch runtime to V4 behavior policy flag default.
- Delete V3-only dead behavior code paths not used by V4.
- Keep only reusable shared utilities (model client, json parser, types).
- Update docs and test suite to reflect new architecture.

## Acceptance Criteria
- Runtime boots and runs on Mac dummy backends.
- Behavior ticks remain non-blocking with remote failures.
- Model decision parse success >= 95% in probe runs.
- No mode thrash (bounded by dwell/cooldown rules).
- Demo flow can complete end-to-end with deterministic fallback.

## Immediate Next Steps
1. Implement Phase A scaffolding and tests.
2. Wire Phase B single-call loop and trace logging.
3. Add `wake_sequence` and `greet_user` first for rapid demo iteration.
