# Behavior V4 Demo Runbook

## Goal
Run a reproducible V4 behavior demo where Cosmos is in the decision loop and deterministic guardrails keep behavior stable.

## Runtime Contract
- Perception writes structured behavior signals into `PerceptionState.debug`.
- Behavior V4 reads only structured behavior signals for semantic transitions.
- Planner returns `pala.behavior_decision.v1` JSON.
- ActionGuard validates mode/mood/skill/primitive/dwell/cooldown/staleness.

## Canonical `debug` Fields
Use these keys in `PerceptionState.debug`:
- `person_present` (bool)
- `person_conf` (float 0..1)
- `search_requested` (bool)
- `search_complete` (bool)
- `assist_complete` (bool)
- `user_ack` (bool)
- `task_active` (bool)
- `home_requested` (bool)
- `home_completed` (bool)
- `cancel_requested` (bool)
- `health_degraded` (bool)

## Expected Mode Flow
Typical demo flow:
1. `boot_awaken` -> `idle_presence`
2. `idle_presence` -> `social_interact` when person signal is stable
3. `social_interact` -> `search_assist` on request
4. `search_assist` -> `social_interact` on complete+ack
5. `social_interact` -> `return_home` on home request
6. `return_home` -> `idle_presence` on home completion

## Key Config Knobs (`config/robot.yaml`)
Under `cosmos`:
- `planner_hz`
- `mode_min_dwell_s`
- `mode_boot_timeout_s`
- `mode_return_home_settle_s`
- `mode_recover_settle_s`
- `action_guard_stale_after_s`
- `action_guard_orient_cooldown_s`
- `action_guard_glance_cooldown_s`
- `action_guard_nod_cooldown_s`
- `action_guard_home_cooldown_s`

## Run Steps
1. Set endpoint vars if needed:
   - `export PALA_COSMOS_BASE_URL=...`
   - `export PALA_COSMOS_MODEL=nvidia/cosmos-reason2-2b`
2. Run locally:
   - `uv run python -m pala.main`
3. Run on Jetson:
   - `make go`

## Logs to Watch
- `logs/behavior_planner.jsonl`
- `logs/behavior_trace.jsonl`
- `logs/actions.jsonl`

For scoped runs:
- `logs/runs/<RUN_ID>/behavior_planner.jsonl`
- `logs/runs/<RUN_ID>/behavior_trace.jsonl`

## Healthy Run Signals
- Planner status mostly `ok`.
- Parse stage mostly `raw` or `defenced`.
- Guard rejections are occasional and explicit (not continuous).
- Mode transitions match expected event triggers.

## Failure Triage
1. **Continuous `parse_fail`**
   - Check planner response format and schema version.
2. **Mostly fallback actions**
   - Check guard reasons in trace (`mode_not_current`, `mood_not_allowed`, cooldown).
3. **No social transitions**
   - Verify `debug.person_present` and `debug.person_conf` values.
4. **Search never exits**
   - Verify `search_complete`, `assist_complete`, and `user_ack` signals.
5. **Slow/late actions**
   - Reduce `planner_hz` burden and tighten stale/cooldown values.

## Demo Safety Defaults
- Keep motion style calm by default.
- Maintain nonzero dwell and cooldown values.
- Keep fail-closed behavior: invalid model outputs should fallback to safe primitives.
