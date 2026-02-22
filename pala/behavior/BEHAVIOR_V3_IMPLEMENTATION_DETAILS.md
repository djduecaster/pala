# Behavior V3 Implementation Details

## Objective
Rebuild behavior as a remote-first, Cosmos-first stack with deterministic local arbitration and no legacy V2 parser/policy paths.

## Core Design
- Keep runtime contracts unchanged: `PerceptionState -> ActionPlan -> HardwareCommand`.
- Keep loop topology unchanged (perception/behavior/control/hardware).
- Split behavior into:
  1. `EnvSummarizer` (remote scene+event+hypothesis summarization)
  2. `IntentProposer` (remote ranked intent proposals)
  3. deterministic local stack (`Governor`, `Arbiter`, `ActionCompiler`, `IdleEngine`, `HealthManager`)
- Treat long reasoning as debug-only logs; do not feed reasoning text into planner memory.

## Behavior V3 Modules
- `types.py`: canonical behavior dataclasses (`IntentProposal`, `ProposerResponse`, etc.).
- `schemas.py`: strict JSON schema contracts for env and intent responses.
- `prompts.py`: compact contracts that require JSON-only responses.
- `context_builder.py`: deterministic, bounded planner/env context packet builder.
- `env_summarizer.py`: schema-validated parser and latest-only request bookkeeping.
- `intent_proposer.py`: schema-validated parser and latest-only request bookkeeping.
- `governor.py`: deterministic validation + risk gating + base utility scoring.
- `arbiter.py`: hysteresis, min dwell, anti-collapse utility adjustments.
- `action_compiler.py`: proposal -> validated `ActionPlan` conversion.
- `idle_engine.py`: deterministic low-amplitude presence proposals.
- `health_manager.py`: quality/transport health and breaker states.
- `trace_bus.py`: per-step trace logging.
- `policy.py`: orchestrates async remote calls, arbitration, commit discipline.

## Data Flow (per behavior tick)
1. Ingest latest frame into rolling window.
2. Drain completed remote env/planner calls.
3. Update world state from env summary (`scene/events/hypotheses/summary/delta/features`).
4. Update planner health from parse/quality outcomes.
5. Schedule next env/planner calls (latest-only inflight semantics).
6. Build candidate set:
   - fresh remote proposals (if not stale)
   - deterministic idle proposals (heartbeat / micro-scan)
7. Governor validates and scores.
8. Arbiter decides `commit` vs `keep_current`.
9. On commit only:
   - compile to `ActionPlan`
   - append decision tail entry
10. Emit trace log.

## Critical Invariants
- Non-committed model decisions are never persisted into decision memory.
- Planner/env control path uses strict JSON parse only.
- Remote calls are latest-only, max one inflight per component.
- Local logic never generates high-risk behavior.
- Behavior loop remains non-blocking.

## Memory Discipline
- `event_tail`: env-processor dense event summaries only.
- `decision_tail`: committed actions only.
- `latest_env_snapshot`: compact world state features for planner context.

## Logging
- `behavior_env.jsonl`: env request lifecycle, parse status, summary, delta.
- `behavior_planner.jsonl`: proposer request lifecycle, parse status, top proposal.
- `behavior_reasoning.jsonl`: optional remote reasoning text (debug only).
- `behavior_trace.jsonl`: deterministic arbitration boundary trace.

## Config Surface
Behavior V3 uses existing cosmos-derived config plus:
- `proposer_max_age_s`
- `planner_max_proposals`
- `arbiter_min_dwell_s`
- `arbiter_base_margin`
- `idle_after_s`
- `idle_glance_after_s`
- `trace_log_path`

## Known Tradeoffs
- Idle engine is deterministic and intentionally low-amplitude; it prevents dead-looking behavior but does not replace remote semantic authority.
- Parsers strip common transport wrappers (for example fenced JSON) and then enforce schema validation on canonical payloads.
