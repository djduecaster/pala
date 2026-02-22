# PALA Architecture

## Runtime Loops (unchanged)
- Perception loop: ~15-30 Hz
- Behavior loop: ~2-5 Hz
- Control loop: ~50-100 Hz
- Hardware loop: ~50-200 Hz with deadman timeout

Data contract remains:
`PerceptionState -> ActionPlan -> HardwareCommand`

## Behavior V3 (remote-first)
Behavior is now a Cosmos-first stack with deterministic local arbitration.

### Remote components
1. **EnvSummarizer** (slow):
   - Inputs: short frame sequence
   - Output: strict JSON `scene/events/hypotheses/summary_short/delta_score/features`
2. **IntentProposer** (fast):
   - Inputs: compact context + optional latest frame
   - Output: strict JSON ranked intent proposals

### Deterministic local components
- **ContextBuilder**: bounded context packet for env/proposer
- **Governor**: validates/risk-gates proposals
- **Arbiter**: hysteresis + dwell + anti-collapse commit logic
- **ActionCompiler**: proposal -> validated `ActionPlan`
- **IdleEngine**: low-amplitude heartbeat proposals (non-semantic)
- **HealthManager**: quality/transport breaker states
- **TraceBus**: per-cycle arbitration trace logging

## Memory model
- `latest_env_snapshot`: latest structured world summary
- `event_tail`: env summaries only (dense text)
- `decision_tail`: committed actions only
- identity/session digest remain persistent markdown files under `memory/`

## Logging
- `behavior_env.jsonl`: env request lifecycle and parse outcomes
- `behavior_planner.jsonl`: proposer request lifecycle and top proposal
- `behavior_reasoning.jsonl`: optional remote reasoning text (debug only)
- `behavior_trace.jsonl`: deterministic arbitration boundary traces
