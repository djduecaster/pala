# PALA Architecture

## Runtime Loops

- Perception loop: approximately 15-30 Hz
- Behavior loop: approximately 2-5 Hz
- Control loop: approximately 50-100 Hz
- Hardware loop: approximately 50-200 Hz with deadman timeout

The stable runtime contract remains:

```text
PerceptionState -> ActionPlan -> HardwareCommand
```

## Current Reset Baseline

PALA is between behavior architectures. The V4 mode FSM, skills, prompts,
decision schema, and ActionGuard were removed intentionally before designing the
next model-driven behavior agent.

The behavior loop currently emits one persistent `hold` action. This is not a
fallback behavior architecture; it is a temporary contract-preserving baseline
that keeps the four-loop runtime executable while Phase 3 is designed.

## Perception

The active perception path performs capture only:

```text
FrameSource -> PerceptionNode -> LatestFrameCache + PerceptionState
```

`PerceptionState` reports frame identity, freshness, timing, FPS, and source
health. It does not report people, objects, zones, gestures, or pointing.

Local DeepStream detection was removed from the runtime. Reintroduction notes
are preserved in `pala/perception/DEEPSTREAM_REINTRODUCTION.md`.

## Behavior

The behavior package currently retains only:

- model transport clients
- deterministic JSON extraction
- a hold-only boundary policy

The next behavior architecture will be designed before adding model calls,
state transitions, semantic skills, or action validation.

## Control and Hardware

Control remains deterministic. `TrajectoryExecutor` converts typed
`ActionPlan` objects into rate-limited, joint-clamped `HardwareCommand` objects.
The hardware loop applies commands through dummy or PCA9685 servo backends and
enforces the deadman timeout independently of behavior/model availability.

## Logging

The reset baseline writes:

- `perception.jsonl`: capture state and source health
- `actions.jsonl`: emitted hold actions
- optional preview image and metadata files for telemetry

Model-decision and reasoning logs will be specified with the Phase 3 contract.
