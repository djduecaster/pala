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

The configured rates are 20 Hz perception, 3 Hz behavior, 80 Hz control, and
120 Hz hardware. The deadman is a software check in the hardware thread; it
is not an independent electrical watchdog.

The executor uses latest-intent arbitration: a different intent replaces the
active primitive, equivalent intents preserve it, and a completed action ID
does not restart. `cancel_current` remains in the stable contract but is not
interpreted by the executor. Gesture sequencing must account for this behavior.

The executor starts at a software zero estimate and exposes commanded
completion, not measured physical arrival. The current servo interface has no
position feedback. Calibration numbers are unchanged by the cleanup; for
example, pitch2's scale/offset maps roughly -25 to +65 joint degrees into the
servo's unsaturated 0–180 degree range, narrower than the configured software
limits. This mathematical range is not a validated physical safety envelope.

Simulation supplies a private clock to the same executor. It does not patch
global time or change the runtime's real monotonic clock. Configuration rejects
nonfinite or invalid rates/limits/styles, and the servo backend validates the
entire command before beginning channel writes. I2C failures during a valid
batch are still possible; software validation is not bus-level atomicity.

Standalone gesture dry-runs use dummy hardware regardless of runtime mode.
The main runtime's four-loop deadman does not apply to these standalone tools.

## Logging

The reset baseline writes:

- `perception.jsonl`: capture state and source health
- `actions.jsonl`: emitted hold actions
- optional preview image and metadata files for telemetry

Model-decision and reasoning logs will be specified with the Phase 3 contract.

Live telemetry defaults to the runtime view. Joint positions and enable state
are explicitly commanded values; applied hardware and deadman status are
unavailable without a structured status producer. Historical reasoning,
capture/replay, and curation remain optional sidecar tooling.
