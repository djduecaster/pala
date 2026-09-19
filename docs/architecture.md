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

## Runtime modes

The default behavior emits one persistent `hold` action and uses dummy backends
in dev mode. Opt-in manual performances and the supervised camera/Gemini demo
share the same four loops and typed control/hardware contracts.

The [live interaction](live_interaction.md) runner uses stationary camera snapshots
for semantic observations. Local code selects and sequences fixed performances;
the model never writes joint targets or hardware commands.

## Perception

The active perception path performs capture only:

```text
FrameSource -> PerceptionNode -> LatestFrameCache + PerceptionState
```

`PerceptionState` reports frame identity, freshness, timing, FPS, and source
health. It does not report people, objects, zones, gestures, or pointing.

Local DeepStream detection is not part of the active runtime. Semantic image
interpretation runs only in the opt-in model tools.

## Behavior

The behavior package includes a hold-only default policy, manual performance
request gate, model transport clients, and validation for social observations.
The live-demo supervisor uses one model request in flight and rejects stale,
invalid, or context-mismatched results before requesting a local performance.

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
position feedback. Following operator confirmation with the calibration tool,
pitch2's software limits are -25 to +65 joint degrees. These match its
unsaturated servo mapping; scale 2 and offset 50 remain unchanged. Physical
behavior was confirmed by the operator, not measured by software feedback.

Simulation supplies a private clock to the same executor. It does not patch
global time or change the runtime's real monotonic clock. Configuration rejects
nonfinite or invalid rates/limits/styles, and the servo backend validates the
entire command before beginning channel writes. I2C failures during a valid
batch are still possible; software validation is not bus-level atomicity.

Standalone gesture dry-runs use dummy hardware regardless of runtime mode.
The main runtime's four-loop deadman does not apply to these standalone tools.

## Logging

The default runtime writes:

- `perception.jsonl`: capture state and source health
- `actions.jsonl`: emitted hold actions
- optional preview image and metadata files for telemetry

Manual/live runs also write `interaction.jsonl` and recipe/configuration snapshots.
Live model observations, source images, and session settings are recorded under
`logs/attention_probe/<session>/`; see the live interaction guide.

Live telemetry defaults to the runtime view. Joint positions and enable state
are explicitly commanded values; applied hardware and deadman status are
unavailable without a structured status producer. Historical reasoning,
capture/replay, and curation remain optional sidecar tooling.

## Opt-in manual performances

`--manual` enables the [manual interaction](manual_interaction.md). The supervisor
polls terminal commands without adding a worker loop. The behavior loop gates
requests and publishes immutable performance recipes through a latest-value
channel. The control loop expands each recipe into the existing typed
`ActionPlan` steps and executes them through `TrajectoryExecutor`;
`PerceptionState`, `ActionPlan`, and `HardwareCommand` schemas are unchanged.
Control feedback reports commanded completion to the behavior gate.

This control-rate sequencer preserves short holds, rejects normal preemption,
and permits an explicit shutdown trajectory from the current command estimate.
It uses unit style multipliers and 1e-6-radian commanded completion tolerance,
matching the workshop. Default executor tolerance remains unchanged. Rest and
zero are distinct. Ctrl-C stops without recovery. Once manual commands have
started, a hardware deadman expiry stops the session without automatic resume.

Manual runs add recipe/configuration snapshots and `interaction.jsonl` execution
reports; their `actions.jsonl` records step transitions instead of hold-only
decisions. The separate live-demo composition has supervised rehearsal evidence described
in the live interaction guide. This does not establish unattended reliability.
