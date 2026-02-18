# PALA Architecture (Scaffold)

## Runtime Modes (Planned)
- **dev**: fully dummy perception + dummy hardware; safe for Mac-only development.
- **jetson_perception**: reserved for later; needs careful design to avoid confusing closed-loop behavior.
- **jetson_full**: full Jetson perception + control + hardware actuation (near-term focus).

## Four-Loop Runtime
1. **Perception Loop (15–30 Hz)**
   - Reads from a `FrameSource` (dummy by default).
   - Produces `PerceptionState` with normalized bbox and optional pointing target.

2. **Behavior Loop (2–5 Hz)**
   - Consumes latest `PerceptionState`.
   - Applies dwell + hysteresis on left/center/right zones.
   - Emits `ActionPlan`.

3. **Control Loop (50–100 Hz)**
   - Converts typed `ActionPlan` commands → `HardwareCommand` via `TrajectoryExecutor`.
   - Single active primitive with priority preemption and safety clamps.

4. **Hardware Loop (50–200 Hz)**
   - Sends `HardwareCommand` to servo backend.
   - Applies deadman timeout; disables on stale command.

## Data Contracts
- `PerceptionState`: timestamp(s), fps, latency, normalized bbox, optional pointing target.
- `ActionPlan`: primitive kind, typed command payload, confidence, optional explanation.
- `HardwareCommand`: timestamp + joint_angles_rad + enable flag.

## Evaluation & Reproducibility
- **Inputs**: camera frames (Jetson modes), config in `config/robot.yaml` plus optional `--mode` override at runtime.
- **Outputs**: JSONL logs for perception/actions when a single logging flag is enabled.
- **Optional outputs**: runtime telemetry preview tap (`telemetry_preview`) writes latest JPEG + metadata files for sidecar viewers.
- **Steps**: run a mode, capture logs, and (optionally) replay or compare action outputs between runs.

## Notes
- Live preview/telemetry streaming is optional and may be enabled for debugging/demos, ideally as sidecar tooling.
- Default runtime uses dummy hardware and perception, safe for Mac.
- Jetson compatibility maintained via TODO stubs in hardware/perception.
