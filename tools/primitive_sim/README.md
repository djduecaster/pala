# Lamp Sim (Sidecar Tool)

Lamp Sim is a sidecar workflow for quickly understanding primitives, joint geometry, and behavior modes before hardware runs.

It keeps runtime behavior math aligned by reusing `TrajectoryExecutor` from `pala/control/executor.py`.

## Unified shell (recommended)

Run any Lamp Sim scenario server:

```bash
uv run python tools/primitive_sim/run.py --scenario studio --port 8766
```

Then open the unified shell:

```text
http://127.0.0.1:8766/tools/primitive_sim/web/lamp_sim.html
```

The shell provides one persistent navigation/workflow UI for Studio, Scenario Lab, State Machine, and Joint Checker.

Shell UX features:
- One persistent mode navigation + workflow checklist
- Mode-aware primary action button in top bar
- Global command palette (`Cmd/Ctrl+K`) for fast mode/action switching
- Global toast/status feedback from active mode actions
- Keyboard mode shortcuts (`1..4`) and suite playback shortcut (`r`)
- Cross-mode handoff: send FSM recommended primitive into Scenario builder/steps

## Primitive Studio mode (recommended)

Run the interactive tuning tool:

```bash
uv run python tools/primitive_sim/run.py --scenario studio --port 8766
```

Raw mode URL:

```text
http://127.0.0.1:8766/tools/primitive_sim/web/index.html?studio=1
```

Studio, Joint Checker, State Machine, and Scenario Lab pages are available from this same server instance.

Studio features:
- Select any runtime primitive (`hold`, `home`, `move_to`, `gaze_to`, `glance`, `nod`, `breath`, `orient_to_zone`)
- Tune primitive command fields
- Live auto-run preview with configurable debounce (default 200ms)
- Run preview simulation with 3D lamp playback
- Compare mode: baseline (cyan) vs draft (orange) in `overlay` or `split`
- Save tuned values to baseline params
- Save all baseline params in one write (`Save All`)
- Reload from baselines on startup
- Top bar mode buttons: switch between Studio, Joint Checker, State Machine, and Scenario Lab
- `Run Suite Playback` button: generates `logs/primitive_sim/latest_trace.json` and opens playback

## Joint checker mode

Run the dedicated joint slider page:

```bash
uv run python tools/primitive_sim/run.py --scenario joint_checker --port 8766
```

Raw mode URL:

```text
http://127.0.0.1:8766/tools/primitive_sim/web/joint_checker.html
```

Joint Checker, Studio, State Machine, and Scenario Lab pages are available from this same server instance.

Joint checker features:
- Per-joint sliders generated from `joint_names` + `joint_limits_rad`
- Live angle readouts in radians and degrees
- 3D pose rendering driven by current slider values
- DH parameter table loaded from `config/robot.yaml`
- Top bar mode buttons and `Run Suite Playback` button

## State machine mode

Run the behavior mode simulator:

```bash
uv run python tools/primitive_sim/run.py --scenario state_machine --port 8766
```

Raw mode URL:

```text
http://127.0.0.1:8766/tools/primitive_sim/web/state_machine.html
```

State machine features:
- Visual mode graph for `idle_presence`, `scan_explore`, `engage_track`, `acknowledge`, `recover_reset`
- Step or auto-run mode transitions from editable input signals
- Signal controls: person presence/confidence, activity, novelty, env delta, health breakers
- Live primitive response table using deterministic `IdleEngine` proposals
- Allowed-primitive readout per mode
- Reuses runtime `ModeManager` + `IdleEngine` semantics for transition/proposal behavior

## Scenario lab mode

Run scenario composition + evaluation:

```bash
uv run python tools/primitive_sim/run.py --scenario scenario_lab --port 8766
```

Raw mode URL:

```text
http://127.0.0.1:8766/tools/primitive_sim/web/scenario_lab.html
```

Scenario Lab features:
- Build multi-step primitive scenarios from a step builder or direct JSON
- Validate scenarios before execution (`dry_run` compile path)
- Run scenarios and auto-open embedded playback preview
- Scenario metrics (`duration`, `path_length`, `peak/mean velocity`, `limit margin/violations`, `switch count`)
- Run parameter sweeps on one step command (grid search + weighted score ranking)
- Generate sweep templates from the selected target step primitive
- Apply best sweep patch back into scenario JSON
- Promote best sweep candidate directly to primitive baseline params
- Save experiment records to JSONL history and reload trace previews from history

Default experiments path:

```text
logs/primitive_sim/experiments.jsonl
```

Override path:

```bash
uv run python tools/primitive_sim/run.py --scenario scenario_lab --experiments path/to/experiments.jsonl
```

## Roadmap

Phase 1 (implemented):
- Primitive Studio tuning
- Joint Checker
- State Machine simulator
- Scenario Lab (compose + run + evaluate + save history)

Phase 2 (in progress):
- Parameter sweep / optimizer for scenario step params (implemented in Scenario Lab native shell)
- Batch A/B eval runner (baseline vs candidate) with scorecards (next)
- Servo/hardware emulation layer (latency, deadband, quantization) (next)

Phase 3 (next):
- Fault-injection scenarios for breaker/perception dropouts
- Reachability/workspace map and safety envelope visualizer
- Evidence-pack exporter for demo/ablation artifacts

## Baseline params file

Default path:

```text
tools/primitive_sim/baseline_params.json
```

Override path:

```bash
uv run python tools/primitive_sim/run.py --scenario studio --baseline path/to/baseline.json
```

Baseline schema stores command defaults per primitive plus metadata:

```json
{
  "version": 2,
  "updated_by": "primitive_studio",
  "updated_at_utc": "2026-02-27T00:00:00Z",
  "primitives": {
    "breath": {"amp_rad": 0.08, "period_s": 6.5, "rate_rad_s": 1.0}
  }
}
```

Baseline loading is strict:
- Baseline files must already be v2.
- Missing primitive payloads or invalid command fields fail fast.

## Trace modes (existing CLI)

Generate suite trace:

```bash
uv run python tools/primitive_sim/run.py --scenario suite --serve
```

Generate one primitive trace:

```bash
uv run python tools/primitive_sim/run.py \
  --scenario single \
  --primitive nod \
  --duration-s 1.2 \
  --amp-rad 0.2 \
  --rate-rad-s 1.8 \
  --serve
```

Generate without serving:

```bash
uv run python tools/primitive_sim/run.py --scenario suite --output logs/primitive_sim/latest_trace.json
```

## Scripted scenarios

Use `--scenario script --script <file.json>`.

Script format:

```json
{
  "segments": [
    {
      "name": "home",
      "max_s": 2.0,
      "stop_on_done": true,
      "action": {
        "primitive": "home",
        "command": {"rate_rad_s": 1.2},
        "confidence": 1.0,
        "style": "calm"
      }
    }
  ]
}
```

## Viewer geometry sources

Viewer geometry is included in trace metadata and used by the 3D renderer.

Priority:
1. `dh_params`-derived values from `config/robot.yaml` (link lengths + theta offsets)
2. Explicit overrides from `lamp_geometry` sections
3. Viewer defaults for missing values

Pitch convention:
- Viewer defines all pitch joints about local `-Z`.
- Viewer applies `theta = q + theta0`, so positive `q` pitches forward by construction.

Supported override locations:
- `lamp_geometry`
- `sim_viewer.lamp_geometry`
- `primitive_sim.lamp_geometry`
- `tools.primitive_sim.lamp_geometry`

Example:

```yaml
sim_viewer:
  lamp_geometry:
    mast_height_m: 1.28
    hub_rise_m: 0.02
    upper_arm_len_m: 0.52
    fore_arm_len_m: 0.42
    wrist_stub_len_m: 0.08
    shade_neck_len_m: 0.08
    shade_len_m: 0.18
    shade_base_radius_m: 0.068
    shade_tip_radius_m: 0.046
    pitch1_zero_offset_rad: 1.57079632679
    pitch2_zero_offset_rad: 0.0
    pitch3_zero_offset_rad: 0.0
```
