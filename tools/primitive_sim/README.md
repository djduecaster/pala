# Lamp Sim (Sidecar Tool)

Lamp Sim is a sidecar workflow for quickly understanding primitives, joint geometry, and behavior loops before hardware runs.

It keeps runtime behavior math aligned by reusing `TrajectoryExecutor` from `pala/control/executor.py`.

## Unified shell (recommended)

Run the Lamp Sim sidecar server:

```bash
uv run python tools/primitive_sim/run.py --scenario studio --port 8766
```

Then open the unified shell:

```text
http://127.0.0.1:8766/tools/primitive_sim/web/lamp_sim.html
```

The shell provides navigation between Studio, Joint Checker, and Playback.
Generate a suite trace in Studio before opening Playback. All modes are local
simulation tools; they do not command hardware or establish physical acceptance.

## Primitive Studio mode (recommended)

Run the interactive tuning tool:

```bash
uv run python tools/primitive_sim/run.py --scenario studio --port 8766
```

Raw mode URL:

```text
http://127.0.0.1:8766/tools/primitive_sim/web/index.html?studio=1
```

Studio, Joint Checker, and Playback are available from this same server instance.

Studio features:
- Select any runtime primitive (`hold`, `home`, `move_to`, `gaze_to`, `glance`, `nod`, `breath`, `orient_to_zone`)
- Tune primitive command fields
- Live auto-run preview with configurable debounce (default 200ms)
- Run preview simulation with 3D lamp playback
- Compare mode: baseline (cyan) vs draft (orange) in `overlay` or `split`
- Save tuned values to baseline params
- Save all baseline params in one write (`Save All`)
- Reload from baselines on startup
- Camera toolbar (`zoom +/-`, `orbit <- ->`, `reset view`) + mouse-wheel zoom
- Parameter filter and expanded nudge controls (`--`, `-`, `+`, `++`)
- Top bar mode buttons: switch between Studio, Joint Checker, and Playback
- `Run Suite Trace` button: generates `logs/primitive_sim/latest_trace.json` and opens playback

## Joint checker mode

Run the dedicated joint slider page:

```bash
uv run python tools/primitive_sim/run.py --scenario joint_checker --port 8766
```

Raw mode URL:

```text
http://127.0.0.1:8766/tools/primitive_sim/web/joint_checker.html
```

Joint Checker, Studio, and Playback are available from this same server instance.

Joint checker features:
- Per-joint sliders generated from `joint_names` + `joint_limits_rad`
- Live angle readouts in radians and degrees
- 3D pose rendering driven by current slider values
- Camera toolbar (`zoom +/-`, `orbit <- ->`, `reset view`) + mouse-wheel zoom
- Per-joint `+/-` nudge buttons adjacent to angle input
- DH parameter table loaded from `config/robot.yaml`
- Top bar mode buttons and `Run Suite Trace` button

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
