# Primitive Studio (Sidecar Tool)

Primitive Studio is a sidecar workflow for quickly understanding what each primitive looks like before hardware runs.

It keeps runtime behavior math aligned by reusing `TrajectoryExecutor` from `pala/control/executor.py`.

## Studio mode (recommended)

Run the interactive tuning tool:

```bash
uv run python tools/primitive_sim/run.py --scenario studio --port 8766
```

Then open:

```text
http://127.0.0.1:8766/tools/primitive_sim/web/index.html?studio=1
```

Both Studio and Joint Checker pages are available from this same server instance.

Studio features:
- Select any runtime primitive (`hold`, `home`, `move_to`, `gaze_to`, `glance`, `nod`, `breath`, `orient_to_zone`)
- Tune primitive command fields
- Run preview simulation with 3D lamp playback
- Save tuned values to baseline params
- Reload from baselines on startup
- Top bar mode buttons: switch between Studio and Joint Checker
- `Run Suite Playback` button: generates `logs/primitive_sim/latest_trace.json` and opens playback

## Joint checker mode

Run the dedicated joint slider page:

```bash
uv run python tools/primitive_sim/run.py --scenario joint_checker --port 8766
```

Then open:

```text
http://127.0.0.1:8766/tools/primitive_sim/web/joint_checker.html
```

Both Joint Checker and Studio pages are available from this same server instance.

Joint checker features:
- Per-joint sliders generated from `joint_names` + `joint_limits_rad`
- Live angle readouts in radians and degrees
- 3D pose rendering driven by current slider values
- DH parameter table loaded from `config/robot.yaml`
- Top bar mode buttons and `Run Suite Playback` button

## Baseline params file

Default path:

```text
tools/primitive_sim/baseline_params.json
```

Override path:

```bash
uv run python tools/primitive_sim/run.py --scenario studio --baseline path/to/baseline.json
```

Baseline schema stores command defaults per primitive (command fields only):

```json
{
  "version": 1,
  "primitives": {
    "breath": {"amp_rad": 0.08, "period_s": 6.5, "rate_rad_s": 1.0}
  }
}
```

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
