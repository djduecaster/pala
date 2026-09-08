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
- Pitch1 and pitch3 use `theta = theta0 + q`.
- Pitch2 uses `theta = theta0 - q`: the operator confirmed positive pitch2
  opens/lifts the elbow away from the ground. This display convention applies
  to both Playback/Studio and Joint Checker; it does not reverse servo commands.

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

## Scripted desk interaction (simulation only)

Generate the current expression study, then serve this repository locally:

```bash
uv run python -m tools.primitive_sim.social_scene
uv run python -m http.server 8766 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8766/tools/primitive_sim/web/index.html?studio=0&trace=/logs/primitive_sim/desk_scene.json
```

Use **Play / pause** above the lamp, or jump between the five labeled moments.
The timeline and speed controls allow closer inspection. The approximately
56-second scene is authored in `tools/primitive_sim/desk_scene.json`:

1. Established rest, pitch1=-40/pitch2=+25.
2. Chair enters: a quiet yaw=12/pitch1=-34/pitch2=30/roll=3 glance.
3. Eye contact: a broader greeting with yaw 11..39, pitch1 -22..-8,
   pitch2 23..39, roll -28..28 degrees, followed by a damped rebound.
4. Thumbs-up: two alternating bounces, yaw 20..50, pitch1 -23..-5,
   pitch2 27..47, roll -32..32 degrees, then yaw=35/pitch1=-15/pitch2=25/roll=10/pitch3=0.
5. Back to work: four-second grace period, release tilt, face forward,
   move both pitch joints to rest, and hold.

Human actions are captions, not a simulated human model. Observation delays
(2.5/2.8 seconds) are illustrative. There is no Gemini call, camera access,
thumbs-up detector, or hardware output. The scene exercises TrajectoryExecutor
at 80 Hz with unity style scales and the workshop's 1e-6-radian tolerance.
Pitch3 stays zero. Configured targets are checked before generating the trace.
The existing live performance library is not changed or deployed by this tool.

This viewer uses the retained kinematic geometry; it cannot establish physical
clearance, camera framing after motion, or how servo loading changes the rhythm.
Before promotion, review the scene visually, then try each new movement on the
lamp. Gemini presence/attention/affirmative-gesture decisions and automatic
settling remain a separate integration step; scripted timing is not evidence
that those decisions work live.

Pitch2 direction correction: the operator confirmed positive pitch2 lifts away
from the ground. Both viewer paths now subtract the joint value from the elbow
zero offset. The temporary higher pitch2 scene targets were reverted because
they compensated for an incorrect rendering direction. Hardware calibration,
limits, zero offsets, and live performance recipes were not changed.

Hardware workshop revision: greeting pitch2 targets were lowered by 10 degrees
(30, 39, 23, 35, then 30 at the attention landing). This applies only to the
greeting beat; other beats retain their candidates pending individual tests.
The operator accepted the revised greeting at 5/5 in trial 4c1f6a2d62ff.

Excited workshop candidate v2 shifts the oscillation center +10 degrees in yaw
and lowers pitch2 5 degrees. Trial 996eff8ac06a was accepted at 5/5 at 70% of the original authored rates.
The simulation now uses those accepted rates too. Settling starts from yaw=35,
pitch1=-15, pitch2=25, roll=10 and keeps pitch2 at 25 throughout, avoiding an
unwanted head lift. Settling was accepted at 5/5 in physical trial 3df8f9c049a1.
