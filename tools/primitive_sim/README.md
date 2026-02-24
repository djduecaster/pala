# Primitive Simulator (Sidecar Tool)

This simulator runs PALA control primitives offline and renders a simple 3D lamp playback in a browser.

It is sidecar-only under `tools/` and reuses the existing control executor math from `pala/control/executor.py`.

## Quick start

Generate a suite trace and launch the viewer:

```bash
uv run python tools/primitive_sim/run.py --scenario suite --serve
```

Generate one primitive:

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

Then open:

```text
/tools/primitive_sim/web/index.html?trace=/logs/primitive_sim/latest_trace.json
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

Notes:
- `action` uses the same schema as `ActionPlan` payloads.
- If `stop_on_done` is omitted, non-continuous primitives stop on completion and continuous primitives (`hold`, `breath`) run until `max_s`.

## Viewer Kinematic Model

The 3D viewer uses a DH-like chain aligned to the current lamp build:
- fixed floor base + long vertical mast
- yaw joint at mast top (rotates around vertical axis)
- `pitch1` shoulder with default zero pose pointing straight up
- `pitch2` elbow with zero pose collinear with upper arm
- `roll` twist around the forearm tube
- `pitch3` lampshade tilt (end-effector pitch)

## Geometry from config/robot.yaml

If present, the simulator reads lamp geometry from config and embeds it in trace metadata.

Supported config locations:
- `lamp_geometry`
- `sim_viewer.lamp_geometry`
- `primitive_sim.lamp_geometry`
- `tools.primitive_sim.lamp_geometry`

Example (meters):

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
