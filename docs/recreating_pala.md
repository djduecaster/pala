# Recreating PALA V1

The software and demo recipes are available; the mechanical build package is not
complete yet. CAD/STL files will be added separately. The existing `tools/camera mount fit check 2.step`
is a camera-mount fit artifact, not the full mechanism design. You can run the simulator
now, but the repository is not yet a complete shopping-and-assembly kit.

## Parts and connections

| Component | Known build detail | Still needed for an exact replica |
|---|---|---|
| Lamp | IKEA NYMÅNE mechanism | Exact product variant |
| Compute | NVIDIA Jetson | Board variant and tested JetPack/OS version |
| Actuation | Five hobby servos, PCA9685 driver | Servo model per joint, driver board variant |
| Camera | Logitech USB camera mounted on shade | Model and mounting photo |
| Mechanical parts | Custom printed mounts | CAD/STL, material, print settings, bearings and fasteners |
| Power | Jetson and servo power connections | Supply ratings and annotated wiring photo |

The current configuration maps PCA9685 channels **0–4** to **yaw, pitch1, pitch2,
roll, pitch3**, respectively. It uses I²C **bus 7**, address **0x40**, and **50 Hz**
PWM. These are values for the existing build, not universal Jetson pin assignments.
Confirm the bus and board pinout for your hardware before connecting anything.

The missing wiring photo should label SDA/SCL, logic supply, servo V+, ground
connections, and supply ratings. Do not infer servo supply voltage from the
Jetson GPIO voltage or treat the Jetson header as a servo power supply. Keep the
lamp's lighting power separate from the low-voltage servo wiring; this repository
does not provide instructions for modifying mains wiring.

## Software environment

Start with the [README's Mac/dummy quickstart](../README.md#try-it-without-hardware).
Python dependencies are declared in `pyproject.toml` and locked in `uv.lock`.
The Jetson additionally needs system GStreamer/PyGObject, the JPEG/V4L2 capture
plugins, I²C device access, and optionally FFmpeg for recording. Those system
components are not installed by `uv sync`.

Use a JetPack-supported system Python within the project's Python 3.10–3.12
range. For a **fresh** Jetson checkout, create the environment with system package
visibility before syncing:

```bash
uv venv --python /usr/bin/python3 --system-site-packages
uv sync
uv run python -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; Gst.init(None); print(Gst.version_string())"
```

This is a setup recipe, not a claim that a clean Jetson installation was retested.
The exact tested OS/package inventory is still to be recorded. If `gi` cannot be
imported, check both the system installation and the virtual environment's Python
version; installing Python requirements alone will not resolve a missing system
GStreamer installation. Do not replace an existing working environment blindly.

Complete this environment setup **before** `make go` or `run_on_jetson.sh`.
Their automatic bootstrap creates a plain virtual environment and does not expose
system GStreamer packages. They are not a fresh-Jetson installer.

The deployment scripts assume a locally configured `jetson` SSH alias; see the
[Jetson workflow](jetson_agent_workflow.md). `make go` starts the default runtime,
not the camera-driven social demo.

## Zero, directions, and calibration

**Joint degrees are not raw servo degrees.** `config/robot.yaml` contains the
mapping scale, offset, reversal and pulse range for each joint. Horn alignment,
linkage ratios and servo variants can require different values on another build.

- Zero means all five logical joint commands are zero. It is a known starting
  posture, not the relaxed camera-viewing pose. An annotated physical-zero photo
  is still needed before another builder can reproduce it reliably.
- Positive pitch1 moves forward on the current mechanism; negative moves back.
  Increasing pitch2 raises the lamp head away from the ground.
- Current pitch2 limits are −25° to +65°. Verify every joint's usable range on
  your own mechanism rather than copying this build's envelopes untested.
- The accepted rest command is `[0, -40, 25, 0, 0]` in joint order. Shutdown returns
  to zero. Neither posture is measured by the software; these servos have no
  position feedback.

After establishing your own zero alignment and safe initial mapping, stop other
servo owners and use the interactive calibration tool on the Jetson:

```bash
uv run python -m tools.hw_calibrate --enable --repl --slew-rate-deg-s 5
```

Enter `yaw 5`, then `yaw 0` to test a small displacement; substitute another joint
only after verifying its clearance. `neutral` commands **all joints** to logical
zero. `off` disables outputs; `q` exits. These commands can move hardware, and
other joints may be commanded to their current software estimates too. Start
with small increments, supported hardware, and an accessible power disconnect.
The tool starts from a zero estimate; it cannot find or sense physical zero.

The tool rejects nonfinite targets and targets outside the configured joint limits
before commanding movement. These software limits do not establish mechanical
clearance. All targets, including zero, must fit the configured envelope.

The REPL does not save calibration changes to YAML. Record results and update
`servo_calibration` / `joint_limits_rad` explicitly, then recheck small movements.
The [gesture workshop](gesture_workshop.md) comes after calibration.

## First successful run

Proceed in this order, with only one camera/servo owner at a time:

1. **Dummy runtime and tests:** use the README quickstart; expect a clean timed
   shutdown and passing tests without hardware or model calls.
2. **Camera only:** run the command below; expect frame statistics and a readable
   `logs/camera_snapshot.jpg`. Check seated framing at the intended rest pose.
3. **Individual joints:** verify zero, directions and modest movements with the
   calibration tool. Do not begin with the full performance library.
4. **Manual gestures:** follow [manual interaction](manual_interaction.md) and
   review clearance, motion and shutdown before enabling model-triggered actions.
5. **Model probe:** use [attention probe](attention_probe.md) to check empty,
   passing, looking-at-camera and looking-away scenes without model actuation.
6. **Live interaction:** follow [live interaction](live_interaction.md), including
   credential/model selection and automatic arming behavior. Use `shutdown` for
   a controlled return to zero; `stop` disables immediately without returning.

```bash
uv run python -m tools.test_camera_fps \
  --mode jetson_full --seconds 20 --threaded --snapshot
```

Gemini requires a separately provisioned API key and a model available to your
account. Keep credentials outside the repository. The live guide documents the
rehearsal model; provider availability and latency may change. Camera frames sent
to the model leave the device, and local logs may contain snapshots of people.

## Remaining owner-supplied items

One short hardware update can close most of the remaining gaps: exact parts and
supply ratings, tested Jetson software versions, a wiring photograph, and an
annotated zero/joint-direction photograph. CAD and print/assembly notes can follow
later. Until then, describe PALA as a documented prototype, not a fully replicated
reference build.
