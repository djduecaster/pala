# PALA: Programmable Autonomous Lamp Assistant

PALA is a five-degree-of-freedom desk lamp built from an IKEA NYMANE lamp,
custom printed servo mounts, COTS servos, a PCA9685 board, a Logitech camera,
and an NVIDIA Jetson. The current goal is a repeatable portfolio demo: notice
a person, acknowledge them expressively, attend, and settle.

## Current baseline

The runtime captures frames and emits a persistent **hold** action. It does
not currently recognize people, call Gemini/Cosmos, or choose gestures.
The four-loop foundation, servo mapping, motion primitives, and optional
telemetry remain. Gesture choreography is exercised through separate tools
before autonomous behavior is connected.

Earlier NVIDIA competition and V3/V4 behavior systems are historical. See
[architecture](docs/architecture.md), [next steps](docs/todo.md), and the
[project assessment](docs/assessment_2026-09-04.md).

## Run locally on Mac

From `/Users/djduecaster/development/pala`, with `uv` installed:

```bash
uv sync
uv run python -m pala.main
```

The checked-in configuration defaults to dummy camera and servo backends.
For a bounded check:

```bash
PALA_MAX_RUNTIME_S=3 uv run python -m pala.main --mode dev
uv run pytest -q
```

Tests use pytest; hardware tests use fakes. Python is pinned in
`.python-version`, with the supported range defined in `pyproject.toml`.

| Mode | Camera | Servo | Behavior |
|---|---|---|---|
| `dev` | Dummy RGB frames | Dummy | Hold |
| `jetson_perception` | Jetson GStreamer | Dummy | Hold |
| `jetson_full` | Jetson GStreamer | PCA9685 | Hold |

`--mode` overrides `config/robot.yaml`. The camera-only Jetson mode is wired
in code; on-device operation still needs verification in the current lab.
Starting `jetson_full` can actuate the lamp even with hold-only behavior:
the executor's initial position is a software zero estimate, not an encoder
measurement of the physical posture.

## Gesture workshop tools

Run motion without hardware:

```bash
uv run python tools/expressive_movement_demo.py --runtime-mode dev --dry-run
uv run python tools/validate_primitives.py --runtime-mode dev --dry-run --scenario single --primitive nod --duration-s 1 --no-neutral-start --no-neutral-end
uv run python -m pala.control.primitive_tuner show
```

The simulator retains primitive traces, playback, and joint geometry checking.
Its simulated positions are commanded trajectories, not a physics model of
servo backlash, torque, gravity, or mechanical safety. See the
[simulator guide](tools/primitive_sim/README.md) for launch commands.

The expressive demo and validation runner require `--enable` for real servo
writes; `--dry-run` avoids constructing the real servo backend. These tools
are separate runners and do not inherit the main runtime's hardware-loop
deadman. Review starting posture and calibrated motion limits before an
operator-supervised powered session.

`tools/hw_calibrate.py` is the dedicated hardware calibration tool.
`tools/test_camera_fps.py` measures capture and can save snapshots. Camera and
servo checks are separate; do not run concurrent owners of the same device.

## Telemetry

The optional preview tap writes a reduced-rate JPEG and metadata. The existing
SSH sidecar and Mac viewer expose camera/perception information and commanded
joint state:

```bash
uv run python -m tools.telemetry.mac_viewer --jetson-host jetson-wifi --focus runtime
```

This command connects to a running Jetson setup; it is not a local dummy
runtime command. Live GUI video requires Python with Tk support. The viewer
does not measure actual servo position or prove that a command was applied.
Old reasoning and curation facilities remain opt-in for historical sessions.
See [telemetry usage](tools/telemetry/README.md).

The runtime writes `perception.jsonl` and `actions.jsonl` under a run directory
in `logs/runs/`. Preview files live separately under `logs/telemetry/preview/`.
Use the matching run when reviewing evidence; older behavior logs do not
describe the current process.

## Jetson workflow

The Mac checkout is the source of truth; the Jetson mirror is `~/pala`.
The existing deployment commands remain:

```bash
make deploy
make run
make go
```

`make go` deploys and runs over foreground SSH. Deployment scripts use the
`jetson` USB alias; direct Wi-Fi agent work uses `jetson-wifi` and the shared
`pala` tmux session. `make go-tmux` is a proposal, not an implemented target.
Read [Jetson agent workflow](docs/jetson_agent_workflow.md) before using that
shared shell. Deployment uses rsync deletion: preserve any Jetson-only
artifacts outside the mirror or pull them back before deployment.

Jetson camera capture requires GStreamer/PyGObject system packages visible to
the Python environment. For initial setup, create a system-site-packages venv
and install the project editable (`uv venv --system-site-packages`, `uv sync`,
`uv pip install -e .`). Inspect the existing environment before recreating it.
DeepStream is not required by the current runtime; historical bring-up notes
remain in [the reintroduction reference](pala/perception/DEEPSTREAM_REINTRODUCTION.md).

Jetson-only secrets remain in `~/.config/pala/env.sh`. Keep credentials out of
the repository and YAML. Retained model transport/probes are independent
diagnostics; changing `cosmos.enabled` does not activate runtime planning.

## Validation boundary

Passing local tests establishes software behavior with dummy/fake backends.
Gesture quality, current calibration, startup posture, camera coverage, and
physical stop behavior require a supervised hardware session. Known remaining
issues are recorded in [the bug log](docs/bug_log.md).
