# Manual interaction demo

The opt-in manual runtime runs the selected gesture vocabulary through PALA's
existing four loops. It does not detect visitors or call a model. Default runtime
behavior remains hold-only.

## Run on Mac

From the repository root:

```bash
uv run python -m pala.main --manual --mode dev
```

Startup holds software zero for two seconds, enters rest, and holds it for two
seconds. Wait for startup completion before typing a command, followed by Enter.
The terminal lists the active performance, step number, movement name, and phase.

| Command | Result |
|---|---|
| `help` or `?` | Show common commands. |
| `status` | Show state, active performance, progress, and any failure. |
| `greet` | Look up, wave with yaw and roll, then remain attentive. |
| `attend` | Move directly into the attention pose and hold three seconds. |
| `settle` | Release roll, center yaw, then move both pitch joints toward rest together; hold five seconds. |
| `demo` | Greeting → attention hold → settling, ending at rest. |
| `shutdown`, `q`, or `quit` | Interrupt the current performance, return to zero at 8°/s, hold two seconds, then disable outputs and exit. |
| `stop`, `abort`, or Ctrl-C | Disable outputs and exit without a recovery trajectory. |

Requests during a performance are rejected, except shutdown. Nothing is queued.
A completed greeting cannot be repeated until settling. `demo` ends at rest and
can be repeated once complete. Already-resting `settle` and already-attentive
`attend` requests are rejected. State reports describe the last completed pose;
`busy` and the step/phase indicate motion in progress.

Closing stdin requests normal shutdown. First SIGTERM requests normal shutdown;
a second SIGTERM or Ctrl-C stops without returning to zero. In manual mode,
`PALA_MAX_RUNTIME_S` requests normal shutdown when elapsed, so total execution
can extend beyond that value. Shutdown has a 30-second ceiling; failure disables
outputs. These are software paths, not an independent electrical emergency stop.

## Poses and recipes

`config/performances.json` is the version-controlled recipe library. Joint order
is yaw, pitch1, pitch2, roll, pitch3; values are joint degrees:

| Pose | Values |
|---|---|
| Zero | `[0, 0, 0, 0, 0]` |
| Rest | `[0, -40, 25, 0, 0]` |
| Attention | `[25, -15, 40, 10, 0]` |

Rest was revised after the camera probe showed only lower legs from the previous
`[0, -25, 0, 0, 0]` posture. Startup and settling reference the same named pose.
The new greeting approach, settling endpoint, and shutdown path need renewed
physical acceptance; earlier ratings describe the previous rest posture.

Each step specifies a full target or named pose, per-joint rate in degrees/second,
and a hold after commanded arrival. Joints start together but can finish at
different times. All steps are checked against YAML joint limits and servo
mapping before hardware initialization. Style multipliers are unity, matching
the workshop. Pitch3 stays zero throughout the checked-in library.

The accepted source trials and 5/5 operator ratings are recorded in the library
and [workshop report](workshop_2026-09-07.md). Integration removes workshop setup
and reset movements. Greeting adds a roll-to-10° transition after the accepted
wave to end at the attention pose. `attend` moves directly to that pose. Startup,
these transitions, and the complete interaction still require physical acceptance;
the three accepted trials do not establish repeatability of the composed demo.

## Supervised Jetson check

After deploying the reviewed files, stop other servo owners, including the
workshop. In the shared `pala` tmux session, from `~/pala`:

```bash
uv run python -m pala.main --manual --mode jetson_full --enable
```

It asks you to type `ZERO` before initializing hardware. Establish the known
starting posture first. Startup will automatically enter rest after confirmation.
The executor has no position feedback: even the zero setup and completion reports
are commanded estimates. After an emergency stop, re-establish the starting
posture before restarting.

For first physical acceptance, check startup → rest, then `greet`, then `settle`,
then `shutdown`. Evaluate the new transitions before running `demo`. Repeat the
accepted complete interaction three times and note any inconsistent motion,
interpretation, or mechanical concern. Record video only when the transitions
are satisfactory. Camera-triggered behavior is a later milestone.

## Evidence and failure behavior

With logging enabled (default), each run saves `interaction.jsonl`, per-step
`actions.jsonl`, perception logs, and copies of the recipe library and robot YAML
under `logs/runs/<run-id>/`. `interaction.jsonl` contains requests and control-rate
execution reports with commanded angles. These logs do not measure servo position.
`--performances PATH` selects an alternate library. Logs are local artifacts,
not automatically committed.

The control loop advances steps at 80 Hz, preserving short holds independently
of the 3 Hz behavior loop. The hardware deadman remains in the hardware loop.
Once commands have started, a stale command in manual mode disables outputs and
terminates the session rather than resuming motion. Executor failure also stops
the session. Hardware-thread cleanup disables outputs before slower resource
cleanup. Optional preview remains separate from gesture selection.
