# Gesture workshop

The workshop runs named pose sequences through the production trajectory
executor. It has no camera or model dependency. Start from the repository root:

```bash
uv run python -m tools.gesture_workshop
```

This uses dummy servos even if `robot.yaml` selects `jetson_full`. It executes
at real time so you can inspect the sequence and terminal interaction locally.
On the Jetson, after syncing this tool and confirming the correct calibration:

```bash
cd ~/pala
uv run python -m tools.gesture_workshop --hardware --enable
```

Both hardware flags are required. The tool does not initialize the servo backend
until the first confirmed run. Stop other servo controllers first. The first
hardware trial asks you to type `ZERO` to confirm the known physical starting
posture. This does not measure position or safely home from an unknown pose.

## Sequence and controls

Type `help` or `?` at `workshop>` for the command guide, editing examples,
units, comparison workflow, file locations, and stop behavior. Help never
commands motion. Use `--help` at launch for command-line options.

The initial greeting contains: prepare neutral and wait, small anticipation,
acknowledgment and pause, recovery and observation. Attention and settling are
also provided. Templates are tuning hypotheses, not accepted physical gestures.

Each numbered row displays the full resolved absolute joint targets in degrees,
a per-joint speed limit, and a pause after the move completes. Unspecified joints
keep their preceding commanded target. The first step must specify zero for all
joints. During execution the tool prints the active step and elapsed time; after
a trial it displays changes when comparing an edited candidate with that trial.

```text
show
edit 2 pitch3=-1
edit 3 pitch3=9 rate=12 hold=0.3
edit 4 rate=8 hold=3
run
rate 4 Clear acknowledgment; small rebound on recovery
save greeting-a
edit 3 pitch3=6
run
rate 5 Smaller version feels better
save greeting-b
load greeting-a
run
use attention
edit 2 yaw=-10
edit 3 roll=5 hold=3
run
rate 4 Attention direction is clear
save attention-a
use settling
edit 2 yaw=-10 roll=5 hold=2
edit 3 rate=8 hold=3
run
rate 4 Calm return
save settling-a
favorites
q
```

After `run`, the sequence is shown again. Press Enter at the separate run prompt
to execute, or type anything to cancel. Do not paste a whole example block into
the interactive prompt: issue each command and respond to its prompts separately.
Numbers are joint degrees, not raw servo degrees. `name="new step name"` edits a
label; `yaw=keep` removes that step's yaw override. Edits are validated before
replacing the current candidate. Change one parameter at a time for comparisons.
`use` and `load` replace the current candidate; save edits you want to retain first.

`neutral` is a separately confirmed return to zero at 8 degrees/s followed by a
two-second hold. It does not overwrite the current candidate, but is recorded as
its own trial. Ratings always refer to the most recent trial, including a neutral
trial. `save NAME` stores the current candidate, not the temporary neutral recipe.
Saving the same favorite name replaces that entry. Saving an edited/unrun candidate
is allowed, but it has no successful-trial provenance.

## Execution and stopping

- One executor and servo connection persist across trials. No position reset occurs
  between moves, pauses, and recovery.
- Every endpoint is checked against YAML joint limits and the servo mapping before
  any trial writes. Linear interpolation between valid endpoints stays within those
  numeric bounds. This is not a collision or load model.
- Rates have unit style multipliers so a displayed 12 degrees/s is not secretly
  changed by the default calm style. These are slew limits, not eased acceleration
  profiles. This first version deliberately uses pose moves rather than timed nod
  primitives, avoiding the older validator's duration mismatch.
- The workshop opts into a 0.000001-radian commanded completion tolerance for small
  poses. The main runtime retains its existing 0.02-radian tolerance. Neither is
  encoder feedback or proof of physical arrival; use pauses for observed settling.
- Each move has a calculated timeout. Timeouts, rejected commands, errors, and
  Ctrl-C disable outputs and abort instead of continuing to the next pose.
- Completed trials leave PWM holding the final pose while you review them. There
  is no background hardware thread or main-runtime deadman in this standalone
  tool. The PCA9685 retains PWM between writes. Do not leave a powered session
  unattended; normal exit, EOF, Ctrl-C, and SIGTERM attempt to disable outputs.
  SIGKILL, process/machine failure, or a failed I2C write cannot guarantee disable.
- `stop` or `q` disables outputs and exits without an automatic return. During a
  running trial use Ctrl-C; the text prompt is available between trials. Account
  for sag when outputs are disabled. Restart after an abort using a known posture.

## Evidence and replay

Each session creates a unique directory under `logs/workshop/` containing:

- `robot.yaml`: the configuration snapshot used for the session.
- `trials.jsonl`: recipes, starting commanded positions, results, ratings, notes,
  and whether hardware mode was enabled.
- `<trial-id>.jsonl`: per-tick commanded joint angles in radians, step/phase,
  elapsed time, status, and commanded enable state.

Use `--session-dir PATH` for a custom new directory; existing directories are
rejected. Favorites live separately in `config/gesture_favorites.json` by default
and can be redirected with `--favorites PATH`. This file is created only when you
save. Preserve/copy it back from the Jetson before deploying, because deployment
can delete Jetson-only files. It is a portable candidate file, not calibration.

Say the printed trial ID aloud on your phone video. Use the same video framing,
rate each candidate, then replay a favorite three times. Software logs describe
commands; video and observation establish actual motion and emotional readability.
The camera is not recorded by this tool.

Sync instructions: use the project's existing deployment workflow after preserving
any Jetson-only calibration or favorites. This tool does not deploy itself and does
not change deployment scripts. A Git commit alone does not update the Jetson mirror.
