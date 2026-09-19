# Gemini attention probe

This experiment starts the existing four-loop manual runtime, enters rest, and
holds there while you collect four labeled observations. Gemini interprets apparent
attention toward the lamp; its answers never request movement. Startup and normal
shutdown are the only motion permitted in probe mode. The camera is on the shade,
Capture uses 1280×720 MJPEG at a requested 30 FPS. Stationary comparisons showed
wider horizontal coverage than 640×480 and essentially the same coverage as
1920×1080. The preview is downscaled without cropping.

This first experiment uses only the stable rest viewpoint. Current rest is
`[yaw=0, pitch1=-40, pitch2=+25, roll=0, pitch3=0]` in joint degrees.

## Offline Mac check

From the repository root:

```bash
uv run python -m tools.attention_probe --mode dev --probe-mock
```

Dummy camera, dummy servos, and explicitly labeled mock responses; no API call or
key needed. The mock always returns uncertain/wait and does not analyze the image.
Wait for `PROBE READY`, then type a number and press Enter.

| Number | Your position at the end of the countdown |
|---|---|
| 1 | Outside the frame: empty scene. |
| 2 | Walk into view without looking at the camera. |
| 3 | Look directly toward the camera. |
| 4 | Look toward the wall beside the camera. |

Each number starts a **five-second countdown**. Capture uses the first available
fresh frame taken after the deadline; scheduling/camera delay is logged. A frame
must be at most 500 ms old. If none arrives within two additional seconds, the
trial ends without an API call. Stay in position until `SNAPSHOT saved` appears.
Then you can return to the keyboard while Gemini responds.

Repeat numbers freely, one trial at a time. New trial requests while counting down
or awaiting a response are rejected rather than queued. Each request contains only
the image and a fixed resting-state prompt; the scenario number and description
are saved locally and deliberately excluded from the model input.

## Key and model setup on Jetson

Use a Gemini API key, not a browser login. Either export `GEMINI_API_KEY` in the
shell launching the runner, or save the key alone in a file outside the repository,
for example `~/.config/pala/gemini_api_key`, and use `--gemini-key-file` below.
`PALA_GEMINI_API_KEY` and `GOOGLE_API_KEY` are also recognized. No key is accepted
on the command line as a literal value or written to probe logs.

To create the key file without putting the key in shell history (Jetson Bash):

```bash
install -d -m 700 ~/.config/pala
(umask 077; read -r -s -p 'Gemini API key: ' key; echo;
 printf '%s' "$key" > ~/.config/pala/gemini_api_key)
chmod 600 ~/.config/pala/gemini_api_key
```

For a temporary environment variable instead (Jetson Bash):

```bash
read -r -s -p 'Gemini API key: ' GEMINI_API_KEY; echo
export GEMINI_API_KEY
export PALA_GEMINI_MODEL='YOUR_AVAILABLE_GEMINI_MODEL_ID'
```

If you maintain exports in `~/.config/pala/env.sh`, source it yourself in that
terminal first. The probe does not read or execute shell configuration files.
There is deliberately no hardcoded model version: use an image-capable Gemini
model available to your account. `--gemini-model` overrides `PALA_GEMINI_MODEL`.
Missing credentials/model fail before hardware initialization. The key's validity
and model access are checked only when the first trial makes a real request.

## Physical session

Stop the existing manual/workshop runtime first; do not start competing servo or
camera owners. After returning to and establishing zero, run in the shared `pala`
tmux session from `~/pala`:

```bash
uv run python -m tools.attention_probe --mode jetson_full --enable
```

Or with a key file:

```bash
uv run python -m tools.attention_probe --mode jetson_full --enable \
  --gemini-key-file ~/.config/pala/gemini_api_key \
  --gemini-model YOUR_AVAILABLE_GEMINI_MODEL_ID
```

Confirm `ZERO` when prompted. Startup moves to rest automatically. The ground-facing
zero view is not submitted. Wait for `PROBE READY` and use 1–4. Only the triggered
snapshots are sent to Google's Gemini API; there is no continuous upload. These
images and the responses are also saved locally.

- `help` / `?`: show the four scenarios and common commands.
- `status`: show runtime posture and progress.
- `cancel`: cancel a countdown; an already-sent API request cannot be recalled.
- `shutdown` / `q` / `quit`: return to zero, hold two seconds, disable, and exit.
- `stop` / `abort` / Ctrl-C: disable and exit without recovery motion.
- Closing input or first SIGTERM requests normal shutdown, as in the manual runner.

Gesture commands (`greet`, `demo`, etc.) are blocked in probe mode. The auxiliary
network worker cannot access the gesture gate or actuators. It uses a 20-second
request timeout and disables SDK retries; control and hardware loops continue
while waiting. Shutdown does not wait for that worker. An unfinished request is
logged as abandoned on exit; the remote service may still finish processing it.

## Results

Each trial prints person presence, apparent attention, `acknowledge` or `wait`, a
short visible-evidence sentence, and elapsed request/validation latency. These
are proposals only. An invalid or failed response is labeled explicitly, not
silently converted into a valid wait decision. A single snapshot cannot establish
walking direction or reliably measure precise eye gaze; sideways-looking case 4
is intentionally a useful ambiguity test.

Outputs live in a unique `logs/attention_probe/<timestamp>_<id>/` directory:

- `session.json`: model, timing, scenarios, mock flag, and posture context.
- `prompt.txt`: exact fixed prompt.
- `performances.json`: recipe snapshot, including exact rest angles for this run.
- `<trial-id>.jpg`: exact JPEG submitted (up to 960 pixels on its longest side).
- `trials.jsonl`: scenario labels, capture age/delay, provider response content and
  usage when available, validated decision, failures, and measured latency.

Normal runtime logs still live under `logs/runs/`. Logs/images are ignored by git.
Use `--probe-output PATH` to select another parent directory. Keep images private
unless you deliberately select them for a portfolio artifact.

The implementation reuses PALA's retained model transport and Google's documented
[OpenAI-compatible image input and JSON output interface](https://ai.google.dev/gemini-api/docs/openai).
It does not restore the removed behavior system. This first experiment tests
acknowledge-versus-wait from rest; sustained attention and disengagement are later
contracts, after we inspect the camera's attention posture.
