# Supervised live Gemini greeting

This test lets you judge response timing by watching the lamp move. The operator
chose to skip additional numbered-camera and shadow-only trials. It is explicitly
armed and allows one Gemini-triggered greeting per arm, not autonomous ongoing
interaction. Rest is `[0, -40, 25, 0, 0]` joint degrees, accepted for a seated,
close-range demo; capture is 1280×720 MJPEG.

Stop the framing workshop and camera-only runtime before launching. This runner
owns both camera and servo output, and continues publishing the existing preview
tap for the Mac browser. From `~/pala` on Jetson:

```bash
/home/dylan/.local/bin/uv run python -m tools.live_greeting \
  --mode jetson_full --enable \
  --gemini-key-file ~/.config/pala/gemini_api_key \
  --gemini-model gemini-3.6-flash
```

Confirm established physical zero. Startup moves to rest. Wait for
`LIVE TEST READY at rest, disarmed`, then type `arm` and Enter. You have five
seconds to get situated. During the next 60 seconds, one current rest-view image
at a time is sent to Gemini, with a one-second pause after each reply. The first
valid, eligible `acknowledge` requests the accepted greeting through the existing
manual behavior gate. The lamp finishes at attention; observations stop. No
automatic settling or rearming occurs.

The gate requires idle rest, a healthy fresh camera frame, matching posture/request
generation, and an image no more than six seconds old when deciding to move. A
pause, shutdown, or posture change invalidates pending decisions. Invalid responses,
unknown attention, waits, and network failures do not trigger motion. API requests
retain the probe's 20-second timeout and zero SDK retries. Failed trials can be
rearmed once the pending call finishes. Image age includes inference time; freshness
checks cannot prove someone has not turned away while Gemini was answering. This
experiment measures that limitation rather than hiding it with another detector.

Commands:

- `arm`: begin the five-second positioning delay and one-greeting observation window.
- `pause` / `cancel`: disarm and discard pending decisions; an active gesture finishes.
- `reset`: disarm and request settling back to rest; waits for completion before rearm.
- `status`: show posture, armed state, and pending request status.
- `help`: show these commands.
- `shutdown` / `q`: disarm through shutdown state, return to zero, then disable and exit.
- `stop` / Ctrl-C: disable and exit without recovery motion.

After the first greeting, evaluate both reaction timing and the camera's attention
view. If suitable, use `reset` then `arm` for another trial. The new rest-to-greeting
and settling transitions still require operator acceptance.

Logs use `logs/attention_probe/<session>/`, with `session.json` identifying
`mode=live_greeting` and `motion_from_model=true`. Snapshots, prompt, recipes,
responses, decision eligibility, image age, and motion-request acceptance are
recorded. Runtime control logs remain under `logs/runs/`. These are commanded
positions, not physical feedback.

For an offline wiring check, use `--mode dev --probe-mock` and omit hardware/key
flags. Mock responses deliberately remain uncertain/wait, so they never greet;
automated integration tests inject an acknowledgment to verify dispatch with dummy
hardware. No live physical acceptance is implied by those tests.
