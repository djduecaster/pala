# Lamp POV recording

## Stationary intro and stop-motion stills

After the live demo has returned to zero and shut down, record the downward
camera view without starting the robot runtime:

```bash
cd ~/pala
uv run python -m tools.record_intro --mode jetson_full
```

This opens only the configured camera. It does not initialize servos, command
zero, actively hold a pose, breathe, arm interaction, or contact Gemini. Leave
the lamp physically at the established zero posture. Stop other camera-owning
tools first; the live demo and intro recorder must not run together.

Recording starts as soon as camera frames arrive. The maximum duration is ten
minutes by default (`--seconds 1200` allows twenty minutes, up to one hour).
Use `--until-stopped` instead for no automatic duration limit; explicitly stop
the recording to finalize the video. Camera or encoder failures can still stop it.
Type `stop`, `q`, or `shutdown` then Enter, or press Ctrl-C, to finalize the MP4.
Here `shutdown` stops the camera recorder only; it never moves the lamp.

- `snap`: save a numbered, full-resolution JPEG after placing each group of parts.
- `status`: show recorded duration and still count.
- `help`: list commands.

Each launch creates a unique `logs/intro/<timestamp>_<id>/` folder containing
`pov.mp4`, the existing recorder timing/result files, and any `still_0001.jpg`
images. Use stills in the edit to build stop-motion without hands in frame.
The video is silent, at most 720p, 20 fps. Capture a light-switch sound separately
if desired. A failed or disconnected camera stops the tool after five seconds
without frames; the command reports failure while finalizing captured footage.

Mac smoke test (dummy images, no camera or servos):

```bash
uv run python -m tools.record_intro --seconds 2 --no-console
```

## Live interaction recording

Add `--record-pov` to the existing live demo command:

```bash
cd ~/pala
uv run python -m tools.live_interaction \
  --mode jetson_full --enable --record-pov \
  --gemini-key-file ~/.config/pala/gemini_api_key \
  --gemini-model "${PALA_GEMINI_MODEL:?Set an available image-capable Gemini model first}"
```

Normal zero confirmation applies. The live demo automatically arms after startup.
Recording starts when camera frames arrive, before automatic arming, and continues during gestures, breathing,
and snapshot pauses. Use `shutdown` to return to zero, disable outputs, and
finalize the video. It records at most ten minutes per launch. There is no audio.

The optional encoder requires `ffmpeg` with `libx264` on Jetson. A missing encoder
or recording failure logs a warning and does not stop lamp control. Check for
`POV recording started` and review `result.json` after a run; runtime success
alone does not mean video was saved successfully.

Outputs are printed at launch and stored under:

```text
logs/runs/<run>/pov_<id>/
  pov.mp4        H.264, at most 1280×720, 20 fps, no overlays
  session.json   dimensions, rate and monotonic start time
  frames.jsonl   source timing, repeated/stale frames, video timestamps
  encoder.log    encoder diagnostics
  result.json    written frame count and any recording error
```

The recorder shares the existing RGB frame cache; it never opens a second camera.
A separate worker resizes/encodes with bounded memory and two encoder threads.
The 20 fps file repeats frames when the source is slower. Its timing log marks
these and stale frames. The intro and live-demo entry points use fragmented MP4, writing playback
metadata early and fragments throughout the take. They disable the former
one-second encoding-lag abort; inspect frame timing afterward for any lag.
Direct PovRecorder callers retain the bounded-lag default. An interrupted
fragmented recording can still lose the unfinished tail. Quality, CPU headroom, and effective frame rate still need a
Jetson trial. This does not fix focus, glare, framing, or motion blur. Normal
shutdown finalizes MP4; power loss or forced process termination may leave it
unplayable.

Copy a chosen completed run to the Mac (substitute the printed run and POV IDs):

```bash
mkdir -p logs/pov_jetson
rsync -az jetson-wifi:~/pala/logs/runs/<run>/pov_<id>/ logs/pov_jetson/<run>/
```

Logs and videos stay Git-ignored. Trim the POV footage alongside your external
camera recording, then publish the reviewed edit as a GitHub Release asset and link it from the
README. Publishing is a separate step from local recording. Review the
frame log if you need to align motion and model events by monotonic timestamp.

For an entirely hardware-free smoke test on Mac:

```bash
PALA_MAX_RUNTIME_S=3 uv run python -m pala.main --record-pov
```

This records the dummy image source and verifies encoding, not the physical camera.
