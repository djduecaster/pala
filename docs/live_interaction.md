# Supervised desk interaction

This opt-in V1 demo uses shade-camera snapshots and Gemini observations to select
fixed local performances: notice, greet, excitement, directional pointing, and
settling. It has been exercised in supervised Jetson rehearsals. The default
`pala.main` runtime remains hold-only. This guide describes the checked-in
configuration; individual rehearsal results below refer to their recorded version.

## Launch

Stop other camera/servo owners, establish physical zero, and run in a named tmux
window using the [Jetson workflow](jetson_agent_workflow.md):

```bash
cd ~/pala
export PALA_GEMINI_MODEL='YOUR_AVAILABLE_GEMINI_MODEL_ID'
uv run python -m tools.live_interaction \
  --mode jetson_full --enable \
  --gemini-key-file ~/.config/pala/gemini_api_key \
  --gemini-model "$PALA_GEMINI_MODEL"
```

Choose an image-capable model available to your account; the placeholder above
is not a real model ID. Complete [key and model setup](attention_probe.md#key-and-model-setup-on-jetson)
first. Credentials stay in the local secret file. Confirm `ZERO` only with
the lamp at the known starting posture. Startup moves to rest, then automatically
arms with a five-second countdown. Add `--record-pov` for optional
[POV recording](pov_recording.md).

For a Mac smoke test with dummy camera/servos and no API call:

```bash
uv run python -m tools.live_interaction --mode dev --probe-mock
```

Mock observations are uncertain and do not trigger social gestures. Automated
tests separately inject scripted observations to exercise dispatch.

## Interaction and controls

1. Enter the tested close-range seated camera view and work. Presence triggers
   quiet noticing once. Look toward the shade camera to invite greeting.
2. A clear thumbs-up with attention at rest or after noticing accepts greeting
   followed by excitement. No second cue is required between those movements.
   A thumbs-up after greeting can also trigger excitement, once per interaction.
3. After greeting or excitement, point clearly left or right in the unmirrored
   camera image. Hold the cue long enough for a snapshot. Release your hand
   between cues; another point requires a valid present-person/gesture-none
   observation and at least eight seconds since the last accepted point.
4. Return to work. Two valid away/absent observations spanning at least four
   seconds trigger settling. Before greeting, the away grace is 20 seconds.
5. Normal settling automatically rearms after a 0.75-second stationary pause.
   Presence memory prevents repeated notice/settle cycles for someone still
   working. Two valid absent-at-rest observations spanning four seconds permit
   a future arrival to trigger notice again.

Each arm has a 180-second observation window. Expiry disarms social decisions;
the current posture remains and enabled idle breathing may continue at rest.

| Command | Effect |
|---|---|
| `arm` | Start a five-second countdown; during breathing, wait for rest midpoint first |
| `pause` / `cancel` | Disarm observation, suppress automatic rearming, and suspend breathing |
| `reset` | Disarm and suspend breathing; settle if idle away from rest; during motion, issue again after completion |
| `breathe-on` / `breathe-off` | Enable/suspend subsequent idle cycles; active cycle finishes at midpoint |
| `status` | Report runtime and observation state |
| `shutdown` / `q` | Controlled return to zero, disable outputs, and exit |
| `stop` / Ctrl-C | Disable immediately without recovery motion |

After pause/reset, use `arm` to resume observations and `breathe-on` if desired.

## Current motion and observation settings

`config/desk_performances.json` is the live motion source of truth. Joint order
is yaw, pitch1, pitch2, roll, pitch3, in degrees:

| Pose | Values |
|---|---|
| Rest | `[0, -40, 25, 0, 0]` |
| Attention | `[25, -15, 30, 10, 0]` |
| Excited endpoint | `[35, -15, 25, 10, 0]` |
| Point left / right | `[70, -5, 15, 0, 0]` / `[-10, -5, 15, 0, 0]` |

Pointing holds three seconds then returns to attention. It follows a general
direction, not a triangulated target. Excitement leans pitch1 to +10, recoils
through -1 to -30 with staged yaw/pitch2 sweeps, then returns to its endpoint.

Idle breathing cycles pitch1 ±8 degrees around rest with pitch2 compensation;
every third cycle adds yaw ±6 degrees. Pitch3 remains zero. Current 16°/s rates
produce approximately two seconds of motion, followed by stationary waiting and
inference. This faster motion setting is distinct from the four-second breathing
cycles recorded in the September 12 rehearsal snapshots.

Requests use `reasoning_effort=minimal`, a 0.25-second inter-request pause, one
request in flight, and a 0.75-second stationary-frame wait. The lamp holds through
the model response. Capture cadence therefore includes motion and inference;
it is not continuous tracking or a fixed frame interval.

Gemini reports presence, apparent attention, visible gesture, and brief evidence.
Local code owns stage transitions, fixed recipes, limits, and execution. A
six-second image-age gate, current-camera freshness, and plan/stage/generation
matching reject invalid or obsolete observations. The worker cannot actuate.
Requests have a 20-second provider timeout and no automatic transport retries.
A later fresh observation may be requested after failure.

Early thumbs-up accepts a two-movement response with a 30-second continuation
expiry. Greeting must complete at its expected endpoint before excitement.
Pause, reset, disarm, rejection, failure, shutdown, or unexpected completion
clears the continuation. Lowering a hand during greeting does not cancel the
already accepted response; no image is interpreted between those movements.

## Rehearsal evidence and limitations

The September 12 full-choreography review recorded probe
`20260912_153135_3334c8` and runtime `20260912_153200`: notice, greeting, staged
excitement, both pointing directions, settling, breathing, sway, and a clean
controlled shutdown. The operator called it “a very successful run.” Across 32
valid responses, median request latency was 2.67 seconds; one stale result was
rejected. These are local rehearsal records, not a published benchmark.

The later minimal-thinking snapshot (`20260912_154543_5a0b12`, runtime
`20260912_154604`) recorded 17 valid responses with median 1.76 seconds
(range 1.37–4.51), no stale rejections, and completion of greeting, excitement,
left pointing, and settling. Operator feedback was “pretty much perfect.” That
snapshot was taken during an active run and does not establish final shutdown or
right pointing for that run. Latency comparisons were not same-image experiments.

Physical success is based on operator observation. Logs show commanded angles
and completion, not measured servo position, physical motion onset, or reliability
across users, lighting, and camera views. Earlier gesture ratings in the
[workshop report](workshop_2026-09-07.md) apply to those recipe versions.

Model observations and JPEGs are local under `logs/attention_probe/<session>/`;
command traces and configuration snapshots are under `logs/runs/<run>/`. Logs,
keys, and raw media are not required published assets. Simulator playback
validates configured motion, not physical clearance or Gemini recognition:

```bash
uv run python -m tools.primitive_sim.social_scene
```
