# Supervised desk interaction

This opt-in V1 demo uses the physically reviewed desk gestures and Gemini
observations. Default runtime behavior and the earlier live-greeting test remain
available. A first live trial exercised noticing, greeting, and settling; the separate
thumbs-up-triggered excited response remains unverified in the live sequence.

## Launch on Jetson

Stop other servo owners and establish zero, then run in a named tmux window:

```bash
cd ~/pala
/home/dylan/.local/bin/uv run python -m tools.live_interaction \
  --mode jetson_full --enable \
  --gemini-key-file ~/.config/pala/gemini_api_key \
  --gemini-model gemini-3.6-flash
```

Confirm `ZERO`; startup moves to the established rest (-40/+25). Wait for
`SOCIAL READY`. `arm` gives five seconds before the first image. One arm permits
one interaction, with an observation window of 180 seconds:

1. Start outside the image, then sit in the tested close-range seated position
   and work. Visible presence triggers the quiet noticing gesture once.
2. Look toward the shade camera. A fresh observation of attention triggers
   the accepted greeting once. If a clear thumbs-up accompanies that attention
   at rest or after noticing, the lamp greets and then performs excitement
   automatically. At rest, this explicit invitation skips the quiet notice.
3. If you have not already given an early thumbs-up, you can give one after
   greeting to trigger excitement. Hold the cue for several seconds until it
   is observed; snapshots are not continuous tracking. An accepted early cue
   does not need to be repeated or held through the greeting.
4. Return to work. Two valid away/absent observations spanning at least four
   seconds trigger settling. Brief look-aways and uncertain results do not count
   as confirmed disengagement. Before greeting, the away grace is 20 seconds so
   the initial noticing gesture does not immediately disengage.
5. After settling, observation stops. Explicitly `arm` for another interaction.

`pause` stops observation and invalidates pending decisions without aborting
motion. `reset` disarms and settles if currently idle away from rest; during
motion it only disarms, so issue reset again after completion. `status` reports
runtime and observation state. `shutdown`/`q` returns to zero and disables;
`stop`/Ctrl-C disables immediately without recovery motion. If 180 seconds
expires, observations stop and the current motion/hold remains; reset or shut
down explicitly. No fresh/valid result means no model-triggered movement.

## Evidence and boundaries

`config/desk_performances.json` contains the reviewed workshop segments with
preparation/reset removed. Notice: 4/5, trial b710447c26b0. Greeting: 5/5,
4c1f6a2d62ff. Excited: 5/5, 996eff8ac06a, at the accepted 70% of original
simulator rates. Settling: 5/5, 3df8f9c049a1. Full workshop composition:
c278313a5769, operator said it was pretty great (no numerical rating).

The accepted excited endpoint is yaw=35/pitch1=-15/pitch2=25/roll=10/pitch3=0.
Greeting ends at yaw=25/pitch1=-15/pitch2=30/roll=10/pitch3=0. Alternate
settling approaches from noticing or greeting release their own pose first,
then center and rest; those new paths still need physical review.

Gemini reports only person, apparent attention, visible hand gesture, and brief
evidence. Local state controls gesture selection, one-shot limits, and motion.
The network worker cannot actuate. Capture runs only while stationary, with
0.75 seconds after completed motion before a new frame can be used. There is
one request in flight, a six-second image-age gate, current-camera freshness
checking, and plan/stage/generation matching. Pause, changed context, failed
requests, and invalid responses cannot trigger a gesture. Requests use the
existing 20-second provider timeout with no automatic retries.

Logs and JPEGs are under `logs/attention_probe/<session>/`; command traces and
calibration snapshots are under `logs/runs/<run>/`. Credentials stay on Jetson.
The model receives still images from the moving shade camera. Visibility at
notice, greeting, and excited poses and reliable thumbs-up recognition remain
live-test questions; success of the resting-view probe does not establish them.

For a Mac smoke test use `--mode dev --probe-mock` without `--enable` or a key.
Mock responses are uncertain and do not cause social gestures. Automated tests
inject scripted observations separately to exercise the full runtime dispatch.

## First live review — 2026-09-07

Probe `20260907_191959_d600fe`, runtime `20260907_192021`: seven valid model
responses; median request latency 3.09 seconds (range 2.50–5.96). The first
response was discarded at 6.07 seconds image age, exceeding the six-second
freshness gate. The next observation triggered notice. A later image clearly
showed attention plus thumbs-up; because the state was still noticed, local
logic selected greet. Two post-greeting images showed attention elsewhere and
triggered settle_attention. No excite command occurred. The sequence at the time of that trial
required a new thumbs-up observation after greeting; do not describe this trial
as validating the excited response end to end.

Capture-to-first-command timings: notice 3.21s, greet 3.22s, settling 3.58s.
The local decision-to-command delays were 10–22ms. These are not measured servo
motion onset or time since the operator first presented a cue. No execution
failures were present. Requested shutdown completed zero and exited cleanly.
Operator feedback: mostly worked, looked reasonable, went pretty well; lamp
glare was uncomfortable and a thin white disk was suggested. No numeric rating
was given. No optical modification has been made or evaluated.

Copies and a structured review are under
`logs/attention_probe_jetson/20260907_191959_d600fe/`; runtime evidence is under
`logs/live_greeting_jetson/20260907_192021/`. Review the interaction policy for
an early thumbs-up, or explicitly present the cue again after the greeting in
the next trial. Preserve the current freshness gate pending further evidence.

## Early thumbs-up response

An eligible, valid, fresh observation of a person looking toward the camera
with a thumbs-up, at rest or after noticing, accepts `greet -> excite` as one
response. Greeting must complete at its expected attentive endpoint before
excitement is dispatched once. No second image or hand gesture is needed.
The accepted continuation expires after 30 seconds and is cleared by pause,
reset, disarm, motion rejection, failure, shutdown, or unexpected completion.
The 180-second arm window still applies. Later observations cannot repeat the
excited response in the same interaction.

The six-second freshness gate applies when accepting the original cue. The
second movement completes the accepted response, rather than reusing that image
as new evidence. The camera is not consulted between these two movements, so a
person lowering their hand or looking away during greeting does not cancel the
accepted response. After excitement, fresh observations resume for settling.
`sequence_accepted` and `sequence_continuation` log the originating trial ID.
This revised policy is software-tested; physical end-to-end acceptance is pending.
