# PALA next steps

Current milestone: a clean local workshop baseline, followed by three useful
physical gestures. The lamp can remain unpowered during software cleanup.

## Session 1: software cleanup

- Retire tools and tests tied to deleted V4 behavior contracts.
- Preserve primitive simulation, playback, and joint checking without V4.
- Isolate simulated time from the runtime clock.
- Default telemetry to current capture/action/command data; retain historical
  reasoning and curation only as opt-in tools.
- Validate configuration and complete servo command batches before use.
- Make gesture dry-runs independent of hardware initialization.
- Pass the full pytest suite, bounded dummy runtime, and retained tool checks.

Completion evidence is recorded in [the session report](session_1_cleanup.md).

## Session 2: three physical performances

- Inspect current hardware, starting posture, camera coverage, and tested
  calibration limits before enabling motion.
- Tune greeting, curiosity/attention, and settling using the existing demo
  and primitive tools, changing one motion parameter at a time.
- Record chosen parameters and repeated outcomes; commanded completion is
  not physical position feedback.
- Resolve gesture interruption/completion semantics before connecting a
  multi-part skill runner. Current executor behavior is latest-intent wins.

## Next: a narrow camera-driven interaction

- Define a small model decision contract and execution-aware skill runner.
- Evaluate Gemini with representative images/sequences in a motion-disabled
  probe; measure parsing, latency, freshness, and repetitive decisions.
- Connect observe → greet once → attend → settle, retaining the four loops
  and `PerceptionState -> ActionPlan -> HardwareCommand` contracts.
- Demonstrate repeated trials, then publish a short video and reproducible
  portfolio snapshot with honest limits.

Speech, object search, task-light localization, persistent autobiographical
memory, local detection, fine-tuning, and V2 hardware are outside this release.
Future work must not become a prerequisite for the first repeatable greeting.
