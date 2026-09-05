# Telemetry scope

The implemented transport is an optional SSH sidecar. The current workshop
uses the existing runtime preview tap, JSONL logs, and Mac viewer. See
[the usage guide](../tools/telemetry/README.md) for commands and panel choices.

The supported live view shows camera imagery, capture freshness/source
health, actions, and commanded joint positions. It must not present a
command's enable flag as observed servo health or deadman state. Missing
execution feedback is unavailable, not healthy or zero.

Keep capture timestamps and viewer receive times distinct. Cross-machine
monotonic clocks are not comparable. The command attached to a preview frame
is a sampled command snapshot; it is not proof of physical execution.

Existing capture/replay, reasoning, trace analysis, and curation modules are
retained. V3/V4 reasoning tools serve historical sessions; current hold-only
runs do not produce those semantic streams. Full telemetry removal is
deferred until the workshop path has been exercised.

No WebRTC migration, new dashboard, or hardware-status publisher is part of
the cleanup. Any later structured execution-status output must be optional,
failure-isolated, and based on the loop that actually owns that status.
