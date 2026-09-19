# Telemetry sidecar

Telemetry runs separately from the four-loop runtime. Its default live view
(`runtime_core`) shows the camera preview, capture freshness/source health,
actions, and commanded joint angles. Joint values are commands, not measured
servo position or proof of hardware application/deadman health.

The supervised demo's semantic observations and execution reports live in its
own run logs; see [live interaction](../../docs/live_interaction.md). Old V3/V4
reasoning, case review, and dataset modules remain for historical replay. Current
runs do not emit the deleted planner's semantic streams.

## Live viewer

Start the intended runtime separately, following its operator guide. From the
repository root on the Mac, with your SSH alias configured:

```bash
uv run python -m tools.telemetry.mac_viewer \
  --mode live --focus runtime --jetson-host jetson-wifi
```

The GUI needs a Tk-compatible Python interpreter. If your environment requires
one, set `UV_PYTHON` to that interpreter's path. Do not launch another runtime
just to view an already-running demo.

Read-only diagnostic checks are available with:

```bash
uv run python -m tools.telemetry.doctor --help
```

Capture timestamps and viewer receive times are separate. Monotonic clocks from
different machines cannot be subtracted directly. A command sampled alongside a
preview frame does not demonstrate physical execution.

## Saved sessions and historical tools

Use a new local session directory for capture:

```bash
uv run python -m tools.telemetry.pipeline capture \
  --save-session logs/telemetry/session_001 --jetson-host jetson-wifi
uv run python -m tools.telemetry.pipeline compile logs/telemetry/session_001
uv run python -m tools.telemetry.mac_viewer \
  --mode replay --replay logs/telemetry/session_001 --focus runtime
```

The pipeline also offers `review`, `export`, and `report`. Inspect the relevant
subcommand's `--help` before using historical curation features. Indexed review
requires a compiled `session.db`; historical case/reasoning panels may be empty
for current sessions. Their labels are not current runtime model decisions.

Capture artifacts can include frames, event JSONL, indexes, a SQLite database,
and quality reports. Keep these local, including any camera images. Quality
reports describe captured data health, not physical safety or robot reliability.
Known issues from prior review are tracked in the [issue backlog](../../docs/bug_log.md).

Telemetry failure must not prevent core control/safety operation. Optional
[POV recording](../../docs/pov_recording.md) uses its own encoder and timing logs.
