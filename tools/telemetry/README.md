# Telemetry Tools (V2)

This directory contains sidecar telemetry tooling for live debugging and demos.
It is intentionally separate from core runtime control/safety logic.

## V2 Shipment Overview
- Signal packs for role-based debugging (`reasoning_live`, `reasoning_failures`, `demo_overview`, plus legacy packs).
- Field-level payload filtering (`--field-filter source.path<op>value` with `=`, `!=`, `<`, `>`, `~`).
- Timeline and memory stream support:
  - `logs/orchestrator_timeline.jsonl` (primary)
  - `logs/orchestrator_memory.jsonl` (optional/legacy depending on planner mode)
- Reasoning-first terminal UX:
  - `Reasoning Stream`, `Request Detail`, `Reasoning Health` panels
  - keyboard command bar and panel presets
  - redacted/truncated reasoning snippets for safe live demos
- Correlated request traces:
  - `Trace List` and `Trace Detail` panels for req-level root-cause drill-down
  - correlation by `req_id`, with bounded time-window fallback for missing ids
- Transport health and robustness improvements:
  - warning coalescing
  - stale stream reconnect
  - worker crash reporting
  - oversized frame guards
- Reproducible session bundles:
  - Agent-side capture (`--capture-dir`)
  - Viewer-side local capture (`--save-session`)
  - Offline replay (`--replay`)

## Compatibility Notes
- Existing Phase 1 commands still work (for example `--video-source tap` and text/video dashboard usage).
- V2 adds optional controls; you can adopt incrementally by adding `--pack`, `--field-filter`, and capture/replay flags.

## Architecture (V2)
- Mac: `tools.telemetry.mac_viewer`
  - Starts Jetson sidecar over SSH for live mode.
  - Can run fully offline in replay mode (`--replay`).
- Jetson: `tools.telemetry.jetson_agent`
  - Tails log sources, reads optional video, emits NDJSON telemetry stream.
- Runtime: unchanged core loops; telemetry remains optional and failure-isolated.

## Core Signals
- Mac-run live text dashboard plus optional live video window with overlays.
- In-window 2D lamp command visualizer panel (joint angles + simplified lamp sketch).
- Jetson streamed telemetry via SSH sidecar agent.
- Inputs:
  - `logs/perception.jsonl`
  - `logs/actions.jsonl`
  - `logs/orchestrator_timeline.jsonl` (primary planner lifecycle stream)
  - `logs/orchestrator_memory.jsonl` (optional)
  - `tegrastats`
  - filtered `journalctl` lines
  - optional live frames from runtime preview tap (`logs/telemetry/preview/latest.jpg` + metadata)

## Start Here (V2 Quickstart)
1. On Jetson, run runtime:
```bash
cd ~/pala
PALA_LOG_LEVEL=INFO uv run python -m pala.main
```
2. On Mac, run viewer (default is reasoning-first UI + `reasoning_live` pack):
```bash
cd /Users/djduecaster/development/pala
UV_PYTHON=/opt/homebrew/bin/python3.10 uv run python -m tools.telemetry.mac_viewer --jetson-host jetson
```
3. Add targeted visibility as needed:
```bash
# Reasoning-focused live view (explicit)
uv run python -m tools.telemetry.mac_viewer --pack reasoning_live

# Failure triage view
uv run python -m tools.telemetry.mac_viewer --pack reasoning_failures

# Show only low-confidence planner actions
uv run python -m tools.telemetry.mac_viewer --pack planner_debug --field-filter 'actions_log.data.confidence<0.5'
```

## Session Capture and Replay (V2)
Capture on Mac (recommended for post-training prep):
```bash
uv run python -m tools.telemetry.mac_viewer \
  --pack reasoning_live --pack planner_debug \
  --save-session logs/telemetry/session_001 \
  --capture-frames keyframes
```

Replay locally:
```bash
uv run python -m tools.telemetry.mac_viewer --replay logs/telemetry/session_001 --replay-speed 1.0
```

Capture directly on Jetson sidecar:
```bash
uv run python -m tools.telemetry.jetson_agent \
  --pack all \
  --capture-dir logs/telemetry/session_jetson_001 \
  --capture-frames keyframes
```

Bundle contents:
- `manifest.json` (schema/version + metadata)
- `events.jsonl` (telemetry stream)
- `index.json` (frame index)
- `reasoning_index.json` (lightweight reasoning event index)
- `trace_index.json` (correlated request trace index)
- `frames/` (captured JPEG files when enabled)

## Prerequisites
- Jetson host alias reachable from Mac (default host: `jetson`).
- Jetson repo path (default: `~/pala`).
- Runtime logs enabled in `config/robot.yaml` for perception/action JSONL.
- Runtime preview tap enabled in `config/robot.yaml` (`telemetry_preview.enabled: true`) for `--video-source tap`.
- `uv` installed on both Jetson and Mac.
- macOS windowed mode: Homebrew Python 3.10 + Tk support (`python@3.10`, `python-tk@3.10`, `tcl-tk@8`).

Quick connectivity check from Mac:
```bash
ssh jetson 'echo ok $(hostname)'
```

If you do not have a `jetson` SSH alias, pass hostname/IP explicitly:
```bash
uv run python -m tools.telemetry.mac_viewer --jetson-host 192.168.1.50
```

## First-Time Setup
1. Jetson: run PALA runtime (this writes perception/action logs and preview tap files):
```bash
cd ~/pala
PALA_LOG_LEVEL=INFO uv run python -m pala.main
```

2. Mac: prepare a Tk-compatible `uv` environment (required for video window):
```bash
cd /Users/djduecaster/development/pala
UV_PYTHON=/opt/homebrew/bin/python3.10 uv sync
export UV_PYTHON=/opt/homebrew/bin/python3.10
```

3. Mac: verify window stack once:
```bash
uv run python - <<'PY'
import tkinter as tk
from PIL import Image, ImageTk
root = tk.Tk(); root.withdraw()
ImageTk.PhotoImage(Image.new("RGB", (8, 8)))
root.destroy()
print("Tk video window stack OK")
PY
```

4. Mac: run telemetry viewer:
```bash
# Windowed viewer (recommended)
UV_PYTHON=/opt/homebrew/bin/python3.10 uv run python -m tools.telemetry.mac_viewer --jetson-host jetson --jetson-dir '~/pala' --video-source tap

# Headless viewer (terminal only)
UV_PYTHON=/opt/homebrew/bin/python3.10 uv run python -m tools.telemetry.mac_viewer --jetson-host jetson --jetson-dir '~/pala' --video-source tap --no-video-window
```

Optional: add this to your shell profile (`~/.zshrc`) so you do not have to prefix commands:
```bash
export UV_PYTHON=/opt/homebrew/bin/python3.10
```

## Judge/Demo Checklist
Use this sequence for a clean demo with video:
1. Start runtime on Jetson first.
2. Start telemetry on Mac with `--video-source tap`.
3. Confirm dashboard shows:
   - `connected=True`
   - `Video` section with `frame_id=...` and increasing `received=...`
   - `Event Counts` includes `video_frame` increasing over time.
   - `Command` section with joint angles.
4. Resize video window as needed by dragging edges/corners.
5. If windowed mode is not available, switch to `--no-video-window` and continue terminal telemetry.

## Where To Run What
- Jetson:
  - Run the main runtime in one terminal.
- Mac:
  - Run the telemetry viewer in a separate terminal.
  - The viewer starts the Jetson telemetry agent over SSH automatically.

Example runtime on Jetson:
```bash
cd ~/pala
PALA_LOG_LEVEL=INFO uv run python -m pala.main
```

Example telemetry viewer on Mac:
```bash
uv run python -m tools.telemetry.mac_viewer --jetson-host jetson
```

List built-in signal packs:
```bash
uv run python -m tools.telemetry.mac_viewer --list-packs
uv run python -m tools.telemetry.jetson_agent --list-packs
```

`jetson_dir` path notes:
- Default is `~/pala` on the Jetson.
- If you pass a local Mac absolute path like `/Users/.../pala`, the viewer remaps it to `~/pala` automatically.
- If you pass `~/pala` manually from your shell, quote it (`'~/pala'`) to avoid local expansion.

Useful flags:
```bash
# Reasoning-first UI mode (default)
uv run python -m tools.telemetry.mac_viewer --ui-mode reasoning

# Classic layout mode
uv run python -m tools.telemetry.mac_viewer --ui-mode classic

# Lower update rate if terminal redraw is heavy
uv run python -m tools.telemetry.mac_viewer --refresh-hz 2

# Disable journal tailing
uv run python -m tools.telemetry.mac_viewer --no-journal

# Choose signal packs (repeatable)
uv run python -m tools.telemetry.mac_viewer --pack reasoning_live --pack demo_overview

# Add field-level filters
uv run python -m tools.telemetry.mac_viewer --field-filter 'actions_log.data.confidence<0.5'
uv run python -m tools.telemetry.mac_viewer --field-filter 'timeline_log.data.type~(req_start|req_end)'

# Timeline file path (primary orchestrator structured log)
uv run python -m tools.telemetry.mac_viewer --timeline-log logs/orchestrator_timeline.jsonl

# Optional legacy memory log (if your branch still emits it)
uv run python -m tools.telemetry.mac_viewer --memory-log logs/orchestrator_memory.jsonl --timeline-log logs/orchestrator_timeline.jsonl

# Disable tegrastats polling
uv run python -m tools.telemetry.mac_viewer --no-tegrastats

# Disable live video stream/window (text telemetry only)
uv run python -m tools.telemetry.mac_viewer --no-video

# If camera is busy, use synthetic dummy video frames
uv run python -m tools.telemetry.mac_viewer --video-source dummy

# Use runtime preview tap explicitly (default)
uv run python -m tools.telemetry.mac_viewer --video-source tap

# Fallback: open camera directly in telemetry agent (requires camera not in use)
uv run python -m tools.telemetry.mac_viewer --video-source gst

# Resize video stream and preview
uv run python -m tools.telemetry.mac_viewer --video-max-width 480 --video-max-height 270 --video-window-scale 1.2

# Tighten frame size limits (agent output + viewer decode guard)
uv run python -m tools.telemetry.mac_viewer --video-max-bytes 500000 --max-frame-bytes 1200000

# Reconnect more aggressively on stale streams
uv run python -m tools.telemetry.mac_viewer --stale-timeout-s 8 --reconnect-delay-s 1 --reconnect-backoff 1.4 --reconnect-max-delay-s 10

# Tune warning coalescing / worker restart cadence
uv run python -m tools.telemetry.mac_viewer --warning-throttle-s 3 --worker-restart-delay-s 1.0

# Limit dashboard to selected panels
uv run python -m tools.telemetry.mac_viewer --panel system --panel transport --panel logs

# Reasoning snippet controls
uv run python -m tools.telemetry.mac_viewer --reasoning-redact on --reasoning-snippet-max-chars 200

# Trace correlation tuning
uv run python -m tools.telemetry.mac_viewer --trace-match-window-s 2.0 --trace-max-events 1000

# Save a local reproducible session bundle on Mac (events + optional frame refs)
uv run python -m tools.telemetry.mac_viewer --save-session logs/telemetry/session_001 --capture-frames keyframes

# Replay an existing local bundle
uv run python -m tools.telemetry.mac_viewer --replay logs/telemetry/session_001 --replay-speed 1.5

# The window is also mouse-resizable (drag edges/corners)

# Disable the 2D lamp panel if needed
uv run python -m tools.telemetry.mac_viewer --no-lamp-panel

# Set lamp panel width (pixels)
uv run python -m tools.telemetry.mac_viewer --lamp-panel-width 300
```

Keyboard controls in reasoning mode:
- `?`: toggle hotkey help
- `h/l`: move focus panel
- `j/k`: previous/next reasoning event
- `u/i`: previous/next trace
- `o`: focus trace detail panel
- `p`: pin/unpin selected trace
- `f`: cycle reasoning filter (`all`, `errors`, `slow`)
- `r`: toggle reasoning redaction
- `1/2/3`: panel presets

## Tk/Tcl Video Window Troubleshooting (macOS)
If telemetry works but the video window fails with an `init.tcl` error:

```bash
cd /Users/djduecaster/development/pala
UV_PYTHON=/opt/homebrew/bin/python3.10 uv sync
export UV_PYTHON=/opt/homebrew/bin/python3.10
uv run python - <<'PY'
import tkinter as tk
from PIL import Image, ImageTk
root = tk.Tk(); root.withdraw()
ImageTk.PhotoImage(Image.new("RGB", (8, 8)))
root.destroy()
print("Tk video window stack OK")
PY
```

Then re-run telemetry viewer without `--no-video-window`.

## Common Issues
- `connected=False`:
  - Check SSH host and key access.
  - Verify `--jetson-dir` points to the repo on Jetson.
- Frequent disconnect/reconnect loops:
  - Verify SSH alias is non-interactive (`BatchMode=yes` compatible).
  - Increase `--stale-timeout-s` to reduce aggressive stale reconnection.
  - Increase `--ssh-connect-timeout-s` if DNS/network setup is slow.
- Memory/timeline panels show `no data yet`:
  - Confirm timeline logging path in `config/robot.yaml` (`cosmos.orchestrator_timeline_jsonl_path`).
  - Confirm timeline file exists on Jetson:
    - `~/pala/logs/orchestrator_timeline.jsonl`
  - `orchestrator_memory.jsonl` is optional/legacy and may be absent in transcript-only planner mode.
- `Video: waiting for frames` with `--video-source tap`:
  - Ensure `pala.main` is running on Jetson.
  - Confirm `telemetry_preview.enabled: true` in `config/robot.yaml`.
  - Confirm tap files exist on Jetson:
    - `~/pala/logs/telemetry/preview/latest.jpg`
    - `~/pala/logs/telemetry/preview/latest.json`
- `video_capture_failed` with `--video-source gst`:
  - Camera is likely already in use by runtime.
  - Use `--video-source tap` for normal runtime+telemetry operation.
- `Command` section shows `no data yet`:
  - Use `--video-source tap` (joint command metadata is attached to tap frames).
  - Wait a few seconds after runtime starts for first preview frame update.
- `video_window unavailable ... init.tcl`:
  - Recreate `.venv` with Homebrew Python and export `UV_PYTHON` as above.

## Run Jetson Agent Directly (Optional)
```bash
uv run python -m tools.telemetry.jetson_agent
```

This prints NDJSON events to stdout, intended for piping to another tool.

With direct agent usage, enable video explicitly:
```bash
uv run python -m tools.telemetry.jetson_agent --video-source tap --video-fps 6

# Direct camera capture mode (single-consumer; may conflict with runtime camera owner)
uv run python -m tools.telemetry.jetson_agent --video-source gst --video-fps 6

# Coalesce repeated warning spam and restart subprocess workers quickly
uv run python -m tools.telemetry.jetson_agent --warning-throttle-s 3 --worker-restart-delay-s 1.0

# Use packs + field filters directly on agent
uv run python -m tools.telemetry.jetson_agent --pack reasoning_live --pack planner_debug --field-filter 'actions_log.data.confidence<0.5'

# Tune trace correlation window for agent-side capture index generation
uv run python -m tools.telemetry.jetson_agent --trace-match-window-s 2.0

# Capture a reproducible bundle on Jetson side
uv run python -m tools.telemetry.jetson_agent --pack all --capture-dir logs/telemetry/session_jetson_001 --capture-frames keyframes
```
