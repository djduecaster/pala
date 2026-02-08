# Telemetry Tools (Phase 1)

This directory contains sidecar telemetry tooling for live debugging and demos.
It is intentionally separate from core runtime control/safety logic.

## What Phase 1 Provides
- Mac-run live text dashboard plus optional live video window with overlays.
- Jetson streamed telemetry via SSH sidecar agent.
- Inputs:
  - `logs/perception.jsonl`
  - `logs/actions.jsonl`
  - `tegrastats`
  - filtered `journalctl` lines
  - optional live frames from runtime preview tap (`logs/telemetry/preview/latest.jpg` + metadata)

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

`jetson_dir` path notes:
- Default is `~/pala` on the Jetson.
- If you pass a local Mac absolute path like `/Users/.../pala`, the viewer remaps it to `~/pala` automatically.
- If you pass `~/pala` manually from your shell, quote it (`'~/pala'`) to avoid local expansion.

Useful flags:
```bash
# Lower update rate if terminal redraw is heavy
uv run python -m tools.telemetry.mac_viewer --refresh-hz 2

# Disable journal tailing
uv run python -m tools.telemetry.mac_viewer --no-journal

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

# The window is also mouse-resizable (drag edges/corners)
```

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
- `Video: waiting for frames` with `--video-source tap`:
  - Ensure `pala.main` is running on Jetson.
  - Confirm `telemetry_preview.enabled: true` in `config/robot.yaml`.
  - Confirm tap files exist on Jetson:
    - `~/pala/logs/telemetry/preview/latest.jpg`
    - `~/pala/logs/telemetry/preview/latest.json`
- `video_capture_failed` with `--video-source gst`:
  - Camera is likely already in use by runtime.
  - Use `--video-source tap` for normal runtime+telemetry operation.
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
```
