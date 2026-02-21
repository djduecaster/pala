# Telemetry Tools (V3)

Telemetry remains a sidecar under `tools/telemetry` and does not change the core 4-loop runtime.

## V3 Shipment
- Backward-compatible V2 capture files are still written:
  - `events.jsonl`, `index.json`, `reasoning_index.json`, `trace_index.json`, `manifest.json`
- New V3 artifacts are now emitted automatically on capture close:
  - `session.db` (sqlite index for fast querying)
  - `quality_report.json` (capture health score + gates)
  - `labels.weak.jsonl` (heuristic weak labels for post-training prep)
- Viewer V3 controls:
  - `--index-mode auto|off|sqlite`
  - `--query '...'` + `--query-limit N`
  - `--quality-gate off|warn|strict`
  - New panels: `quality`, `query`
- Capture manifest defaults now ship as schema version `3`.

## What V3 Solves
- Faster offline triage with indexed telemetry search.
- Standardized quality scoring to reject low-signal sessions.
- Weak-label generation to bootstrap post-training datasets.
- Replay dashboards that can show quality + query context directly.

## Quickstart
### 1) Run runtime on Jetson
```bash
cd ~/pala
PALA_LOG_LEVEL=INFO uv run python -m pala.main
```

### 2) Run viewer on Mac
```bash
cd /Users/djduecaster/development/pala
UV_PYTHON=/opt/homebrew/bin/python3.10 uv run python -m tools.telemetry.mac_viewer --jetson-host jetson
```

### 3) Capture a V3 session bundle
```bash
uv run python -m tools.telemetry.mac_viewer \
  --pack reasoning_live \
  --save-session logs/telemetry/session_v3_001 \
  --capture-frames keyframes
```

### 4) Replay with indexed query + quality gate
```bash
uv run python -m tools.telemetry.mac_viewer \
  --replay logs/telemetry/session_v3_001 \
  --index-mode auto \
  --query 'status:parse_fail severity:error' \
  --quality-gate warn
```

## Session Bundle Contents (V3)
- `manifest.json`: session metadata, schema version, V3 artifact pointers
- `events.jsonl`: full event stream
- `index.json`: frame index
- `reasoning_index.json`: normalized reasoning rows
- `trace_index.json`: request-level traces
- `frames/`: optional JPEG frames
- `session.db`: sqlite index (`events`, `reasoning`, `traces`, `trace_events`, `meta`)
- `quality_report.json`: numeric score + pass/warn/fail grade + gate checks
- `labels.weak.jsonl`: weak labels keyed by `event_index`
- `dataset_rows.jsonl`: optional export output

## Migration / Re-index Existing Sessions
Upgrade any older session directory in-place:
```bash
uv run python -m tools.telemetry.migrate_session logs/telemetry/session_old
```

Also export dataset rows:
```bash
uv run python -m tools.telemetry.migrate_session \
  logs/telemetry/session_old \
  --export-dataset
```

## Query Syntax (`--query`)
Space-delimited tokens. Supported key filters:
- `source:<name>`
- `severity:<info|warning|error>`
- `status:<status>`
- `phase:<phase>`
- `req:<id>`
- `trace:<trace-id-fragment>`

Unkeyed tokens do substring matching on snippets/payload text.

Example:
```bash
--query 'source:timeline_log status:parse_fail req:42 timeout'
```

## Quality Gate Behavior
- `off`: no gating
- `warn`: flag sessions with grade `fail`
- `strict`: viewer exits non-zero if grade is not `pass`

## Notes
- Keep telemetry optional and failure-isolated. Core runtime must keep operating if telemetry tooling fails.
- For local Mac viewer environments that need Tk windowing, keep `UV_PYTHON` pinned to a Tk-compatible interpreter.
