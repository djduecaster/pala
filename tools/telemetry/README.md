# Telemetry Tools (V3.x)

Telemetry remains a sidecar under `tools/telemetry` and does not change the core 4-loop runtime.

## V3 Shipment
- Backward-compatible V2 capture files are still written:
  - `events.jsonl`, `index.json`, `reasoning_index.json`, `trace_index.json`, `manifest.json`
- New V3 artifacts are now emitted automatically on capture close:
  - `session.db` (sqlite index for fast querying)
  - `quality_report.json` (capture health score + gates)
  - `reasoning_trace_index.json` (joined reasoning traces with perception/video context)
  - `labels.weak.jsonl` (heuristic weak labels for post-training prep)
- Viewer V3 controls:
  - `--index-mode auto|off|sqlite`
  - `--query '...'` + `--query-limit N`
  - `--quality-gate off|warn|strict`
  - New panels: `quality`, `query`, `alignment`, `integrity`, `annotations`
- Capture manifest defaults now ship as schema version `3`.
- New V3.x robustness:
  - `integrity.json` artifact checksums + replay verification
  - Optional live indexing (`--index-live-every N`) for long runs
- Preset-driven viewer UX (`--preset ...`) with compact primary CLI
- Bookmark annotations (`annotations.jsonl`) for post-training curation
- Dataset export profiles (`fast|strict|hard_cases`)
- One-click curation export on viewer exit (`--curate-on-exit`)
- Curation guardrails:
  - In live mode, `--curate-on-exit` auto-creates `--save-session` if omitted.
  - Curation export fails fast if zero rows are produced.
  - `dataset_manifest.json` now includes coverage ratios (annotation/label/hard-case) and inclusion reason counts.

## What V3 Solves
- Faster offline triage with indexed telemetry search.
- Standardized quality scoring to reject low-signal sessions.
- Weak-label generation to bootstrap post-training datasets.
- Replay dashboards that can show quality + query context directly.

## Quickstart
### 0) Run telemetry doctor (recommended)
```bash
uv run python -m tools.telemetry.doctor --jetson-host jetson
```

### 1) Run runtime on Jetson
```bash
cd ~/pala
PALA_LOG_LEVEL=INFO uv run python -m pala.main
```

### 2) Run viewer on Mac
```bash
cd /Users/djduecaster/development/pala
UV_PYTHON=/opt/homebrew/bin/python3.10 uv run python -m tools.telemetry.mac_viewer \
  --preset baseline \
  --jetson-host jetson
```

### 3) Capture a V3 session bundle
```bash
uv run python -m tools.telemetry.mac_viewer \
  --preset baseline \
  --save-session logs/telemetry/session_v3_001 \
  --capture-frames keyframes
```

### 4) Replay with indexed query + quality gate
```bash
uv run python -m tools.telemetry.mac_viewer \
  --replay logs/telemetry/session_v3_001 \
  --preset posttrain-curation \
  --query 'status:parse_fail severity:error' \
  --quality-gate strict
```

### 5) One-click post-training curation export
```bash
uv run python -m tools.telemetry.mac_viewer \
  --replay logs/telemetry/session_v3_001 \
  --curate-on-exit \
  --curate-profile hard_cases
```

## Session Bundle Contents (V3)
- `manifest.json`: session metadata, schema version, V3 artifact pointers
- `events.jsonl`: full event stream
- `index.json`: frame index
- `reasoning_index.json`: normalized reasoning rows
- `trace_index.json`: request-level traces
- `reasoning_trace_index.json`: canonical joined rows linking env/planner reasoning to perception + video context
- `frames/`: optional JPEG frames
- `session.db`: sqlite index (`events`, `reasoning`, `traces`, `trace_events`, `reasoning_traces`, `meta`)
- `quality_report.json`: numeric score + pass/warn/fail grade + gate checks
- `integrity.json`: artifact checksum report used by replay/migrate checks
- `annotations.jsonl`: user bookmarks from viewer hotkey (`b`)
- `labels.weak.jsonl`: weak labels keyed by `event_index`
- `dataset_rows.jsonl`: optional export output
- `dataset_manifest.json`: profile/run summary for dataset export
  - includes `inclusion_reason_counts` (`hard_case`, `annotation`, `weak_label`, `baseline`)

## Migration / Re-index Existing Sessions
Upgrade any older session directory in-place:
```bash
uv run python -m tools.telemetry.migrate_session logs/telemetry/session_old
```

Also export dataset rows:
```bash
uv run python -m tools.telemetry.migrate_session \
  logs/telemetry/session_old \
  --export-dataset \
  --dataset-profile hard_cases
```

## Query Syntax (`--query`)
Space-delimited tokens. Supported key filters:
- `source:<name>`
- `severity:<info|warning|error>`
- `status:<status>`
- `phase:<phase>`
- `req:<id>`
- `trace:<trace-id-fragment>`
- `component:<env_processor|planner|...>` (for joined reasoning traces)
- `kind:<event|reasoning|trace|joined>`
- `latency_ms>2000` / `latency_ms<500`
- `duration_ms>1000` (trace duration)
- `ts:[start,end]` (wall-clock timestamp range)
- `sort:latency|severity|ts`
- `order:asc|desc`

Unkeyed tokens do substring matching on snippets/payload text. Use `|` for OR inside a key value, e.g. `status:parse_fail|timeout`.

Example:
```bash
--query 'source:timeline_log status:parse_fail req:42 timeout'
--query 'kind:joined component:env_processor req:42'
--query 'kind:event latency_ms>2000 sort:latency order:desc'
--query 'kind:trace duration_ms>1000 severity:error'
```

## BehaviorV2 Sources (Jetson Agent)
Agent can tail BehaviorV2 logs directly:
- `--behavior-env-log` (default `logs/behavior_env.jsonl`)
- `--behavior-planner-log` (default `logs/behavior_planner.jsonl`)
- `--behavior-reasoning-log` (default `logs/behavior_reasoning.jsonl`)

Use pack:
```bash
--pack behavior_v2_debug
```

## Presets + Compact CLI
Viewer now defaults to a compact, preset-driven CLI surface (roughly <=20 primary flags).

List presets:
```bash
uv run python -m tools.telemetry.mac_viewer --list-presets
```

Common presets:
- `baseline`
- `headless-debug`
- `posttrain-curation`
- `demo`

`baseline` is now intentionally lean: summary + trace list + reasoning stream + alignment + quality + video.

Preset file override:
```bash
uv run python -m tools.telemetry.mac_viewer --preset-file tools/telemetry/presets.yaml --preset baseline
```

## Bookmarks / Annotations
- Press `b` in viewer to bookmark the currently selected trace/reasoning context.
- Writes to `annotations.jsonl` in the active replay/capture session directory.
- Annotation panel can be enabled with `--panel annotations`.
- `--curate-on-exit` exports `dataset_rows.jsonl` + `dataset_manifest.json` using the selected curation profile.

## Quality Gate Behavior
- `off`: no gating
- `warn`: flag sessions with grade `fail`
- `strict`: viewer exits non-zero if grade is not `pass`

## Notes
- Keep telemetry optional and failure-isolated. Core runtime must keep operating if telemetry tooling fails.
- For local Mac viewer environments that need Tk windowing, keep `UV_PYTHON` pinned to a Tk-compatible interpreter.
