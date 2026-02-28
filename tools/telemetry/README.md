# Telemetry Tools (V4 CaseOps)

Telemetry remains a sidecar under `tools/telemetry` and does not change the core 4-loop runtime.

## V4 Shipment
- Case-centric telemetry workflow:
  - `capture` -> `compile` -> `review` -> `export` -> `report`
  - unified entrypoint: `tools.telemetry.pipeline`
- Core artifacts emitted on capture close:
  - `session.db` (sqlite index for fast querying)
  - `quality_report.json` (capture health score + gates)
  - `reasoning_trace_index.json` (joined reasoning traces with perception/video context)
  - `labels.weak.jsonl` (heuristic weak labels for post-training prep)
- New V4 sqlite case tables:
  - `cases`, `case_events`, `case_labels`, `case_reviews`
  - canonical case source tag: `sqlite.cases.v4`
- Viewer V4 controls:
  - `--index-mode auto|off|sqlite`
  - `--query '...'` + `--query-limit N`
  - `--quality-gate off|warn|strict`
  - Core panels: `case_list`, `case_detail`, `quality`, `query`, `annotations`
- Stream-health diagnostics in viewer summary panel:
  - RX rate (current + peak), agent transport queue pressure, local viewer queue pressure
  - reconnect counters (`total`, `stale`, `disconnect`, `start_fail`)
- Capture manifest defaults now ship as schema version `3`.
- New V3.x robustness:
  - `integrity.json` artifact checksums + replay verification
  - Optional live indexing (`--index-live-every N`) for long runs
- Mode-driven viewer UX (`--mode live|replay|curate`) with compact primary CLI
- Bookmark annotations (`annotations.jsonl`) for post-training curation
- Dataset export profiles (`fast|strict|hard_cases`)
- One-click curation export on viewer exit (`--curate-on-exit`)
- Curation guardrails:
  - In live mode, `--curate-on-exit` auto-creates `--save-session` if omitted.
  - Curation export fails fast if zero rows are produced.
  - `dataset_manifest.json` now includes coverage ratios (annotation/label/hard-case) and inclusion reason counts.

## What V4 Solves
- Faster offline triage with indexed telemetry search.
- Standardized quality scoring to reject low-signal sessions.
- Weak-label generation to bootstrap post-training datasets.
- Replay dashboards that can show quality + query context directly.
- Case Explorer: deterministic, sqlite-backed case triage with review decisions.

## Quickstart
### 0) Run telemetry doctor (recommended)
```bash
uv run python -m tools.telemetry.doctor --jetson-host jetson
```
When `--session-dir` is provided, doctor now validates viewer run artifacts and surfaces run-health alerts.

### 1) Run runtime on Jetson
```bash
cd ~/pala
PALA_LOG_LEVEL=INFO uv run python -m pala.main
```

### 2) Run viewer on Mac
```bash
cd /Users/djduecaster/development/pala
UV_PYTHON=/opt/homebrew/bin/python3.10 uv run python -m tools.telemetry.mac_viewer \
  --mode live \
  --jetson-host jetson
```

### 3) Capture + compile with pipeline CLI
```bash
uv run python -m tools.telemetry.pipeline capture \
  --save-session logs/telemetry/session_v4_001 \
  --jetson-host jetson
uv run python -m tools.telemetry.pipeline compile logs/telemetry/session_v4_001
```

### 4) Review cases
```bash
uv run python -m tools.telemetry.pipeline review \
  logs/telemetry/session_v4_001 \
  --quality-gate strict
```

### 5) Export post-training rows
```bash
uv run python -m tools.telemetry.pipeline export \
  logs/telemetry/session_v4_001 \
  --profile hard_cases
```

### 6) Summarize run history across sessions
```bash
uv run python -m tools.telemetry.pipeline report --root logs/telemetry
```
Report output now includes session coverage (`sessions_with_runs` vs `sessions_without_runs`) to catch missing run artifacts early.
It also aggregates stream-health indicators (queue pressure peaks, reconnect churn, RX throughput peaks, and event-volume percentiles).
For CI-style gating, fail when the latest run looks unhealthy:
```bash
uv run python -m tools.telemetry.run_report --root logs/telemetry --strict
```
Pipeline wrapper also supports strict gating:
```bash
uv run python -m tools.telemetry.pipeline report --root logs/telemetry --strict
```
`pipeline report` scans only telemetry-like session directories (marker artifacts present) and prints a compact alert preview when issues are detected.

## Session Bundle Contents (V4)
- `manifest.json`: session metadata, schema version, V3 artifact pointers
- `events.jsonl`: full event stream
- `index.json`: frame index
- `reasoning_index.json`: normalized reasoning rows
- `trace_index.json`: request-level traces
- `reasoning_trace_index.json`: canonical joined rows linking env/planner reasoning to perception + video context
- `frames/`: optional JPEG frames
- `session.db`: sqlite index (`events`, `reasoning`, `traces`, `trace_events`, `reasoning_traces`, `cases`, `case_events`, `case_labels`, `case_reviews`, `meta`)
- `quality_report.json`: numeric score + pass/warn/fail grade + gate checks
- `integrity.json`: artifact checksum report used by replay/migrate checks
- `annotations.jsonl`: user bookmarks from viewer hotkey (`b`)
- `labels.weak.jsonl`: weak labels keyed by `event_index`
- `dataset_rows.jsonl`: optional export output
- `dataset_manifest.json`: profile/run summary for dataset export
  - includes `inclusion_reason_counts` (`hard_case`, `annotation`, `weak_label`, `baseline`)
- `viewer_summary.json`: viewer exit summary (`run_id`, mode, query, drops, event counts, quality gate, curation result)
  - includes stream-health peaks (`transport_queue_peak_utilization`, `local_queue_peak_utilization`, `rx_rate_peak_5s`) and reconnect counters
  - includes `event_count_total` for low-activity detection on long live runs
- `viewer_runs.jsonl`: append-only history of viewer runs for the session bundle
  - includes case explorer diagnostics (`case_source`, `case_rows_total`, `case_rows_visible`, `case_reviewed_visible`, `case_unavailable_reason`)

## Migration / Re-index Existing Sessions
Upgrade any older session directory in-place:
```bash
uv run python -m tools.telemetry.migrate_session logs/telemetry/session_old
```
Case Explorer requires compiled sqlite artifacts: run `pipeline compile` (or `migrate_session`) before case triage.

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
Agent tails BehaviorV2 logs directly:
- `--behavior-env-log` (default `logs/behavior_env.jsonl`)
- `--behavior-planner-log` (default `logs/behavior_planner.jsonl`)
- `--behavior-reasoning-log` (default `logs/behavior_reasoning.jsonl`)
- `--behavior-trace-log` (default `logs/behavior_trace.jsonl`)

## Modes + Compact CLI
Viewer now defaults to a compact mode-driven CLI surface.

Common mode workflow:
- `--mode live`: stream from Jetson
- `--mode replay`: inspect a saved session
- `--mode curate`: replay + export dataset rows on exit

## Bookmarks / Annotations
- Press `b` in viewer to bookmark the currently selected trace/reasoning context.
- Writes to `annotations.jsonl` in the active replay/capture session directory.
- `--curate-on-exit` exports `dataset_rows.jsonl` + `dataset_manifest.json` using the selected curation profile.

## Case Explorer
- Enabled by default in viewer modes.
- Canonical source is `sqlite.cases.v4` from `session.db`.
- No memory fallback: if `session.db` is unavailable, viewer shows compile guidance.
- Query panel is also sqlite-only; run `pipeline compile` before indexed search.
- Use existing hotkeys:
  - `j/k`: move reasoning selection, or case selection when case panel is focused.
  - `o`: jump to case detail panel.
  - `a/x/n/m`: case review decisions (`accept|reject|needs_context|label`).

## Quality Gate Behavior
- `off`: no gating
- `warn`: flag sessions with grade `fail`
- `strict`: viewer exits non-zero if grade is not `pass`

## Notes
- Keep telemetry optional and failure-isolated. Core runtime must keep operating if telemetry tooling fails.
- For local Mac viewer environments that need Tk windowing, keep `UV_PYTHON` pinned to a Tk-compatible interpreter.
