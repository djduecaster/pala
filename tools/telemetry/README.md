# Telemetry Tools (V3)

Telemetry remains a sidecar under `tools/telemetry` and does not change the core 4-loop runtime.

## V3 Shipment
- Backward-compatible V2 capture files are still written:
  - `events.jsonl`, `index.json`, `reasoning_index.json`, `trace_index.json`, `manifest.json`
- New V3 artifacts are now emitted automatically on capture close:
  - `session.db` (sqlite index for fast querying)
  - `quality_report.json` (capture health score + gates)
  - `improvement_report.json` (actionable bottlenecks + recommendations)
  - `doctor_report.json` (bundle integrity + readiness score + issues)
  - `incident_report.json` + `incident_report.md` (ops-ready incident bundle)
  - `labels.weak.jsonl` (heuristic weak labels for post-training prep)
- Viewer V3 controls:
  - Unified wrapper CLI: `python -m tools.telemetry.telemetry`
  - `--index-mode auto|off|sqlite`
  - `--query '...'` + `--query-limit N`
  - `--query-export <path>` (periodic query snapshot output)
  - `--query-slice-export <path>` (dataset-slice style JSONL export from query hits)
  - `--index-refresh-s` (live recency of local sqlite during capture)
  - `--alert-policy demo|training|debug|custom`
  - `--quality-gate off|warn|strict`
  - `--doctor-gate off|warn|strict`
  - New panels: `alerts`, `throughput`, `quality`, `doctor`, `incident`, `insights`, `story`, `scoreboard`, `query`
  - Alert thresholds: `--alert-stale-s`, `--alert-heartbeat-s`, `--alert-video-idle-s`, `--alert-dropped-events`
  - Session context tags: `--scenario-tag`, `--goal-tag`, `--runbook`, `--golden-session`
- Capture manifest defaults now ship as schema version `3`.

## What V3 Solves
- Faster offline triage with indexed telemetry search.
- Standardized quality scoring to reject low-signal sessions.
- Weak-label generation to bootstrap post-training datasets.
- Replay dashboards that can show quality + query context directly.

## Quickstart
### 0) Unified entrypoint (optional but recommended)
```bash
uv run python -m tools.telemetry.telemetry packs
```

### 1) Run runtime on Jetson
```bash
cd ~/pala
PALA_LOG_LEVEL=INFO uv run python -m pala.main
```

### 2) Run viewer on Mac
```bash
cd /Users/djduecaster/development/pala
UV_PYTHON=/opt/homebrew/bin/python3.10 uv run python -m tools.telemetry.telemetry viewer -- --jetson-host jetson
```

### 3) Capture a V3 session bundle
```bash
uv run python -m tools.telemetry.telemetry viewer -- \
  --pack reasoning_live \
  --save-session logs/telemetry/session_v3_001 \
  --scenario-tag desk_demo \
  --goal-tag post_training \
  --runbook "baseline daytime pass" \
  --capture-frames keyframes
```

### 4) Replay with indexed query + quality gate
```bash
uv run python -m tools.telemetry.telemetry viewer -- \
  --replay logs/telemetry/session_v3_001 \
  --index-mode auto \
  --golden-session logs/telemetry/golden_session_a \
  --query 'status:parse_fail severity:error' \
  --query-slice-export logs/telemetry/query_slice.jsonl \
  --alert-policy training \
  --quality-gate warn
```
`--index-mode auto` now attempts to build `session.db` on replay startup if it is missing.

## Session Bundle Contents (V3)
- `manifest.json`: session metadata, schema version, V3 artifact pointers
- `events.jsonl`: full event stream
- `index.json`: frame index
- `reasoning_index.json`: normalized reasoning rows
- `trace_index.json`: request-level traces
- `frames/`: optional JPEG frames
- `session.db`: sqlite index (`events`, `reasoning`, `traces`, `trace_events`, `meta`)
- `quality_report.json`: numeric score + pass/warn/fail grade + gate checks
- `improvement_report.json`: bottleneck summary + ranked recommendations
- `doctor_report.json`: artifact consistency checks + readiness gate
- `incident_report.json`: concise incident package (severity, issues, top traces/events, actions)
- `incident_report.md`: markdown incident handoff for docs/issues
- `labels.weak.jsonl`: weak labels keyed by `event_index`
- `dataset_rows.jsonl`: optional export output

## Migration / Re-index Existing Sessions
Upgrade any older session directory in-place:
```bash
uv run python -m tools.telemetry.telemetry migrate -- logs/telemetry/session_old
```

Run doctor checks explicitly:
```bash
uv run python -m tools.telemetry.telemetry doctor -- logs/telemetry/session_v3_001 --gate warn
```

Compare candidate vs baseline session:
```bash
uv run python -m tools.telemetry.telemetry compare -- \
  logs/telemetry/baseline_001 \
  logs/telemetry/candidate_001 \
  --output logs/telemetry/compare_candidate_001.json
```

Build incident package:
```bash
uv run python -m tools.telemetry.telemetry incident -- \
  logs/telemetry/session_v3_001 \
  --query 'kind:trace severity:error status:timeout status:parse_fail' \
  --limit 10
```

Scoreboard analytics:
```bash
uv run python -m tools.telemetry.telemetry scoreboard -- \
  --scoreboard-path logs/telemetry/scoreboard.json \
  --top-n 12
```

Multi-session watchdog:
```bash
uv run python -m tools.telemetry.telemetry watchdog -- \
  logs/telemetry/baseline_001 \
  logs/telemetry/candidates_root \
  --discover \
  --fail-on-warn
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
- `kind:<event|reasoning|trace>`
- `since:<duration>` (`ms`, `s`, `m`, `h`; example: `since:15m`)

Unkeyed tokens do substring matching on snippets/payload text. Quoted phrases are supported.

Example:
```bash
--query 'source:timeline_log status:parse_fail req:42 timeout'
--query 'since:10m "camera timeout"'
--query 'kind:trace status:timeout'
```

## Quality Gate Behavior
- `off`: no gating
- `warn`: flag sessions with grade `fail`
- `strict`: viewer exits non-zero if grade is not `pass`

## Throughput and Alerts
- `Throughput` panel shows recent and lifetime event rates, plus per-source rates.
- `Alerts` panel is threshold-driven and can include:
  - stale stream
  - stale heartbeat
  - idle video stream
  - dropped events
  - warning burst
  - low throughput (when `--alert-min-events-per-s > 0`)
  - no traces after grace window
- Optional query snapshot export:
  - `--query-export logs/telemetry/query_snapshot.json`

## Improvement Reports
- Generated automatically for captures and can be regenerated with:
```bash
uv run python -m tools.telemetry.migrate_session logs/telemetry/session_v3_001
```
- `Insights` panel surfaces top recommendations with rationale and next actions.
- Includes failure fingerprints and optional golden-session deltas when `--golden-session` is provided.

## Doctor + Compare
- `doctor` validates session integrity (artifact presence, event JSON health, frame refs, count consistency).
- Doctor emits readiness `score` + `grade` (`pass|warn|fail`) and recommendations.
- Viewer can load/show doctor data (`doctor` panel) and optionally gate via `--doctor-gate warn|strict`.
- `compare` computes baseline vs candidate deltas (`quality`, `doctor`, `parse_fail_rate`, `timeout_rate`) and flags regressions.

## Incident + Watchdog
- `incident` builds a handoff-ready incident bundle from telemetry signals (doctor + improvement + indexed failures).
- `incident_report.md` is suitable for immediate issue filing / debugging handoff.
- `watchdog` runs compare gates across many candidates against one baseline and returns an overall verdict.
- `watchdog --discover` lets you point at a root directory of session runs for rapid batch regression checks.

## Scoreboard Analytics
- Scoreboard now supports leaderboard summaries by `scenario_tag` and `goal_tag`.
- Use `telemetry scoreboard` to quickly see top-performing and risky operating regimes.

## Story + Scoreboard
- `Story` panel: compact timeline narrative for the selected trace plus current query slice.
- `Scoreboard` panel: cross-session trend summary (quality / parse-fail / timeout deltas).
- Scoreboard defaults to `logs/telemetry/scoreboard.json` and auto-updates on capture close/migration.
- Disable scoreboard writes with:
  - viewer capture: `--no-scoreboard-update`
  - migrate: `--no-scoreboard-update`

## Notes
- Keep telemetry optional and failure-isolated. Core runtime must keep operating if telemetry tooling fails.
- For local Mac viewer environments that need Tk windowing, keep `UV_PYTHON` pinned to a Tk-compatible interpreter.
