# PALA Probe Web (Behavior V4)

Web sidecar for inspecting single-call Behavior V4 model packets and testing the deterministic mode FSM simulator.

## Run

```bash
uv run python -m tools.probe_web --host 127.0.0.1 --port 8787
```

Open: `http://127.0.0.1:8787`

## Required env

- `PALA_COSMOS_API_KEY`

## Main capabilities

- Ordered 4-image upload packet with preview and reordering.
- Full Behavior V4 packet inspection:
  - model target/knobs
  - policy fields
  - auto/effective context
  - response format schema
  - full redacted payload
- Result diagnostics:
  - raw content
  - parsed decision
  - parser stage/error
  - guard result and committed action
- Deterministic mode FSM simulator panel:
  - step with signals
  - force mode transitions
  - reset and inspect snapshots
- Scenario presets panel:
  - one-click FSM+packet setup for social greeting, search assist, and recover-reset tuning
- Run compare panel:
  - compare two prior runs for parse outcomes, guard reasons, final primitives, latency, and token deltas
- Base UX improvements:
  - sticky run action bar + run-state status
  - JSON validate/format helpers for override fields
  - local form persistence between refreshes

## Artifacts

Runs are saved under:

- `logs/probe_web_v4/<run_id>/`

with config, packet views, response outputs, parsed output, guard/final action files, FSM snapshots, and packet images.
