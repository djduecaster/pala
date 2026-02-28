# PALA Probe Web

Web sidecar for inspecting env packets and env→planner chained probes with ordered 4-image uploads.

## Run

```bash
uv run python -m tools.probe_web --host 127.0.0.1 --port 8787
```

Open: `http://127.0.0.1:8787`

## Required env

- `PALA_COSMOS_API_KEY`

Optional overrides in UI:

- provider/model/base URL
- timeout, env max tokens, planner max tokens
- temperature, top_p, presence_penalty (env and planner)
- env contract, policy fields, planner prompt
- inter-frame spacing
- planner max proposals and planner env-context toggle
- planner system prompt, planner image indices, context/user-text overrides
- full planner payload override JSON

Modes:
- `Run Env Probe`: env-only packet/request.
- `Run Env, Preview Planner Inputs`: runs env and prepares full planner packet for inspection.
- `Run Planner From Prepared Env`: executes planner against a prepared env run id.
- `Run Env + Planner`: one-shot chained execution.

Use the `Input/Output Help` overlay in the header for a full contract summary of env and planner inputs/outputs.

## Artifacts

Runs are saved under:

- `logs/probe_web/<run_id>/`

with config, packet view, response outputs, parsed output, and packet images.

For chained runs (`Env + Planner` mode), planner request/response artifacts and effective input snapshots are saved in the same run folder.
