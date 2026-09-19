# PALA — Agent Working Notes (Read First)

## North Star
PALA V1 is a supervised physical AI desk companion running on Jetson.
The default runtime holds position. The opt-in live demo uses stationary camera
snapshots and Gemini structured observations to select deterministic local
performances: notice, greeting, excitement, pointing, settling and idle breathing.
The V1 portfolio demo is operator-reviewed. Preserve this working scope; new
capabilities belong in a deliberate follow-up rather than a broad rewrite.
Model observations never supply joint commands. Servo position is estimated from
commands, not measured feedback.

## Repo workflow
- Mac is source of truth: /Users/djduecaster/development/pala
- Jetson mirror: ~/pala
- Sync + run: `make go`
- Local run: `uv run python -m pala.main`

## Non-negotiables (DO NOT BREAK)
1) Do NOT modify infra scripts without explicit request:
   - deploy_jetson.sh, run_jetson.sh, Makefile, SSH/rsync setup
2) Keep the 4-loop runtime model:
   - Perception loop (~15–30 Hz)
   - Behavior loop (~2–5 Hz)
   - Control loop (~50–100 Hz)
   - Hardware loop (~50–200 Hz, deadman timeout)
3) Data contracts are stable and must be used everywhere:
   - PerceptionState -> ActionPlan -> HardwareCommand
4) Must run on Mac without Jetson hardware using dummy backends by default.
5) Live preview/telemetry streaming is allowed when scoped for debugging or demos.
   - Prefer a sidecar/tooling implementation (for example under `tools/`) rather than core loop wiring.
   - Keep it optional and failure-isolated (core control/safety logic must run without it).
6) Prefer to avoid keeping dead code or fallbacks. It's much easier to keep a simple architecture without a complicated mess of fallbacks and old code.

## Key files
- `pala/main.py` — 4-loop runtime wiring
- `pala/types/models.py` — PerceptionState, ActionPlan, HardwareCommand
- `config/robot.yaml` — calibration + loop rates + limits
- `docs/architecture.md` — loop rates + contracts
- `docs/README.md` — current operator and development guides

## Secrets
- Jetson-only secrets live in: `~/.config/pala/env.sh`
- Never commit secrets. Never add secrets to config/robot.yaml.

## Coding standards
- No side effects at import time
- No globals for shared state; use dataclasses + thread-safe “latest value”
- Use `logging` (not print)
- Type hints + dataclasses
- Make changes small; prefer surgical commits

## V1 maintenance
- Keep media and archived production notes under ignored `logs/local/` or `logs/`.
- Distribute the portfolio film as a GitHub Release asset, not in Git history.
- Preserve the simulator and optional telemetry sidecar.
- Treat historical ratings as evidence for the tested recipe, not all future edits.

## How to work (expected agent behavior)
- Propose a plan before making edits.
- Keep scope tight: 1 capability per change set.
- Show diffs / summarize changes.
- Ensure `uv run python -m pala.main` still runs.
- Keep Mac-only and Jetson-only differences explicit.

## Quick commands
- Run locally: `uv run python -m pala.main`
- Sync/run Jetson: `make go`
- Add dependency: `uv add <pkg>`

## Direct Jetson agent workflow
- Read `docs/jetson_agent_workflow.md` before using the shared Jetson shell.
- Prefer `jetson-wifi` for direct SSH agent work after Wi-Fi is available; the existing `jetson` alias is the USB path used by the current deployment scripts.
- The shared Jetson tmux session is named `pala`. Agents send short commands with `tmux send-keys`, then inspect results with `tmux capture-pane`.
- Keep the shared shell available for operator inspection. Put long-running PALA processes in a named tmux window and check for an existing runtime before starting another one.
- `make go` currently deploys and runs in a foreground SSH command; it does not attach to tmux. A future opt-in `make go-tmux` target is preferred over changing the meaning of `make go`.

## Lessons Learned (2-1-26)
- GStreamer PTS uses a different clock; use PTS deltas or monotonic deltas, not cross-clock subtraction.
- Camera FPS depends heavily on auto-exposure/lighting; use `v4l2-ctl` to inspect modes and note lighting impact.
- Keep perception capture “latest-only” to avoid stale frames when inference stalls.
- Dev-mode dummy sources should emit real frames/packets so CI and local runs are meaningful.
- Jetson system packages (e.g., `python3-gi`) require `uv venv --system-site-packages` or an editable install for imports.

# Repository Guidelines

## Project Structure & Module Organization
- `pala/` contains the core Python package, grouped by subsystem: `perception/`, `behavior/`, `control/`, `hardware/`, `config/`, `types/`, and `utils/`. The old planner package was removed.
- `config/robot.yaml` holds runtime configuration (loop rates, logging, limits).
- `docs/` includes architecture and current operator guides.
- Root scripts (`deploy_jetson.sh`, `run_jetson.sh`, `run_on_jetson.sh`) implement the Mac→Jetson dev loop.
- `main.py` at repo root is a boot check; the main runtime entry is `pala/main.py`.

## Build, Test, and Development Commands
- `make deploy` syncs the repo to the Jetson via `rsync`.
- `make run` runs the Jetson entry script over SSH.
- `make go` performs deploy + run in one step.
- `uv run python -m pala.main` runs locally on Mac using dummy backends.
- `uv sync` installs dependencies from `pyproject.toml`.

## Coding Style & Naming Conventions
- Use 4-space indentation and PEP 8 naming (snake_case functions/vars, CapWords classes).
- Prefer explicit type hints and dataclasses (see `pala/types/models.py`).
- No formatter or linter is configured; avoid stylistic churn.

## Testing Guidelines
- Pytest is configured in `pyproject.toml`; run the full suite with `uv run pytest -q`.
- Add focused `tests/test_*.py` regressions for changed contracts. Hardware tests use fakes; passing tests does not establish physical acceptance.

## Commit & Pull Request Guidelines
- Keep commits short, focused, and descriptive (sentence-case summaries).
- PRs should include: summary, run/test notes (Mac vs Jetson), and any hardware impact. Add logs or screenshots for observable behavior changes.

## Security & Configuration Tips
- Jetson-only secrets may be sourced from `~/.config/pala/env.sh` (see `run_on_jetson.sh`).
- Do not commit secrets or store them in `config/robot.yaml`.
