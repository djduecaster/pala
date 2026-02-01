# PALA — Agent Working Notes (Read First)

## North Star
PALA is a physical AI desk companion (lamp) running on Jetson. The system uses:
- Fast local perception + deterministic control
- Slow high-level behavior decisions (using Cosmos Reason 2 for NVIDIA challenge)
Goal: a reproducible, contest-ready demo with clean architecture and clear evaluation.

## Repo workflow
- Mac is source of truth: ~/dev/pala
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
5) No MJPEG/preview server for now (file-based logs only).

## Key files
- `pala/main.py` — 4-loop runtime wiring
- `pala/types/models.py` — PerceptionState, ActionPlan, HardwareCommand
- `config/robot.yaml` — calibration + loop rates + limits
- `docs/architecture.md` — loop rates + contracts
- `docs/porting_plan.md` — legacy mapping from `../pala_old/...`

## Secrets
- Jetson-only secrets live in: `~/.config/pala/env.sh`
- Never commit secrets. Never add secrets to config/robot.yaml.

## Coding standards
- No side effects at import time
- No globals for shared state; use dataclasses + thread-safe “latest value”
- Use `logging` (not print)
- Type hints + dataclasses
- Make changes small; prefer surgical commits

## Migration policy (from ../pala_old/pala_project/src)
Migrate capabilities, not folders. Suggested order:
1) Servo backend: PCA9685 + calibration mapping
2) Control primitives: MoveTo, Hold, Glance/Breath; enforce limits
3) Camera pipeline: GStreamer low-latency capture
4) Perception: preprocess + TRT detector + pose + pointing head
5) Planner: Cosmos interface + post-training dataset exporter + eval harness

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

# Repository Guidelines

## Project Structure & Module Organization
- `pala/` contains the core Python package, grouped by subsystem: `perception/`, `planner/`, `behavior/`, `control/`, `hardware/`, `config/`, `types/`, and `utils/`.
- `config/robot.yaml` holds runtime configuration (loop rates, logging, limits).
- `docs/` includes architecture notes and porting plans.
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
- No test framework is configured yet.
- If adding tests, create `tests/` with `test_*.py` and document the runner (e.g., `python -m pytest`).

## Commit & Pull Request Guidelines
- Keep commits short, focused, and descriptive (sentence-case summaries).
- PRs should include: summary, run/test notes (Mac vs Jetson), and any hardware impact. Add logs or screenshots for observable behavior changes.

## Security & Configuration Tips
- Jetson-only secrets may be sourced from `~/.config/pala/env.sh` (see `run_on_jetson.sh`).
- Do not commit secrets or store them in `config/robot.yaml`.
