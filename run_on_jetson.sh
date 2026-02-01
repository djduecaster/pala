#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Ensure uv is on PATH (common install location)
export PATH="$HOME/.local/bin:$PATH"

# Optional: load Jetson-only secrets/config (not in repo)
if [ -f "$HOME/.config/pala/env.sh" ]; then
  source "$HOME/.config/pala/env.sh"
fi

# One-time env bootstrap (safe to run every time)
if [ ! -d ".venv" ]; then
  echo "Creating venv on Jetson..."
  uv venv
fi

echo "Syncing deps on Jetson..."
uv sync

echo "Running PALA..."
uv run python main.py
