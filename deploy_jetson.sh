#!/usr/bin/env bash
set -euo pipefail

JETSON_HOST="jetson"
JETSON_DIR="~/pala"

# Sync current folder (repo root) to Jetson
rsync -az --delete \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".DS_Store" \
  --exclude "node_modules/" \
  --exclude "build/" \
  --exclude "dist/" \
  ./ "${JETSON_HOST}:${JETSON_DIR}/"

echo "✅ Synced to ${JETSON_HOST}:${JETSON_DIR}"
