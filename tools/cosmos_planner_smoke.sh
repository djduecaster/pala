#!/usr/bin/env bash
set -euo pipefail

SECONDS_RUN="25"
MODE="dev"
ACTIONS_PATH="logs/actions.jsonl"
RUN_LOG_PATH=""
BASE_URL="${PALA_COSMOS_BASE_URL:-}"
MODEL="${PALA_COSMOS_MODEL:-nvidia/cosmos-reason2-2b}"

usage() {
  cat <<'EOF'
Usage: tools/cosmos_planner_smoke.sh [options]

Runs a short runtime and verifies remote Cosmos planner output appears.

Options:
  --seconds <sec>         Runtime duration (default: 25)
  --mode <mode>           Runtime mode (default: dev)
  --base-url <url>        Cosmos base URL (e.g. http://<ip>:8000)
  --model <name>          Cosmos model name (default: nvidia/cosmos-reason2-2b)
  --actions <path>        actions jsonl path (default: logs/actions.jsonl)
  --run-log <path>        Run log path (default: /tmp/pala_cosmos_smoke.log)
  -h, --help              Show this help

Required:
  uv, rg
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seconds)
      SECONDS_RUN="${2:?missing value for --seconds}"
      shift 2
      ;;
    --mode)
      MODE="${2:?missing value for --mode}"
      shift 2
      ;;
    --base-url)
      BASE_URL="${2:?missing value for --base-url}"
      shift 2
      ;;
    --model)
      MODEL="${2:?missing value for --model}"
      shift 2
      ;;
    --actions)
      ACTIONS_PATH="${2:?missing value for --actions}"
      shift 2
      ;;
    --run-log)
      RUN_LOG_PATH="${2:?missing value for --run-log}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not found" >&2
  exit 1
fi
if ! command -v rg >/dev/null 2>&1; then
  echo "rg is required but not found" >&2
  exit 1
fi
if [[ -z "${BASE_URL}" ]]; then
  echo "Cosmos base URL missing. Set PALA_COSMOS_BASE_URL or pass --base-url." >&2
  exit 1
fi

mkdir -p "$(dirname "$ACTIONS_PATH")"
touch "$ACTIONS_PATH"

if [[ -z "$RUN_LOG_PATH" ]]; then
  RUN_LOG_PATH="/tmp/pala_cosmos_smoke.log"
fi

start_lines="$(wc -l < "$ACTIONS_PATH" | tr -d ' ')"

export PALA_COSMOS_BASE_URL="$BASE_URL"
export PALA_COSMOS_MODEL="$MODEL"
export PALA_LOG_LEVEL="${PALA_LOG_LEVEL:-INFO}"
export PALA_MAX_RUNTIME_S="$SECONDS_RUN"

echo "Running PALA smoke: mode=${MODE} seconds=${SECONDS_RUN} base_url=${PALA_COSMOS_BASE_URL}"
uv run python -m pala.main --mode "$MODE" 2>&1 | tee "$RUN_LOG_PATH"

tmp_actions="$(mktemp /tmp/pala_cosmos_actions.XXXXXX)"
tail -n "+$((start_lines + 1))" "$ACTIONS_PATH" > "$tmp_actions"
added_lines="$(wc -l < "$tmp_actions" | tr -d ' ')"

remote_count="$(rg -c '"explanation":"cosmos_remote' "$tmp_actions" || true)"
remote_count="${remote_count:-0}"
fallback_count="$(rg -c '"explanation":"idle presence"' "$tmp_actions" || true)"
fallback_count="${fallback_count:-0}"
stats_line="$(rg 'cosmos stats requests=[0-9]+ successes=[0-9]+' "$RUN_LOG_PATH" | tail -n 1 || true)"

requests="-1"
successes="-1"
if [[ -n "$stats_line" ]]; then
  requests="$(echo "$stats_line" | sed -n 's/.*requests=\([0-9][0-9]*\).*/\1/p')"
  successes="$(echo "$stats_line" | sed -n 's/.*successes=\([0-9][0-9]*\).*/\1/p')"
fi

echo "Summary:"
echo "  new_actions=${added_lines}"
echo "  remote_actions=${remote_count}"
echo "  fallback_idle_presence=${fallback_count}"
echo "  cosmos_stats_line=${stats_line:-none}"
echo "  run_log=${RUN_LOG_PATH}"

if [[ "$remote_count" -lt 1 ]]; then
  echo "No cosmos_remote actions detected in $ACTIONS_PATH" >&2
  echo "Check run log for parse warnings or connectivity issues: $RUN_LOG_PATH" >&2
  rm -f "$tmp_actions"
  exit 1
fi

if [[ "$successes" != "-1" && "$successes" -lt 1 ]]; then
  echo "Cosmos stats showed no successful remote responses." >&2
  rm -f "$tmp_actions"
  exit 1
fi

echo "PASS: remote Cosmos planner responses were observed."
rm -f "$tmp_actions"
