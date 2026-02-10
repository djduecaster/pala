#!/usr/bin/env bash
set -euo pipefail

IMAGE="nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0"
NAME="cosmos"
PORT="8000"
CACHE_DIR="${HOME}/.cache/nim"
REPLACE="0"

usage() {
  cat <<'EOF'
Usage: tools/brev_bootstrap_cosmos.sh [options]

Options:
  --image <image>         NIM image (default: nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0)
  --name <name>           Container name (default: cosmos)
  --port <port>           Host port for NIM (default: 8000)
  --cache-dir <path>      Cache mount path (default: ~/.cache/nim)
  --replace               Remove existing container with same name
  -h, --help              Show this help

Required env:
  NGC_API_KEY             NGC API key for pulling/running NIM images
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="${2:?missing value for --image}"
      shift 2
      ;;
    --name)
      NAME="${2:?missing value for --name}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="${2:?missing value for --cache-dir}"
      shift 2
      ;;
    --replace)
      REPLACE="1"
      shift
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

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not found" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required but not found" >&2
  exit 1
fi

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY is not set in this shell." >&2
  echo "Example: export NGC_API_KEY='...'" >&2
  exit 1
fi

echo "Checking GPU..."
nvidia-smi >/dev/null

echo "Logging into nvcr.io with NGC_API_KEY..."
printf '%s' "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null

mkdir -p "$CACHE_DIR"

if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
  if [[ "$REPLACE" == "1" ]]; then
    echo "Removing existing container: $NAME"
    docker rm -f "$NAME" >/dev/null
  else
    echo "Container '$NAME' already exists. Use --replace to recreate it." >&2
    echo "Existing status:"
    docker ps -a --filter "name=^/${NAME}$"
    exit 1
  fi
fi

echo "Starting container '$NAME' from image '$IMAGE'..."
docker run -d --name "$NAME" \
  --restart unless-stopped \
  --gpus all \
  -p "${PORT}:8000" \
  -e NGC_API_KEY \
  -v "${CACHE_DIR}:/opt/nim/.cache" \
  "$IMAGE" >/dev/null

echo "Container started."
echo
echo "Next commands:"
echo "  docker ps --filter name=^/${NAME}$"
echo "  docker logs -f ${NAME}"
echo "  curl -sS http://127.0.0.1:${PORT}/v1/health/ready"
