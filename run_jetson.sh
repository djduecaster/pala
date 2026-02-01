#!/usr/bin/env bash
set -euo pipefail

ssh jetson "cd ~/pala && ./run_on_jetson.sh"
