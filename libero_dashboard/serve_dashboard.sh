#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
REFRESH_SECONDS="${REFRESH_SECONDS:-30}"

cd "$ROOT"

refresh_loop() {
  while true; do
    "$PYTHON_BIN" libero_dashboard/build_dashboard_data.py >/tmp/libero_dashboard_refresh.log 2>&1 || true
    sleep "$REFRESH_SECONDS"
  done
}

refresh_loop &
REFRESH_PID=$!

cleanup() {
  kill "$REFRESH_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

"$PYTHON_BIN" -m http.server "$PORT" --bind "$HOST"
