#!/usr/bin/env bash
# Start the resumable model copy in the background on server 231.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HOME/autodl-tmp/aigc/dep_qwen38/logs"
LOG_FILE="$LOG_DIR/qwen38_nvfp4_rsync.log"
PID_FILE="$LOG_DIR/qwen38_nvfp4_rsync.pid"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
  echo "Sync is already running (PID $(<"$PID_FILE")). Log: $LOG_FILE"
  exit 0
fi

nohup "$SCRIPT_DIR/sync_qwen38_nvfp4_to_bjb1_892.sh" >>"$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"
echo "Started sync (PID $pid). Log: $LOG_FILE"
