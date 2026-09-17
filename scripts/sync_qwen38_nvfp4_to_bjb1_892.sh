#!/usr/bin/env bash
# Run this script on server 231 (192.168.8.231) as user zenking.
# It is deliberately resumable: re-run it after any interruption.
set -Eeuo pipefail

SOURCE_DIR="/nfs-data/metahuman_work/models/Inferact/Qwen3.8-27B-NVFP4/"
TARGET_HOST="root@connect.bjb1.seetacloud.com"
TARGET_PORT="54492"
TARGET_DIR="/root/autodl-tmp/aigc/dep_qwen38/models/Inferact/Qwen3.8-27B-NVFP4/"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source model directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=20 -p "$TARGET_PORT")

"${SSH[@]}" "$TARGET_HOST" "mkdir -p '$TARGET_DIR'"

# --append-verify preserves and verifies partial large files, rather than
# transferring them from zero after a network/server interruption. No --delete:
# this command must never remove files from the destination model directory.
rsync -aH --partial --append-verify --mkpath \
  --human-readable --info=progress2,stats2 --timeout=600 \
  -e "ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=20 -p $TARGET_PORT" \
  "$SOURCE_DIR" "$TARGET_HOST:$TARGET_DIR"
