#!/usr/bin/env bash
set -euo pipefail

variant="${1:-}"
action="${2:-start}"
host="${QWEN36_SSH_HOST:-zenking@192.168.8.231}"
container="qwen36-ab-${variant,,}"

# A: no MTP, no eager; B: no MTP + eager;
# C: MTP@4 + eager; D: MTP@4, no eager.
# 除 --speculative-config 和 --enforce-eager 外，四组服务端参数完全一致。

if [[ ! "$variant" =~ ^[ABCD]$ ]]; then
  echo "用法: $0 A|B|C|D [start|stop|status|logs]" >&2
  exit 2
fi

case "$action" in
  stop)
    ssh "$host" "docker stop '$container' >/dev/null 2>&1 || true"
    exit 0
    ;;
  status)
    ssh "$host" "docker ps -a --filter name='$container'; nvidia-smi"
    exit 0
    ;;
  logs)
    ssh "$host" "docker logs --tail 200 '$container'"
    exit 0
    ;;
  start) ;;
  *) echo "未知操作: $action" >&2; exit 2 ;;
esac

speculative=""
eager=""
case "$variant" in
  A) ;;
  B) eager="--enforce-eager" ;;
  C) eager="--enforce-eager"; speculative="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4,\"moe_backend\":\"flashinfer_cutlass\"}'" ;;
  D) speculative="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4,\"moe_backend\":\"flashinfer_cutlass\"}'" ;;
esac

ssh "$host" "docker stop qwen36-ab-a qwen36-ab-b qwen36-ab-c qwen36-ab-d >/dev/null 2>&1 || true
docker run --rm -d \
  --name '$container' \
  --gpus '\"device=0\"' \
  --ipc=host \
  -p 8200:8000 \
  -v /data/metahuman_work/models/nvidia/Qwen3.6-35B-A3B-NVFP4:/models/qwen36:ro \
  -e VLLM_USE_RUST_FRONTEND=0 \
  -e VLLM_HAS_FLASHINFER_CUBIN=1 \
  vllm/vllm-openai:v0.28.0 \
  /models/qwen36 \
  --served-model-name nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --moe-backend marlin \
  --attention-backend flashinfer \
  --attention-config '{\"use_trtllm_attention\":true}' \
  --gpu-memory-utilization 0.85 \
  --max-model-len 20480 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --skip-mm-profiling \
  --language-model-only \
  --reasoning-parser qwen3 \
  $eager $speculative"

echo "已启动 $container；等待接口就绪……"
for attempt in $(seq 1 120); do
  if ssh "$host" "curl -fsS http://127.0.0.1:8200/health >/dev/null"; then
    ssh "$host" "docker inspect --format '{{json .Config.Cmd}}' '$container'"
    exit 0
  fi
  if ! ssh "$host" "docker ps --format '{{.Names}}' | grep -Fx '$container' >/dev/null"; then
    ssh "$host" "docker logs --tail 200 '$container' 2>&1 || true"
    exit 1
  fi
  sleep 5
done

echo "服务在 10 分钟内未就绪" >&2
ssh "$host" "docker logs --tail 200 '$container' 2>&1 || true"
exit 1
