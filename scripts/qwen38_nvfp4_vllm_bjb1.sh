#!/usr/bin/env bash
# 在 bjb1 云服务器上启动/查看/停止 Qwen3.8-27B-NVFP4 的 vLLM 服务。
set -Eeuo pipefail

PROJECT_DIR="/root/autodl-tmp/aigc/dep_qwen38"
ENV_DIR="$PROJECT_DIR/envs/qwen38-vllm"
MODEL_DIR="$PROJECT_DIR/models/Inferact/Qwen3.8-27B-NVFP4"
CUDA_ROOT="$ENV_DIR/lib/python3.12/site-packages/nvidia/cu13"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/vllm-qwen38-nvfp4.log"
PID_FILE="$LOG_DIR/vllm-qwen38-nvfp4.pid"
PORT="8000"
MODEL_NAME="qwen38-27b-nvfp4"

usage() {
  cat <<'EOF'
Usage: ./qwen38_nvfp4_vllm_bjb1.sh {start|status|stop|logs}

  start   Start the vLLM server in the background.
  status  Show process, endpoint, and GPU status.
  stop    Stop only the server recorded in this script's PID file.
  logs    Follow the server log.
EOF
}

load_runtime() {
  if [[ ! -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    echo "Conda initialization script was not found." >&2
    exit 1
  fi
  if [[ ! -d "$ENV_DIR" || ! -d "$MODEL_DIR" || ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
    echo "Required environment, model, or CUDA compiler is missing." >&2
    exit 1
  fi

  # pip CUDA packages use lib/, while FlashInfer expects lib64/ and libcudart.so.
  ln -sfn lib "$CUDA_ROOT/lib64"
  ln -sfn libcudart.so.13 "$CUDA_ROOT/lib/libcudart.so"

  # An empty OMP_NUM_THREADS produces a libgomp warning.
  unset OMP_NUM_THREADS
  export CUDA_HOME="$CUDA_ROOT"
  export PATH="$CUDA_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
}

server_is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null
}

start() {
  load_runtime
  mkdir -p "$LOG_DIR"

  if server_is_running; then
    echo "vLLM is already running (PID $(<"$PID_FILE"))."
    echo "Endpoint: http://127.0.0.1:$PORT/v1/models"
    exit 0
  fi

  rm -f "$PID_FILE"
  nohup vllm serve "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --mm-encoder-tp-mode data \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --host 0.0.0.0 \
    --port "$PORT" \
    >"$LOG_FILE" 2>&1 < /dev/null &
  echo $! >"$PID_FILE"

  echo "vLLM start requested (PID $(<"$PID_FILE"))."
  echo "Log: $LOG_FILE"
  echo "The first launch may take several minutes for NVFP4 JIT compilation and warmup."
}

status() {
  if server_is_running; then
    echo "Process: running (PID $(<"$PID_FILE"))"
  else
    echo "Process: not running"
  fi
  echo "Endpoint:"
  curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" || true
  echo
  echo "GPU:"
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
}

stop() {
  if ! server_is_running; then
    echo "No server process recorded by this script."
    rm -f "$PID_FILE"
    exit 0
  fi
  pid="$(<"$PID_FILE")"
  kill "$pid"
  rm -f "$PID_FILE"
  echo "Stopped vLLM process $pid."
}

case "${1:-}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  logs) mkdir -p "$LOG_DIR"; tail -F "$LOG_FILE" ;;
  *) usage; exit 1 ;;
esac
