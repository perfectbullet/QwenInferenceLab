#!/usr/bin/env bash
set -euo pipefail

run_id="${1:-qwen36-ab-$(date +%Y%m%d-%H%M%S)}"
collection="${QWEN36_BENCHMARK_COLLECTION:-qwen36_ab_benchmark_20260921}"
# 默认 32768，符合 Qwen3.6 官方“多数请求”的输出预算建议。
# 若只做限长速度测试，执行前可设 QWEN36_MAX_TOKENS=4096；四组必须相同。
max_tokens="${QWEN36_MAX_TOKENS:-32768}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

cleanup() {
  for variant in A B C D; do "$script_dir/qwen36-231-service.sh" "$variant" stop || true; done
}
trap cleanup EXIT INT TERM

cd "$repo_dir"
for variant in A B C D; do
  echo "===== Variant $variant ====="
  "$script_dir/qwen36-231-service.sh" "$variant" start
  npx tsx scripts/qwen36-benchmark.ts --variant "$variant" --run-id "$run_id" --collection "$collection" --max-tokens "$max_tokens"
  "$script_dir/qwen36-231-service.sh" "$variant" stop
done

trap - EXIT INT TERM
echo "完成: run_id=$run_id collection=$collection"
