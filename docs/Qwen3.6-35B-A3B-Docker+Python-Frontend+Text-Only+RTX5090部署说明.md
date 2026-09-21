# Qwen3.6-35B-A3B：RTX 5090 单卡部署说明

> 最后核对：2026-09-20 工作记录
> 当前结论：231 服务器已经使用 Docker 成功启动；实测生成速度约 **400 tok/s**。
> 使用组合：NVIDIA NVFP4、Python Frontend、Text Only、MTP@4、单张 RTX 5090。

## 1. 当前有效配置

| 项目 | 当前值 |
| --- | --- |
| 服务器 | 231（RTX 5090 32GB） |
| 模型 | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| 宿主机模型目录 | `/data/metahuman_work/models/nvidia/Qwen3.6-35B-A3B-NVFP4` |
| 容器内模型目录 | `/models/qwen36` |
| Docker 镜像 | `vllm/vllm-openai:v0.28.0` |
| 服务端口 | 宿主机 `8200` → 容器 `8000` |
| API 模型名 | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| 前端 | Python Frontend（`VLLM_USE_RUST_FRONTEND=0`） |
| 模式 | 纯文本（`--language-model-only`） |
| 推测解码 | 内置 MTP，`num_speculative_tokens=4` |
| 实测结果 | 约 **400 tok/s** |

Qwen3.6-35B-A3B 是 MoE 模型，总参数约 35B，每个 token 只激活约 3B 参数。它与 Qwen3.8-27B Dense 模型的计算量不同，因此两者即使都使用 NVFP4，生成速度也不能只按总参数量推断。

## 2. 启动前检查

```bash
nvidia-smi
docker image inspect vllm/vllm-openai:v0.28.0
test -f /data/metahuman_work/models/nvidia/Qwen3.6-35B-A3B-NVFP4/config.json
find /data/metahuman_work/models/nvidia/Qwen3.6-35B-A3B-NVFP4 \
  -maxdepth 1 -type f | sort
ss -ltnp | grep ':8200' || true
```

模型目录必须完整，至少应包含配置、tokenizer、权重分片和权重索引。不要只复制 `.safetensors` 文件。所有大模型统一放在 `/data/metahuman_work/models`，避免 Docker 内再次下载。

## 3. 231 上已验证的启动命令

```bash
docker run --rm -d -it \
  --name qwen36-35b-a3b \
  --gpus '"device=0"' \
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
  --attention-config '{"use_trtllm_attention":true}' \
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
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"moe_backend":"flashinfer_cutlass"}'
```


如果是 Unsloth NVFP4 
```bash
docker run --rm -it \
  --name qwen36-35b-a3b \
  --gpus '"device=0"' \
  --ipc=host \
  -p 8200:8000 \
  -v /data/metahuman_work/models/unsloth/Qwen3.6-35B-A3B-NVFP4:/models/qwen36:ro \
  -e VLLM_USE_RUST_FRONTEND=0 \
  vllm/vllm-openai:v0.28.0 \
  /models/qwen36 \
  --served-model-name unsloth/Qwen3.6-35B-A3B-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --language-model-only \
  --reasoning-parser qwen3
```

关键点：

- `--language-model-only` 不加载视觉部分，把显存留给 KV Cache、MTP 和运行时。
- `--max-model-len 20480` 是当前实测配置，不要因为模型原生支持更长上下文而直接改成 262K。
- `--max-num-seqs 2` 面向低并发场景；修改并发后必须重新测显存和吞吐。
- NVIDIA checkpoint 使用 `modelopt_fp4`、Marlin 和 FlashInfer；这些参数不要直接套到其他发布者的 NVFP4 权重。
- `--rm` 表示容器停止后自动删除；启动命令应保留在文档或脚本中。

## 4. 健康检查与调用

```bash
docker ps --filter name=qwen36-35b-a3b
docker logs -f qwen36-35b-a3b
curl -fsS http://127.0.0.1:8200/health
curl -fsS http://127.0.0.1:8200/v1/models
```

```bash
curl -sS http://127.0.0.1:8200/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "计算 17×23，只给出答案。"}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

Python OpenAI 客户端：

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8200/v1")
response = client.chat.completions.create(
    model="nvidia/Qwen3.6-35B-A3B-NVFP4",
    messages=[{"role": "user", "content": "计算 17×23，只给出答案。"}],
    temperature=0,
    max_tokens=128,
)
print(response.choices[0].message.content)
```

## 5. 运维与排障

```bash
nvidia-smi
docker logs --tail 200 qwen36-35b-a3b
ss -ltnp | grep ':8200'

# 使用了 --rm，停止后容器会被删除
docker stop qwen36-35b-a3b
```

常见问题按以下顺序处理：

1. 启动前先确认 GPU 没被其他进程占满，端口 8200 没被占用。
2. OOM 时先恢复文档中的 20K 上下文、0.85 显存比例和低并发，不要同时改多个参数。
3. 若 MTP 初始化失败，临时移除 `--speculative-config` 验证基础服务；基础服务正常后再定位 MTP/FlashInfer 兼容性。
4. 若换成 Unsloth 等其他 checkpoint，不要继续强制使用 NVIDIA checkpoint 的 `--moe-backend marlin`；另建配置并重新验证。
5. 吞吐测试必须记录提示长度、输出长度、并发、thinking 设置和统计口径。当前“约 400 tok/s”是 2026-09-20 工作记录中的实测结果，不应与不同请求长度的旧数据直接横比。

## 6. 模型迁移

```bash
rsync -avP \
  /data/metahuman_work/models/nvidia/Qwen3.6-35B-A3B-NVFP4/ \
  user@target:/data/metahuman_work/models/nvidia/Qwen3.6-35B-A3B-NVFP4/
```

迁移后至少核对文件数、总大小和权重索引；条件允许时再对大权重文件做 hash 校验。Docker 只挂载本地目录，不需要从 Hugging Face 重复下载。

## 7. 当前结论

- 231 的单张 RTX 5090 可以运行 NVIDIA Qwen3.6-35B-A3B-NVFP4。
- 当前已验证组合是 Docker + Python Frontend + Text Only + FP8 KV + MTP@4。
- 当前记录的约 400 tok/s 明显高于同机 Qwen3.8-27B-NVFP4 的 40+ tok/s；主要背景是 MoE 与 Dense 的每 token 激活计算量不同，但正式模型选型仍需同时比较数学正确率和正确答案端到端耗时。
