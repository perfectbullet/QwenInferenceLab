# Qwen3.8-27B 部署与运维手册

> 合并自 231 选型、模型与镜像清单、vLLM Recipe 中文版、RTX PRO 6000 部署、bjb1-892 重试手册和 2026-09-18 排障记录。
> 当前模型统一存放在 `/data/metahuman_work/models`；231 最新实测约 **40+ tok/s**。

## 1. 平台选择

| 场景 | GPU | 主模型 | 推荐方式 | 状态 |
| --- | --- | --- | --- | --- |
| 231 低并发数学服务 | RTX 5090 32GB | `Inferact/Qwen3.8-27B-NVFP4` | Docker、Text Only、MTP@3、32K | 已验证，约 40+ tok/s |
| 231 历史低延迟实验 | RTX 5090 32GB | 同上 | CUDA Graph、MTP@3、20K、`max-num-seqs=1` | 历史实验约 67.2 tok/s |
| bjb1-892 长上下文 | RTX PRO 6000 96GB | 同上 | Python 环境、262K、MTP@3 | 已成功启动和推理 |

Qwen3.8-27B 是 Dense 模型，内置 MTP draft head。单张 5090 上优先使用 Inferact NVFP4；已有的 Unsloth NVFP4 可保留用于后续 A/B，不作为当前主线。不要使用 MLX 权重；FP8/BF16 主权重也不适合 32GB 单卡的长上下文主方案。

## 2. 模型与镜像

231 的统一目录：

```text
/data/metahuman_work/models/Inferact/Qwen3.8-27B-NVFP4
/data/metahuman_work/models/unsloth/Qwen3.8-27B-NVFP4
```

```bash
nvidia-smi
df -h /data/metahuman_work/models
docker image inspect vllm/vllm-openai:qwen38
find /data/metahuman_work/models/Inferact/Qwen3.8-27B-NVFP4 \
  -maxdepth 1 -type f | sort
```

主模型目录应包含 config、tokenizer、chat template、权重分片和 MTP 权重。容器应挂载本地目录，避免运行时下载。当前主线不需要再下载模型或镜像。

## 3. 231：当前启动方式

### 3.1 默认版本：纯文本 + MTP

这条命令来自 2026-09-20 工作记录，是当前优先复现的版本：

```bash
docker run --gpus all --rm \
  --privileged --ipc=host \
  -p 8200:8000 \
  -v /data/metahuman_work/models:/model \
  vllm/vllm-openai:qwen38 \
  --model /model/Inferact/Qwen3.8-27B-NVFP4 \
  --served-model-name Qwen3.8-27B-NVFP4 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --max-num-seqs 4 \
  --tensor-parallel-size 1 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --language-model-only
```

记录结果：约 **40+ tok/s**。此命令在前台运行且使用 `--rm`；服务化时可以增加固定容器名和 `-d`，但应先保持模型参数不变完成复现。

### 3.2 带工具调用的版本

```bash
docker run --gpus all --name vllm-qwen38 --rm \
  -v /data/metahuman_work/models/Inferact/Qwen3.8-27B-NVFP4:/model:ro \
  -p 8200:8000 \
  vllm/vllm-openai:qwen38 \
  --model /model \
  --served-model-name Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

需要 MTP 时再追加：

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

### 3.3 历史单请求低延迟版本

2026-08-20 曾使用 CUDA Graph 在固定数学题上测得约 67.2 tok/s。关键是删除 `--enforce-eager`、把 capture 限制为单请求，并将窗口设为 20K：

```bash
docker run -d --gpus all --name vllm-qwen38 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /data/metahuman_work/models/Inferact/Qwen3.8-27B-NVFP4:/model:ro \
  -p 8200:8000 \
  vllm/vllm-openai:qwen38 \
  --model /model \
  --served-model-name qwen38-27b \
  --tensor-parallel-size 1 \
  --max-model-len 20480 \
  --max-num-seqs 1 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

此版本贴近 32GB 显存上限，只适合独占 GPU、单请求实验。40+ 与 67.2 tok/s 的请求、日期和配置不同，不能视为性能回退结论；应使用同一 benchmark 复测后再判断。

## 4. 231：验证与运维

```bash
curl -fsS http://127.0.0.1:8200/health
curl -fsS http://127.0.0.1:8200/v1/models
docker logs --tail 200 vllm-qwen38
nvidia-smi
```

```bash
curl -sS http://127.0.0.1:8200/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "Qwen3.8-27B-NVFP4",
    "messages": [{"role": "user", "content": "Calculate 17 times 23. Give only the answer."}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

231 是共享 GPU。MTP 版本启动前应检查可用显存；历史上其他容器占用约 0.86 GiB 就足以导致加载失败。不要为了腾显存擅自停止其他人的服务，先协调资源窗口。

OOM 处理顺序：

1. 核对是否有其他 GPU 进程。
2. 回到 `--enforce-eager` 的稳定基线。
3. 降低 `max-num-seqs`。
4. 将上下文从 32K 降至 20K。
5. 临时关闭 MTP，确认基础模型能启动。

每次只改变一个变量并记录结果。

## 5. RTX PRO 6000 96GB：bjb1-892

### 5.1 环境

| 项目 | 已验证值 |
| --- | --- |
| GPU | RTX PRO 6000 Blackwell Server Edition，SM120 |
| Python 环境 | `/root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm` |
| 模型目录 | `/root/autodl-tmp/aigc/dep_qwen38/models/Inferact/Qwen3.8-27B-NVFP4` |
| vLLM / PyTorch / FlashInfer | 0.29.0 / 2.13.0+cu130 / 0.6.18 |
| API 模型名 | `qwen38-27b-nvfp4` |

创建环境时默认使用清华 PyPI：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -y -p /root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm python=3.12 pip
conda activate /root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  pip install -U pip vllm 'transformers>=5.8.0'
```

### 5.2 已验证启动命令

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm

CUDA_ROOT=/root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm/lib/python3.12/site-packages/nvidia/cu13
unset OMP_NUM_THREADS
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"

vllm serve /root/autodl-tmp/aigc/dep_qwen38/models/Inferact/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen38-27b-nvfp4 \
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
  --port 8000
```

### 5.3 Blackwell 编译排障

已遇到并解决的问题：

- 系统 `/usr/local/cuda` 曾指向 CUDA 11.8，不能编译 SM120 NVFP4 内核。
- CUDA 13.0 的 `ptxas` 不支持所需 PTX 9.4；工具链最终统一为 CUDA 13.4。
- pip CUDA 使用 `lib`，FlashInfer 可能寻找 `lib64`，且链接器需要 `libcudart.so`。
- 空的 `OMP_NUM_THREADS` 会触发 libgomp 警告。

需要重新修复工具链时：

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple pip install --upgrade --force-reinstall \
  'cuda-toolkit==13.4.2' \
  'nvidia-cuda-nvcc==13.4.92' \
  'nvidia-cuda-crt==13.4.92' \
  'nvidia-cuda-runtime==13.4.92'

ln -sfn lib "$CUDA_ROOT/lib64"
ln -sfn libcudart.so.13 "$CUDA_ROOT/lib/libcudart.so"
```

首次启动会编译并调优 FlashInfer 内核。保留 `/root/.cache/flashinfer` 与 `/root/.cache/vllm` 可缩短后续启动时间。升级 PyTorch、vLLM、FlashInfer 或 CUDA 任一组件时，应整体复测，不要默认已有组合仍兼容。

## 6. API 客户端注意事项

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1", timeout=3600)
response = client.chat.completions.create(
    model="qwen38-27b-nvfp4",
    messages=[{"role": "user", "content": "请解释什么是推测解码。"}],
    temperature=1.0,
    top_p=0.95,
    max_tokens=2048,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}},
)
print(response.choices[0].message.content)
```

本项目用过的 vLLM 版本可能把思考文本放在 `message.reasoning`，而不是 `reasoning_content`。看到思考字段为空时，先查看原始响应再判断模型是否未思考。

## 7. 配置变更纪律

- 同一轮只改变一个变量：模型、镜像、上下文、MTP window、eager/graph、并发不能一起改。
- 性能记录必须附带输入/输出 token、TTFT、Decode TPS、总耗时、并发和 thinking 设置。
- 模型选型以 Pass@1 和 Time-to-correct 为主，token/s 只用于解释原因。
- `Qwen3.8-27B-NVFP4` 的 40+ tok/s 是 2026-09-20 当前工作记录；旧实验数据保留为历史，不覆盖当前事实。
