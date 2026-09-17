# Qwen3.8-27B：RTX PRO 6000 vLLM 中文部署说明

来源：[vLLM Qwen3.8-27B recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)

## 模型与目标

Qwen3.8-27B 是一个 270 亿参数的多模态模型，原生支持 262,144 token 上下文，内置 MTP（多 Token 预测）草稿头，可用于推测解码。该配方验证的是以文本推理为主的服务。

本部署的目标组合：

- GPU：单张 RTX PRO 6000 96GB（Blackwell）
- 并行：TP=1
- 权重：`Inferact/Qwen3.8-27B-NVFP4`
- 精度：NVFP4（W4A4）
- KV Cache：FP8
- 功能：Tool Calling、Reasoning、MTP 推测解码、编码器并行模式
- 前端：Python vLLM

此 NVFP4 权重适合 NVIDIA Blackwell。不要误用 MXFP4 权重，vLLM 在 NVIDIA 上对 MXFP4 的线性层支持尚不完整。配方要求 vLLM 至少 `0.17.0`，且 `transformers >= 5.8.0`。

## 安装

在 bjb1-892 确认 GPU 已正常挂载后执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh

conda create -y \
  -p /root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm \
  python=3.12 pip

conda activate /root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm

PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  pip install -U pip vllm 'transformers>=5.8.0'
```

验证：

```bash
python -c '
import torch, vllm, transformers
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
'
```

## 推荐启动命令

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm

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

参数说明：

- `--max-model-len 262144`：使用模型原生 262K 上下文。
- `--kv-cache-dtype fp8`：KV 缓存使用 FP8，释放更多显存给长上下文与并发。
- `--gpu-memory-utilization 0.95`：尽量使用 96GB 显存；若启动 OOM，优先降到 `0.90`。
- `--reasoning-parser qwen3`：将 `<think>` 内容解析为 reasoning，而不是混入最终回答。
- `--enable-auto-tool-choice` 与 `--tool-call-parser qwen3_xml`：启用自动工具调用及 XML 格式解析。
- `--speculative-config`：启用模型内置 MTP 草稿头，适合低延迟、小批量场景。
- `--mm-encoder-tp-mode data`：多模态编码器采用数据并行模式；单卡下无额外并行收益，但与配方功能组合一致。

## 初次运行与排障

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models
```

若出现显存不足，按以下顺序处理：

1. 将 `--gpu-memory-utilization 0.95` 降至 `0.90`。
2. 将 `--max-num-seqs` 降至 `16`。
3. 暂时移除 `--speculative-config`，先验证基础服务。

不要一开始就缩短 `--max-model-len 262144`；RTX PRO 6000 的 96GB 显存应优先尝试保留原生上下文。

## Python 调用

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
    timeout=3600,
)

response = client.chat.completions.create(
    model="qwen38-27b-nvfp4",
    messages=[{"role": "user", "content": "请解释什么是推测解码。"}],
    temperature=1.0,
    top_p=0.95,
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

设置自适应思考强度：

```python
extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}}
```

关闭思考模式：

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

## 当前环境注意事项

bjb1-892 为节省 GPU 计费，空闲时可以主动不挂载 GPU；此时 `/dev/nvidia*` 不存在或 `nvidia-smi` 看不到 GPU 属于预期状态，并不表示实例故障。模型同步不需要 GPU。仅在安装、CUDA 验证或启动 vLLM 前，才需要在云平台挂载 RTX PRO 6000。
