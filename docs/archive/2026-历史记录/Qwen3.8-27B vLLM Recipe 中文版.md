# Qwen3.8-27B — vLLM 部署指南（归档）

## 1. 模型概述

**Qwen3.8-27B** 是 Qwen3.8 系列中的 **270 亿参数 Dense（稠密）模型**。

其主要特点：

- 27B Dense 参数
- 64 层 Transformer
- 其中：
  - 16 层使用 Full Attention
  - 48 层使用 Linear Attention
- 自带 Vision Tower，属于多模态模型
- 内置 **MTP Draft Head**
- 原生上下文长度：**262,144 tokens**
- 可以扩展至约 **1M tokens**
- 支持文本推理模式
- 支持 Thinking / Adaptive Thinking
- 支持 Speculative Decoding

模型采用混合注意力架构，每隔 4 层使用一次 Full Attention，其余层采用具有固定递归状态的 Linear Attention。

本 Recipe 主要验证的是**文本推理服务**。

---

# 2. 环境要求

需要：

```text
transformers >= 5.8.0
```

vLLM 自身会使用对应的 Qwen3.5/Qwen3.8 配置解析模型。

对于多模态处理，较新的 Transformers 主要用于 Qwen 的视觉 Processor。

---

# 3. 通用低延迟部署

## NVFP4，单卡 TP=1

基本命令：

```bash
vllm serve Inferact/Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

这里：

```text
tensor-parallel-size = 1
```

表示模型不进行 Tensor Parallel 切分。

KV Cache 使用：

```text
FP8
```

从而减少 KV Cache 的显存占用。

---

# 4. FP8 + 多卡

如果使用大型 Blackwell GPU，可以运行：

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3
```

这种方案的目的主要是获得更大的 KV Cache。

---

# 5. MTP 推测解码

Qwen3.8-27B checkpoint 内部已经包含 **MTP Head**。

因此不需要单独加载另一个 Draft Model。

启动时增加：

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

即可启用 MTP。

这里：

```text
num_speculative_tokens = 3
```

表示一次尝试提前预测最多 3 个后续 token。

其基本原理可以理解为：

```text
主模型生成 token
        ↓
MTP Head 提前预测后续 token
        ↓
主模型一次性验证
        ↓
预测正确
        ↓
一次接受多个 token
```

目的就是提高 Decode 吞吐。

---

# 6. 两张 RTX 5090

官方 Recipe 已验证：

```text
2 × RTX 5090
TP = 2
```

使用的是消费级 Blackwell：

```text
sm120
```

NVFP4 在 RTX 5090 上使用真正的 NVFP4 Kernel，而不是软件模拟。

vLLM 会选择类似：

```text
FlashInferCutlassNvFp4LinearKernel
```

执行 NVFP4 GEMM。

因此 5090 对 NVFP4 是原生支持路线。

例如：

```bash
vllm serve unsloth/Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

官方验证中，不同 NVFP4 checkpoint 的显存占用和 KV Cache 容量存在较明显差异。

Inferact 和 Unsloth 的 NVFP4 并不是完全相同的量化方式：

- Unsloth 版本属于混合精度方案
- Inferact 更接近统一 W4A4 路线

因此二者：

- 权重显存占用不同
- KV Cache 空间不同
- MTP 接受率也不同

不能简单认为所有“NVFP4”模型完全等价。

---

# 7. 单张 RTX 5090

这是对你最重要的一部分。

## RTX 5090 实际可用显存

虽然 RTX 5090 标称：

```text
32 GB
```

但 vLLM Recipe 测试环境实际可使用的大约是：

```text
31.4 GiB
```

Qwen3.8-27B NVFP4 **可以单卡装下**。

但是：

# 必须使用 `--enforce-eager`

官方单卡命令：

```bash
vllm serve Inferact/Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --reasoning-parser qwen3
```



---

# 8. 为什么必须 `--enforce-eager`

如果不使用：

```bash
--enforce-eager
```

vLLM 会进行 CUDA Graph Capture。

而 CUDA Graph Capture 阶段还需要额外申请显存。

官方测试中会出现大约：

```text
Tried to allocate 784 MiB
```

然后：

```text
torch.OutOfMemoryError
```

也就是说：

```text
模型已经加载
+
KV Cache 已经分配
+
剩余显存极少
+
CUDA Graph 再申请约 784 MiB
=
OOM
```

调：

```bash
--gpu-memory-utilization 0.80
```

或者：

```bash
--gpu-memory-utilization 0.93
```

都不能根本解决这个问题。

**真正决定单卡 5090 能不能启动的参数是：**

```bash
--enforce-eager
```



---

# 9. 单卡 5090 的 KV Cache

在：

```text
max-model-len = 32768
```

条件下，官方测试得到：

### 只使用 enforce-eager

大约：

```text
91,022 KV tokens
```

---

### 再加入纯文本模式

```bash
--language-model-only
```

提升到大约：

```text
135,926 KV tokens
```

---

### 再限制最大并发

```bash
--max-num-seqs 8
```

可以进一步达到大约：

```text
152,917 KV tokens
```

因此对于纯文本服务：

```bash
--language-model-only
--max-num-seqs 8
```

非常有价值。

它们可以进一步释放显存给 KV Cache。

不过要注意：

> 这两个参数不是解决启动 OOM 的关键。

解决 CUDA Graph OOM 的仍然是：

```bash
--enforce-eager
```



---

# 10. FP8 KV Cache 是否必须？

不是。

如果 KV Cache 使用：

```text
BF16
```

官方测试大约还有：

```text
76,458 KV tokens
```

所以 BF16 KV 也能跑。

但是 FP8：

```bash
--kv-cache-dtype fp8
```

能够显著节省 KV Cache 显存。

对 RTX 5090 32GB 来说，FP8 更合理。

---

# 11. 单卡 RTX 5090 + MTP

MTP Head 同样能够放入单张 5090。

官方测试大致获得：

```text
MTP acceptance ≈ 0.754
```

即 Draft Token 有较高比例能够被主模型接受。

因此单卡 5090 可以同时运行：

```text
Qwen3.8-27B NVFP4
+
FP8 KV Cache
+
MTP
```

---

# 12. 推荐的 RTX 5090 纯文本配置

结合 Recipe 中的几个优化，可以使用：

```bash
vllm serve Inferact/Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --language-model-only \
  --max-num-seqs 8 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

对于：

```text
RTX 5090 32GB
+
纯文本
+
数学推理
+
低延迟
```

这是很值得优先使用的配置。

---

# 13. Reasoning Parser

建议启用：

```bash
--reasoning-parser qwen3
```

Qwen 的 Chat Template 会生成 Thinking 内容。

Reasoning Parser 会把：

```text
<think>
...
</think>
```

中的推理内容，与最终回答内容正确拆分。

否则 Thinking 内容可能直接进入普通：

```text
message.content
```

对于 API 服务尤其需要注意。

---

# 14. 客户端调用

vLLM 对外提供 OpenAI Compatible API。

Python 示例：

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600,
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[
        {
            "role": "user",
            "content": "Give me three primes above 100."
        }
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=2048,
)

print(resp.choices[0].message.content)
```

模型默认 Generation Config 大致为：

```text
temperature = 1.0
top_p = 0.95
top_k = 20
```



---

# 15. Thinking 模式

Qwen3.8 支持：

### 不思考

```json
{
  "enable_thinking": false
}
```

这种模式直接回答。

适合：

- 简单问答
- 对延迟敏感的任务
- 简单选择题

---

### 自适应 Thinking

可以设置：

```json
{
  "reasoning_effort": "low"
}
```

可用档位包括：

```text
xhigh
medium
low
```

其中默认最高推理强度为：

```text
xhigh
```



对于高考数学，可以考虑：

```text
简单选择题
→ low

普通大题
→ medium

压轴题
→ xhigh
```

这样比所有问题都使用 xhigh 更有利于降低总体响应时间。

---

# 16. 262K 长上下文

Qwen3.8-27B 原生支持：

```text
262,144 tokens
```

即约：

```text
262K
```

因此正常情况下不需要额外的 RoPE 扩展。

---

# 17. 扩展到 1M Context

如果确实需要：

```text
1,000,000 tokens
```

可以通过 YaRN RoPE Scaling 进行扩展。

vLLM Recipe 会通过：

```text
--hf-overrides
```

修改模型的 RoPE 参数，并将：

```text
--max-model-len
```

提高到：

```text
1,000,000
```

不过官方特别提醒：

> 如果实际上并不需要超长 Prompt，不应该一直开启 YaRN。

因为 Static YaRN 会始终使用固定 Scaling Factor。

例如只需要大约：

```text
524K
```

上下文时，可以适当降低 Scaling Factor。



---

# 18. DFlash2

Qwen3.8-27B 还可以使用：

```text
DFlash2
```

进行另一种 Speculative Decoding。

DFlash2 不是完整模型，而是 Draft Model。

使用：

```text
incoai/Qwen3.8-27B-DFlash2
```

需要：

```text
vLLM >= 0.28.0
```

例如：

```bash
vllm serve Qwen/Qwen3.8-27B \
  --speculative-config \
  '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
```

对于你的 5090，目前优先测试**模型自身内置的 MTP**更简单。

---

# 19. NVIDIA 上不要使用 MXFP4

官方 Troubleshooting 特别指出：

> NVIDIA GPU 当前不要使用 MXFP4 路线。

因为当前 vLLM 的 NVIDIA MXFP4 实现缺少完整的 Linear Method 支持。

对于 RTX 5090：

# 使用 NVFP4

也就是：

```text
NVFP4 ✅

MXFP4 ❌
```



---

# 20. 华为 Ascend

Recipe 同样测试了：

```text
Ascend 950PR
```

包括：

- ModelSlim INT8 / W8A8
- 原生 Block-scaled FP8
- MTP
- vLLM Ascend

其中 W8A8 模型使用：

```bash
--quantization ascend
```

并可通过：

```bash
VLLM_USE_MODELSCOPE=True
```

直接从 ModelScope 加载。

这一部分与你的 RTX 5090 部署无直接关系，可以暂时忽略。

---

# 21. 针对 RTX 5090 的最终重点

如果只把整篇文档浓缩成与你当前环境最相关的部分，就是：

```text
RTX 5090 32GB
        │
        ▼
Qwen3.8-27B-NVFP4
        │
        ├── TP = 1
        │
        ├── 32K Context
        │
        ├── FP8 KV Cache
        │
        ├── --enforce-eager
        │       ↑
        │       单 5090 必须
        │
        ├── --language-model-only
        │       ↑
        │       纯文本数学推荐
        │
        ├── --max-num-seqs 8
        │
        ├── reasoning-parser=qwen3
        │
        └── MTP = 3
```

建议启动参数：

```bash
vllm serve Inferact/Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --language-model-only \
  --max-num-seqs 8 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

其中最容易踩坑的就是：

```text
不要删除 --enforce-eager
```

因为即使模型和 KV Cache 已经勉强塞进 5090，CUDA Graph Capture 仍可能因为额外的显存申请导致启动 OOM。

对于你现在的：

**RTX 5090 32GB + 高考数学 + 纯文本 + 速度优先**

这套 Recipe 的技术路线可以概括成：

> **NVFP4 降低模型权重显存 → language-model-only 去掉视觉部分 → FP8 KV 降低缓存显存 → MTP 提高生成速度 → 32K 限制上下文 → enforce-eager 保证单卡能够正常启动。**
