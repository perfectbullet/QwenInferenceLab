# Qwen3.8-27B 在 231 服务器（RTX 5090 32GB）部署选型指南（归档）

> 服务器：`192.168.8.231`
> GPU：NVIDIA GeForce RTX 5090 32GB
> CUDA：13.0
> PyTorch：2.13.0+cu130
> 目标上下文长度：10K / 20K / 30K tokens

---

## 一、核心结论

| 推荐度 | 模型 | 大小 | 适用场景 |
|---|---|---|---|
| **首选** | `unsloth/Qwen3.8-27B-NVFP4` | 19.87B | 质量与显存平衡最佳 |
| **次选** | `Inferact/Qwen3.8-27B-NVFP4` | 17.63B | 更省显存，长文更稳 |
| **次选** | `RadixArk/Qwen3.8-27B-NVFP4` | 18.59B | 介于前两者之间 |
| **兜底** | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | 27.78B（压缩后约 7–8GB）| 30K 上下文稳跑 |
| **兜底** | `Abiray/Qwen3.8-27B-Q4_K_M-GGUF` | 约 7–8GB | 走 llama.cpp/ollama |

**不推荐**：FP8 / INT8 / W8A16 / w8a8 权重版本（32GB 显存跑 20K+ 上下文会爆）。

---

## 二、为什么 32GB 跑不了 FP8 / INT8 的 30K 上下文？

Qwen3.8-27B 估算参数：64 层、hidden size 4096、head dim 128。

| 项目 | FP16 | FP8 | INT8 | INT4 |
|---|---|---|---|---|
| 模型权重 | ~54 GB | ~27 GB | ~27 GB | ~7–8 GB |
| KV Cache / 1 token | ~1 MB | ~0.5 MB | ~0.5 MB | ~0.25 MB |
| KV Cache / 30K tokens | ~30 GB | ~15 GB | ~15 GB | ~7.5 GB |

组合估算：

- FP8 权重（27 GB）+ FP8 KV（15 GB）≈ **42 GB** → 爆显存 ❌
- FP8 权重（27 GB）+ INT4 KV（7.5 GB）≈ **35 GB** → 仍超 32GB ❌
- NVFP4 权重（~20 GB）+ INT4 KV（~7.5 GB）≈ **27–28 GB** → 可行 ✅
- AWQ-INT4 / GGUF Q4（~8 GB）+ FP8 KV（15 GB）≈ **23 GB** → 很稳 ✅

**结论**：要在 RTX 5090 32GB 上跑 30K tokens，必须用 **4-bit 权重 + KV Cache 量化**。

---

## 三、各版本详解

### 1. `unsloth/Qwen3.8-27B-NVFP4`（19.87B）⭐ 首选

- **优势**：
  - RTX 5090 原生支持 FP4 Tensor Core，NVFP4 在这代卡上速度和显存效率最好。
  - Unsloth 量化质量普遍较高。
  - 19.87B 比 FP8 省约 8GB，可留给 KV Cache。
  - 231 服务器当前 ComfyUI 已在使用 NVFP4/AWQ 混合的 text encoder，环境支持这类格式。

- **注意**：
  - 跑 30K tokens 时，建议同时开启 **KV Cache INT4/INT8/FP8**。
  - 若用 FP8 KV Cache，20GB 权重 + 15GB KV ≈ 35GB，仍可能 OOM。

### 2. `Inferact/Qwen3.8-27B-NVFP4`（17.63B）

- 比 Unsloth 版本更小，长上下文更稳。
- 量化更激进，质量损失可能略大，但 20K–30K 可用性更好。

### 3. `RadixArk/Qwen3.8-27B-NVFP4`（18.59B）

- 大小介于 Unsloth 和 Inferact 之间。
- 如果 Unsloth 版本 30K 不稳，可尝试这个。

### 4. AWQ-INT4 / GGUF Q4_K_M（兜底方案）

- **AWQ-INT4**：`cyankiwi/Qwen3.8-27B-AWQ-INT4`
  走 Transformers / AutoAWQ / vLLM，权重约 7–8GB。
- **GGUF Q4_K_M**：`Abiray/Qwen3.8-27B-Q4_K_M-GGUF` / `ggml-org/Qwen3.8-27B-GGUF`
  走 llama.cpp / ollama，权重约 7–8GB。

- **优势**：30K tokens 稳跑，显存余量充足。
- **劣势**：4-bit 权重质量比 NVFP4 略差；对长文推理质量敏感的任务优先选 NVFP4。

---

## 四、不推荐的版本

| 版本 | 原因 |
|---|---|
| `千问/Qwen3.8-27B-FP8` | 权重 27.78GB，加 KV Cache 后 32GB 不够跑 20K+ |
| `unsloth/Qwen3.8-27B-FP8` | 同上 |
| `KenDua.../Qwen3.8-27B-INT8-W8A16-MTP` | 同上 |
| `Eco-Tech/Qwen3.8-27B-w8a8` | 同上 |
| `cyankiwi/Qwen3.8-27B-AWQ-FP8` | 27.78GB，仍按 FP8 存权重，未解决显存瓶颈 |
| `lmstudio-community/Qwen3.8-27B-MLX-4bit` | **MLX 是 Apple Silicon 格式**，NVIDIA 无法使用 |
| `m1x-community/Qwen3.8-27B-mlx` | 同上 |
| `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` | AMD Quark 方案，与 CUDA 13 / PyTorch 2.13 生态兼容性不确定 |

---

## 五、vLLM 部署示例

### 方案 A：NVFP4 + FP8 KV Cache（推荐先尝试）

```bash
vllm serve unsloth/Qwen3.8-27B-NVFP4 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95
```

### 方案 B：NVFP4 + INT4 KV Cache（30K 不稳时降级）

```bash
vllm serve unsloth/Qwen3.8-27B-NVFP4 \
  --max-model-len 32768 \
  --kv-cache-dtype int4 \
  --gpu-memory-utilization 0.95
```

### 方案 C：AWQ-INT4（最稳但质量稍降）

```bash
vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90
```

> 注：vLLM 的 `kv-cache-dtype` 具体可用值需根据版本确认，常见为 `auto`、`fp8`、`fp8_e5m2`、`int8`、`int4`。

---

## 六、快速决策流程

```
是否需要 30K 上下文？
├── 是 → 是否有 48GB+ 显存？
│   ├── 是 → 用 FP8 / BF16 权重
│   └── 否（32GB）→ 用 NVFP4 权重 + KV Cache 量化
│       ├── 30K 能跑稳吗？
│       │   ├── 能 → unsloth NVFP4
│       │   └── 不能 → AWQ-INT4 或 GGUF Q4_K_M
│       └── 质量要求极高？
│           ├── 是 → 优先 NVFP4
│           └── 否 → AWQ-INT4 也可接受
└── 否（≤10K）→ NVFP4 完全够用
```

---

## 七、关键记忆点

1. **RTX 5090 32GB 跑 Qwen3.8-27B 长上下文，瓶颈在 KV Cache，不在模型权重。**
2. **NVFP4 是 50 系显卡的最优量化格式**，比 AWQ-INT4 更适合 5090。
3. **跑 30K tokens 必须同时压缩权重和 KV Cache**。
4. **MLX 格式是给 Mac 用的，NVIDIA 不能跑。**
5. **AWQ-INT4 / GGUF Q4 是长上下文的兜底方案**，质量略降但稳。

---

## 八、当前部署状态（2026-08-18 更新）

### 8.1 模型下载 ✅ 已完成

| 项目 | 状态 |
|---|---|
| 模型 | `unsloth/Qwen3.8-27B-NVFP4` |
| 落盘位置 | `/nfs-data/metahuman_work/models/unsloth/Qwen3.8-27B-NVFP4` |
| 主权重 | `model.safetensors`（22.5 GB，与索引核对一致） |
| MTP 模块 | `model_mtp.safetensors`（849 MB，可选，用于投机解码） |
| 配套文件 | config / tokenizer / chat_template 等全部就绪（13/13） |
| 下载耗时 | 约 1 小时 52 分（hf-mirror + hf_transfer） |

> 注：目录里 `.cache/huggingface/download/*.incomplete`（3.0 GB）是首次中断下载的残留缓存，确认服务正常后可删除释放空间。

### 8.2 vLLM 镜像 ✅ 已就绪（无需下载）

| 镜像 | 状态 |
|---|---|
| `vllm/vllm-openai:v0.24.0` | ✅ 完整（9.2 GB），**使用此镜像** |
| `vllm/vllm-openai:v0.11.0` | ⚠️ 残缺（69 MB），不可用 |
| `lmsysorg/sglang:v0.5.16-cu129` | 备选推理引擎 |

### 8.3 启动命令（模型已落盘，直接用本地路径）

```bash
docker run -d --gpus all --name vllm-qwen38 \
  -v /nfs-data/metahuman_work/models/unsloth/Qwen3.8-27B-NVFP4:/model \
  -p 8000:8000 \
  --restart unless-stopped \
  vllm/vllm-openai:v0.24.0 \
  --model /model \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95
```

验证服务：

```bash
curl http://192.168.8.231:8000/v1/models
```

若 32K 上下文 OOM，按第五章方案 B 降级（`--kv-cache-dtype int4` 或 `--max-model-len 20480`）。

> 注意：231 是共享服务器，启动前先 `nvidia-smi` 确认显存空闲（ComfyUI 的 comfyui-h3 容器可能占用 GPU）。

### 8.4 运维速查

```bash
# 查看容器日志（vLLM 启动需数分钟加载权重）
docker logs -f vllm-qwen38

# 停止 / 重启
docker stop vllm-qwen38 && docker rm vllm-qwen38
docker restart vllm-qwen38

# 显存占用
nvidia-smi
```

---

*文档生成时间：2026-08-17；最近更新：2026-08-18*

---

## 九、vLLM 单张 RTX 5090 官方配方更正（2026-08-18）

本指南早期将 `unsloth/Qwen3.8-27B-NVFP4` 作为 vLLM 单卡首选。根据 vLLM 官方 Qwen3.8-27B 配方，**单张 RTX 5090 + NVFP4 的正式主模型应改为：**

```text
Inferact/Qwen3.8-27B-NVFP4
```

官方单卡要点：

```bash
vllm serve Inferact/Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --mm-encoder-tp-mode data \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

`--enforce-eager` 是单卡关键参数：官方说明单张 RTX 5090 的 NVFP4 方案需要关闭 CUDA Graph，避免启动阶段的 CUDA Graph capture OOM。

已下载的 Unsloth NVFP4 权重不删除，可作为后续权重 A/B 使用；但不应在未验证的情况下继续作为单卡 vLLM 正式数学基线。对 Unsloth 的运行能力，官方确认其在 Blackwell 上有真实 NVFP4 kernel 路径；但该页面针对**单卡 RTX 5090**给出的验证命令是 Inferact，而非 Unsloth。
