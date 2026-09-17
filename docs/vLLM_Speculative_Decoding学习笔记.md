# vLLM Speculative Decoding 学习笔记

> 基于 [vLLM 官方 Speculative Decoding 文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)。  
> 面向场景：单请求、高考数学长推理、RTX 5090、Qwen3.8 Inferact NVFP4。

## 1. 它要解决什么问题

普通自回归 decode 每一步只能产生一个 token：target 模型要反复做很多次小 batch 的前向计算。投机解码的思路是：

```text
draft 一次提议多个 token
        ↓
target 一次验证这组 token
        ↓
接受最长有效前缀，再继续下一轮
```

目标是减少 target 的串行 decode 次数，从而降低每 token 延迟。vLLM 将它定位为：**中低 QPS、受显存带宽限制的负载**最有机会获益；高并发时增益与参数选择会不同。[官方文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

对你的业务，QPS 低、单题推理长，属于应该优先验证投机解码的场景。

## 2. vLLM 支持的方法：如何理解与选择

| 方法 | 是否需要外部模型 | 适合什么 | 你的优先级 |
|---|---|---|---|
| MTP | 否，模型原生支持时 draft head 内置 | 新模型最快建立投机基线 | **P0：Qwen3.8 Inferact** |
| Draft model | 是，小型普通 draft LLM | target 无原生 MTP 时的通用方案 | P2 |
| EAGLE / Eagle3 | 是，专门训练的 draft head | 通用的高收益模型式投机 | P2 |
| DFlash | 是，块并行 diffusion draft | 同步单请求可能收益很高，但部署复杂 | 已有双 5090 经验，P1 |
| DSpark | 是，DFlash 的扩展 draft | 根据置信度调节验证深度 | P2 |
| N-gram | 否 | 有大量文本重复、快速试验 | P3 |
| Suffix decoding | 否 | 利用前缀/后缀匹配，动态深度 | P3 |

官方的选择原则是：MTP、EAGLE、draft model 等模型式方法一般有更高的延迟收益；n-gram 与 suffix 不需额外模型，但收益较温和。实际收益取决于模型、硬件、采样参数和负载，而不是方法名称本身。

## 2.1 各方法的简单实现原理与选择逻辑

### MTP（Multi-Token Prediction）

```text
target 主干产生当前 hidden state
        ↓
内置 MTP head 同时预测后续多个位置的候选 token
        ↓
target 验证并接受最长有效前缀
```

MTP 的 draft head 是模型训练阶段一起训练、并随 checkpoint 发布的附加 head，不需要单独的草稿模型服务。它的优势是加载简单、target/draft 天然兼容、额外显存较小；限制是只有原生带 MTP 的模型才可用。

**何时选：** target 模型明确原生支持 MTP 时，优先从 MTP 开始。对 Qwen3.8，这是最短、最稳的投机解码路径。

### DFlash

```text
target 输出中间 hidden states
        ↓
独立 DFlash draft 接收 hidden states + mask embeddings
        ↓
并行预测一个 token block
        ↓
target 验证该 block，接受最长有效前缀
```

DFlash 是独立训练的 block-diffusion draft。它的关键是“块内并行”：不同于一枚一枚地产生草稿 token，它用非因果 attention 在一次前向中预测整块 token。潜在单请求加速很高，但需要额外 checkpoint、显存和严格的 target/draft 配对。

**何时选：** 已有针对 target 训练好的 DFlash draft，并且目标是同步、单请求低延迟时。你已有的 Qwen3.6 双 5090 DFlash 经验就是这一类。

### EAGLE / Eagle3

```text
target hidden states
        ↓
独立、较小的 EAGLE draft head
        ↓
自回归地产生多个候选 token
        ↓
target 验证并接受
```

EAGLE 是专门训练的 draft head，利用 target hidden states 提议 token。与 DFlash 的块并行不同，它更接近自回归 draft，因此工程成熟度通常更高，适合作为“没有原生 MTP、但可获得配对 speculator”时的通用高收益方案。

**何时选：** target 没有 MTP，但有可信、已验证的 EAGLE draft；或需要比普通小模型 draft 更紧密的 target 对齐。

### 普通 Draft Model（`draft_model`）

```text
小型 LLM 自回归地产生候选 token
        ↓
大型 target 验证候选 token
```

这是最直观的方式：加载一个更小、更快的 LLM 作为 drafter。默认情况下 target 与 draft 最好共享 tokenizer/词表；异词表可使用 TLI，但会增加限制与复杂度。

**何时选：** 没有原生 MTP、也没有专用 EAGLE/DFlash draft，但有适配的小模型可用。它是通用备选，不是 Qwen3.8 当前第一选择。

### N-gram

```text
在当前 prompt / 已生成 token 中查找重复 n-gram
        ↓
直接复用后续 token 作为候选
        ↓
target 验证
```

不需要任何额外神经网络或模型权重，几乎不增加显存。它依赖文本中已有重复模式，代码补全、固定模板、重复文档的收益更明显；数学推理的新 token 很多，收益通常有限。

**何时选：** 想零模型成本快速尝试，或负载包含大量可复用文本时。

### Suffix decoding

```text
在请求历史/缓存后缀中匹配当前序列
        ↓
按匹配长度动态决定可提议的 token 深度
        ↓
target 验证
```

Suffix decoding 同样无需外部 draft，但它维护后缀匹配结构，利用重复前缀或历史内容动态调整提议深度。适合存在重用和重复片段的请求流。

**何时选：** 多轮、模板化、前缀复用明显的服务；不适合把它当作高考数学单题的主要加速方案。

### 快速选择树

```text
模型是否原生带 MTP？
├── 是 → 先用 MTP
└── 否
    ├── 是否有已验证的 DFlash / EAGLE speculator？
    │   ├── 有 DFlash，且追求同步单请求极速 → DFlash
    │   └── 有 EAGLE → EAGLE
    ├── 是否有兼容的小型 draft LLM？
    │   └── 有 → draft_model
    └── 请求是否高度重复？
        ├── 是 → n-gram 或 suffix
        └── 否 → no-spec 基线，或训练专用 speculator
```

## 2.2 Qwen3.8 官方实际选择了什么，为什么

对你正在部署的 Qwen3.8-27B，vLLM 官方配方选择的是：

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

不是 DFlash、EAGLE 或普通外部 draft。

官方配方的直接依据是：Qwen3.8 checkpoint 内置 MTP head，并说明该 MTP head 在各精度 checkpoint 中可用；配方给出 `num_speculative_tokens=3` 作为启用方式。vLLM 通用文档的选择表也将 MTP 定义为“target 原生支持 MTP 时的最佳选择”。[Qwen3.8 vLLM 配方](https://recipes.vllm.ai/Qwen/Qwen3.8-27B?hardware=rtx_5090&variant=nvfp4&features=tool_calling%2Creasoning%2Cspec_decoding%2Cencoder_parallel) [vLLM 方法选择表](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

这并不表示 DFlash 一定更慢，而是官方选择 MTP 的工程理由更强：

1. 无需额外下载和加载 draft；单张 RTX 5090 的显存预算更容易控制。
2. draft 与 target 来自同一 checkpoint，结构、词表和 chat template 天然匹配。
3. 官方已将该组合写入 Qwen3.8 的可复现部署配方。
4. 对新模型的第一轮部署，先建立 MTP/no-spec 基线，才能公平判断后续 DFlash、DSpark 或其他 speculator 是否值得引入。

## 3. 配置接口：`--speculative-config`

vLLM 将投机相关设置统一放在 JSON 中：

```bash
--speculative-config '{
  "method": "<方法>",
  "model": "<外部 draft，必要时才填写>",
  "num_speculative_tokens": <每轮提议 token 数>
}'
```

最常用字段：

| 字段 | 含义 | 你的注意点 |
|---|---|---|
| `method` | `mtp`、`dflash`、`draft_model` 等 | 显式写出，保证实验可复现 |
| `model` | 外部 draft / EAGLE head / 辅助模型路径 | MTP 通常可省略；DFlash 必填 |
| `num_speculative_tokens` | 每轮 draft 提议数 | 是速度、接受率、显存之间的关键平衡点 |
| `draft_tensor_parallel_size` | 外部 draft 的 TP | 不要误写成 `tensor_parallel_size` |
| `max_model_len` | draft 的最大上下文 | 外部 draft 场景才重点考虑 |
| `rejection_sample_method` | `standard`、`synthetic`、`block` | 正式质量测试先保持默认 `standard` |

采样参数如 `temperature`、`top_p` 不属于 `speculative-config`；它们必须在请求侧或服务默认值中固定。旧式的 `--speculative-model` + 分散参数已被 vLLM 标记为弃用，应统一使用 `--speculative-config`。[官方 schema](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

## 4. 为什么 Qwen3.8 MTP 不写 `model`

Inferact Qwen3.8-27B-NVFP4 带有原生 MTP 结构。其 target checkpoint 同时携带 target 主干与 MTP draft head 的配置/权重；vLLM 选择 `method: mtp` 后，从当前 target checkpoint 中加载该 head。

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

这不是“没有 draft”，而是“draft 已内置”。你不需要单独下载、挂载和传递 draft 路径。

与之相对，Qwen3.6 的 DFlash draft 是独立的 `Qwen3.6-35B-A3B-DFlash` checkpoint。target 没有 DFlash 的 block-diffusion 层，因此必须明确：

```bash
--speculative-config '{
  "method": "dflash",
  "model": "/data/metahuman_work/models/Qwen3.6-35B-A3B-DFlash",
  "num_speculative_tokens": 10
}'
```

## 5. MTP、DFlash 与普通 draft 的工程取舍

| 维度 | MTP | DFlash | 普通 Draft Model |
|---|---|---|---|
| 草稿来源 | checkpoint 内置 head | 独立专用 block draft | 独立小 LLM |
| 部署难度 | 最低 | 最高 | 中等 |
| 额外显存 | 较低 | draft 模型与额外状态 | draft 模型 |
| target 兼容性 | 仅限原生支持的模型 | draft 与 target 必须配对 | 通常需同词表；异词表需要 TLI |
| 你的用途 | Qwen3.8 首选 | 双 5090 已验证、后续优化 | 非原生 MTP 模型备用 |

DFlash 的特点是并行预测整个 block；它会使用 target 中间 hidden states、mask embeddings 和独立 draft 层。潜在收益高，但模型配对、显存和版本兼容也更复杂。[DFlash 官方说明](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)

## 6. “无损”到底意味着什么

vLLM 说明：其 speculative decoding 在理论和算法层面目标是保持 target 的输出分布；greedy decode 应与无投机结果一致，采样模式经 rejection sampler 校正。

但不能把“无损”理解成每次运行的 token 序列绝对逐字一致。浮点精度、batch 大小、数值稳定性和 logprob 稳定性都可能带来差异。因此对你的高考数学业务仍必须实际评测：

```text
固定 prompt、temperature、top_p、top_k、max_tokens、并发
比较 no-spec 与 MTP 的 Pass@1、截断率、Time-to-correct
```

不要只因“理论无损”就跳过数学正确率验证。[官方无损说明](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

## 7. 如何选择 `num_speculative_tokens`

这个值不是越大越好：

- 值小：draft/verify 开销低，但每次推进 token 少。
- 值大：一次 target verify 可推进更多 token，但后段接受率通常下降，draft 和显存开销也增加。
- 最优点由 target 速度、draft 速度、接受长度、输出长度、采样策略和 GPU 决定。

你已有实测已经说明这一点：DFlash `k=10` 的 acceptance rate 低于 `k=5`，但最终 Decode TPS 更高。

对 Qwen3.8 内置 MTP：先遵循官方配方的 `3`，只在 no-spec/MTP 数学基线稳定后逐步扫参；每次只改一个值。

## 8. 你的标准 benchmark 方法

1. 固定模型、权重 revision、镜像、GPU、prompt、采样、最大输出和并发=1。
2. 分开建立 no-spec、MTP、DFlash 的基线；不要同时换模型、量化和投机方法。
3. 记录 TTFT、完整延迟、输出 token、Decode TPS、finish reason、显存、接受长度与每位置接受率。
4. 使用数学题 holdout 判断 Pass@1；只保留正确率不低于基线的加速配置。
5. 以 **Time-to-correct** 作为最终生产排序，而非仅以 token/s 排序。

## 9. 重要边界

- 现有 vLLM 文档指出某些旧版本中 pipeline parallelism 与 speculative decoding 不能组合；部署前应按当前镜像版本复核。
- 若使用外部 draft，默认需要相同词表。异词表 draft 可用 `use_heterogeneous_vocab=true` 的 TLI，但目前受 greedy draft sampling 等限制，不适合你的第一轮正式质量测试。
- 动态 speculative decoding、adaptive verification、custom proposer 属于后续进阶项；在基础 MTP 与 DFlash benchmark 未稳定前，不投入。

## 10. 你的学习检查清单

- [ ] 能解释 MTP 为什么不传 `model`，DFlash 为什么必须传。
- [ ] 能从启动日志确认 MTP / DFlash 已实际加载，而不是静默回退。
- [ ] 能区分客户端 Decode TPS、服务端 generation throughput、acceptance 指标。
- [ ] 能解释一次 k 增大后“接受率下降但速度上升”的原因。
- [ ] 能用固定数学 holdout 判断加速是否牺牲 Pass@1。
- [ ] 能定位权重、KV cache、draft、CUDA Graph、TP 通信和 GDN state 的显存/延迟影响。
