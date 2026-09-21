# MTP 与 DFlash：原理、配置差异与部署学习清单（归档）

## 1. 为什么 MTP 不写 draft 模型路径，DFlash 必须写？

| 项目 | Qwen3.8 内置 MTP | Qwen3.6 + DFlash |
|---|---|---|
| draft 在哪里 | 已在 target checkpoint 内，通常以 MTP 配置与 `model_mtp.safetensors` 等文件存在 | 独立 checkpoint，例如 `Qwen3.6-35B-A3B-DFlash` |
| 谁训练它 | 原模型训练阶段已训练 native MTP head | 另行训练的 block-diffusion speculator |
| vLLM 如何加载 | `method: mtp` 从 target 的 config / 权重中发现并加载 | `method: dflash` 无法从 target 推断 draft，必须由 `model` 给出路径 |
| 草稿产生方式 | target 主干算出 hidden state 后，由内置多 token head 预测后续 token | target 中间 hidden states + mask embeddings 输入独立 DFlash draft，一次并行预测一个 block |
| 显存与部署 | 不需单独挂载 / 下载 draft；额外开销主要是 checkpoint 内置 head 和验证 | target 与独立 draft 都要加载，占用额外显存，也要保证兼容性 |
| 调优对象 | MTP window、acceptance、target runtime | draft checkpoint、block size、anchor、acceptance、target/draft 交互 |

因此 Qwen3.8 的配置可以是：

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

没有 `model` 字段的含义不是“没有 draft”，而是“draft 已经嵌在 target checkpoint 内”。先确认 Inferact 目录含 MTP 权重：

```bash
find /nfs-data/metahuman_work/models/Inferact/Qwen3.8-27B-NVFP4 \
  -maxdepth 1 -iname '*mtp*.safetensors' -type f
```

相对地，DFlash 的 target 本身没有 DFlash 层与 block-diffusion 权重，所以必须显式给出：

```bash
--speculative-config '{
  "method": "dflash",
  "model": "/data/metahuman_work/models/Qwen3.6-35B-A3B-DFlash",
  "num_speculative_tokens": 10
}'
```

## 2. 两种方法的共同点与不同点

两者都遵循同一个大框架：draft 提议多个 token → target 验证 → 接受最长有效前缀 → 继续生成。正确的 accept/reject 实现下，最终分布由 target 保持；投机解码的目标是减少 target 的串行 decode 次数，而不是以未验证的 draft 代替 target。

不同之处在于 MTP 是“模型原生多 token head”，DFlash 是“外接、块并行的 draft 网络”。DFlash 一次可预测整块 token，潜在速度上限很高，但训练、兼容性、显存、版本和参数都更复杂。MTP 的工程路径短，适合作为新模型部署的第一条投机基线。

## 3. 通过部署这两类方法能学到什么

### 第一层：模型工件与兼容性

- 能从 `config.json`、权重文件和模型卡判断：模型是否原生支持 MTP、是否有独立 draft、draft 与 target 是否配对。
- 理解“同为 NVFP4”不等于可替换：量化方式、MTP head、KV scale、chat template 都会影响运行与质量。

### 第二层：推理系统的真实瓶颈

- 分清权重、KV cache、CUDA Graph、draft、Mamba/GDN state、TP 通信分别占什么资源。
- 理解单请求低 QPS 与高并发的最佳参数往往不同。
- 将 TTFT、Decode TPS、完整任务耗时、显存和功耗拆开分析，而不是只看单一 token/s。

### 第三层：投机解码调优

- 用 acceptance length 和每位置接受率解释 draft 质量。
- 但以最终 Decode TPS、数学 Pass@1 和 Time-to-correct 选择 k / window。
- 学会识别“acceptance 下降但整体更快”这类正常现象。

### 第四层：面向业务的评测工程

- 将高考数学题分成选择、填空、证明题，建立不同的答案提取与判题链路。
- 不在调参集上宣布准确率；以固定 holdout 验证 Pass@1。
- 把模型能力、prompt/workflow（COT/TIR）和 runtime 参数分开做 A/B，避免归因混乱。

## 4. 适合你的学习顺序

1. Qwen3.8 Inferact：单卡 vLLM no-spec → 内置 MTP@3；固定数学题、单请求、Pass@1。
2. 学会从启动日志确认 MTP head 已加载、记录 acceptance 与真实输出速度。
3. 在质量不降前提下比较 reasoning workflow、最大输出长度、MTP window。
4. 回到已有双 5090 DFlash 记录，继续 k=12 / k=15；把正确答案耗时加入现有 TPS 结论。
5. 最后才考虑 DSpark、SGLang 深度调优或训练新的 DFlash speculator。

## 5. 参考资料

- [vLLM Qwen3.8-27B RTX 5090 NVFP4 配方](https://recipes.vllm.ai/Qwen/Qwen3.8-27B?hardware=rtx_5090&variant=nvfp4&features=tool_calling%2Creasoning%2Cspec_decoding%2Cencoder_parallel)
- [vLLM Speculative Decoding 文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [vLLM DFlash 文档](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)
- [Speculators 算法概览](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/)
