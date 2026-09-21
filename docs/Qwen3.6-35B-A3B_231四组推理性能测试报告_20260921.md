# Qwen3.6-35B-A3B：231 四组推理性能测试报告

> 测试日期：2026-09-21
> 数据来源：MongoDB collection `qwen36_ab_benchmark_20260921`
> 服务：231，单张 RTX 5090；模型：`nvidia/Qwen3.6-35B-A3B-NVFP4`；vLLM：`v0.28.0`

## 1. 结论

1. 在本机、同一模型与相同服务端基础参数下，**仅加入 `--enforce-eager` 会显著降低单请求 Decode 速度**：无 MTP 的 A 组为 288.91 tok/s，B 组为 11.24 tok/s，约慢 **25.7 倍**。
2. 在 eager 路径上加入 MTP@4，B→C 从 11.24 提升至 39.07 tok/s，约 **3.48 倍**；但仍明显慢于非 eager 路径。
3. D 组（MTP@4、无 eager）使用修正后的 Qwen3.6 官方 sampling 参数，记录平均 **448.75 tok/s**；单题 MATH-040 为 **474.59 tok/s**。
4. A/B/C 使用的是早期错误客户端参数；D 使用修正后的官方参数。因此，A/B/C 之间可以做严格对照，D 仅用于证明“官方参数 + MTP + 非 eager”能跑通且很快，**不应用 D 与 A/B/C 做严格定量差值结论**。

## 2. 四组服务端矩阵

除下表两项外，四组保持相同服务端参数：TP=1、NVFP4、FP8 KV、20,480 context、`max-num-seqs=2`、Text Only、Marlin MoE、FlashInfer attention。

| 组别 | `--speculative-config` | `--enforce-eager` | MongoDB runId |
| --- | --- | --- | --- |
| A | 无 | 无 | `qwen36-ab-20260921` |
| B | 无 | 有 | `qwen36-ab-20260921` |
| C | MTP@4，`flashinfer_cutlass` | 有 | `qwen36-ab-20260921` |
| D | MTP@4，`flashinfer_cutlass` | 无 | `qwen36-manual-d-20260921` |

MTP 配置：

```json
{"method":"mtp","num_speculative_tokens":4,"moe_backend":"flashinfer_cutlass"}
```

## 3. 测试题目与过程

每组均先用“计算 1+1”预热两次，再按单请求、串行方式运行两题。提示词后缀固定：

```text
请逐步推理，并将最终答案放在 \boxed{} 中。
```

| ID | 难度 | 题型 | 参考答案 |
| --- | --- | --- | --- |
| MATH-040 | 简单 | 复数计算 | `-3` |
| MATH-187 | 简单 | 函数 / 绝对值方程 | `[1,\pi]` |

采集字段包括完整服务端参数、客户端参数、题目快照、预热原始结果、reasoning、最终正文、usage、首 token 时间、总耗时、Decode TPS、`\boxed{}` 合规和答案匹配启发式结果。

## 4. 结果

| 组别 | 完成 / 2 | boxed / 2 | 启发式答案匹配 / 2 | 平均总耗时 | 平均 Decode TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 1 | 1 | 1 | 9.89 s | **288.91** |
| B | 0 | 0 | 0 | 364.67 s | **11.24** |
| C | 0 | 0 | 0 | 105.31 s | **39.07** |
| D | 1 | 1 | 1 | 7.20 s | **448.75** |

逐题结果：

| 组别 | MATH-040 | MATH-187 |
| --- | --- | --- |
| A | `stop`，5.35 s，290.13 tok/s，`\boxed{-3}`，匹配 | `length`，14.42 s，287.70 tok/s |
| B | `length`，363.77 s，11.27 tok/s | `length`，365.58 s，11.21 tok/s |
| C | `length`，101.88 s，40.35 tok/s | `length`，108.73 s，37.79 tok/s |
| D | `stop`，4.52 s，474.59 tok/s，`\boxed{-3}`，匹配 | `length`，9.89 s，422.90 tok/s |

## 5. 参数口径与限制

### A/B/C：历史测速参数

```text
temperature=0
top_p=1
top_k=20
max_tokens=4096
chat_template_kwargs={enable_thinking:true, reasoning_effort:"low"}
```

Qwen3.6 不支持 `reasoning_effort` 分档；这三组参数不是该模型的官方推荐设置，但因为三组完全相同，仍可用于判断 eager 与 MTP 的相对性能影响。

### D：修正后的官方参数

```text
temperature=1.0
top_p=0.95
top_k=20
min_p=0.0
presence_penalty=1.5
repetition_penalty=1.0
max_tokens=4096
chat_template_kwargs={enable_thinking:true}
```

官方模型卡对通用 thinking 建议该 sampling 组合，并建议大多数请求使用 32,768 output-token 预算；本次 D 为与历史限长测试一致，仍使用 4,096 上限。详见 [Qwen3.6 官方模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/README.md)。

## 6. 对结果的解释

- `--enforce-eager` 关闭 CUDA Graph；在 Qwen3.6 的 GDN/MoE 推理路径中，单 token 的 kernel launch 开销会成为主导，因此 B 组极慢。
- MTP@4 能在 eager 下批量提出、验证多个 token，故 C 明显快于 B；但它不能完全抵消 eager 的 launch 开销。
- MATH-187 的 reasoning 在全部组别都耗尽 4,096 token，属于此次“逐步推理 + boxed”提示下的反刍/预算截断现象，不能据此判定数学能力错误；该题的最终答案没有输出，所以本报告不将其计为正确。
- 此报告是小样本性能冒烟，不是质量评测。要比较模型质量或正式选型，需在固定 Calibration/Holdout 集上，用官方参数和更充足的输出预算重测。

## 7. 可复用脚本

- `scripts/qwen36-231-service.sh`：启动、停止、查看单个 A/B/C/D 服务。
- `scripts/qwen36-benchmark.ts`：从 MongoDB 读取固定题目，预热、测试并写入 collection。
- `scripts/run-qwen36-ab-231.sh`：顺序运行四组；可用 `QWEN36_MAX_TOKENS` 为四组统一指定输出预算。

例如，手动测试 D：

```bash
./scripts/qwen36-231-service.sh D start
npx tsx scripts/qwen36-benchmark.ts \
  --variant D \
  --run-id qwen36-manual-d-$(date +%Y%m%d-%H%M%S) \
  --collection qwen36_ab_benchmark_20260921 \
  --max-tokens 4096
./scripts/qwen36-231-service.sh D stop
```
