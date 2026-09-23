# QwenInferenceLab Python 评测器

[English](README.md) | 中文

`qwen-evaluator/` 为 QwenInferenceLab 提供可复现、可审计的数学答案评测。它复用仓库根目录 `.env` 和现有 MongoDB。

## 安装

在仓库根目录执行：

```bash
conda activate qwen-evaluator
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple pip install -e "./qwen-evaluator[dev]"
```

要求 Python 3.11；Math-Verify 固定为 0.9.0，ANTLR runtime 固定为 4.13.2。

## 已实现的命令接口

```bash
python -m qwen_inference_lab.evaluator.cli --help
python -m qwen_inference_lab.evaluator.cli audit-gold
python -m qwen_inference_lab.evaluator.cli evaluate-run --run-id <run-id>
python -m qwen_inference_lab.evaluator.cli evaluate --batch-state <state.json>
python -m qwen_inference_lab.evaluator.cli report --batch-state <state.json>
```

以上四个子命令均已实现：

- `audit-gold`：扫描全部 Questions，创建或更新 Gold Profile。
- `evaluate-run`：按 `runId` 调试或重新评测单条 Run。
- `evaluate`：读取一个或多个 batch `state.json`，评测其中明确记录的 Run。
- `report`：输出逐批及 All Batches 汇总统计，并可生成待人工核验的 manifest。

### 单条评测

```bash
python -m qwen_inference_lab.evaluator.cli evaluate-run \
  --run-id <run-id> \
  --max-level 3
```

需要覆盖现有 `evaluator-v1` 结果时添加 `--force`。如默认 Judge 配置不唯一，可通过 `--judge-model-config-id <id>` 明确指定。

### 批量评测

```bash
python -m qwen_inference_lab.evaluator.cli evaluate \
  --batch-state test-results/<batch1>/state.json \
  --batch-state test-results/<batch2>/state.json \
  --max-level 3 \
  --concurrency 4
```

`--batch-state` 可以重复传入。`--max-level` 允许 1、2、3，默认 3；`--concurrency` 范围为 1–16，默认 4，作用于 LLM Judge。当前暂不使用 Math-Verify：除多问题走 Level 2 Judge 外，其余可评测答案统一走 Level 3 语义 Judge。已有最终结果默认跳过，`MAX_LEVEL_REACHED` 占位记录可在提高 level 后续跑。

`--judge-timeout` 控制单次 Judge 超时（默认 60 秒），`--judge-retries` 控制失败重试次数（默认 2）。普通续跑会自动重试 `JUDGE_FAILED`、`JUDGE_UNAVAILABLE` 和 `MAX_LEVEL_REACHED`，不会覆盖已有 correct、incorrect 或 review。

`--local-only` 是维护/修复选项，当前只处理 Runtime Precheck，不调用 LLM Judge。通常无需使用。

### 报告与人工审计

生成约 50 条分层人工审计清单：

```bash
python -m qwen_inference_lab.evaluator.cli report \
  --batch-state <state.json> \
  --audit-manifest qwen-evaluator/evaluation-audit-manifest.json \
  --audit-size 50
```

清单中的 `humanVerdict` 和 `humanNotes` 保持为空，必须由人工填写。它不是自动生成的人工 Gold。

报告会列出每个 Batch 和 All Batches 的 Runtime、评测层级、方法、数学 verdict 与 reasonCode。已知三个目标批次还会保留命令记录中的 concurrency 元数据（4/8/8）。

## 安全与持久化

所有写操作前都会 ping MongoDB，并核验数据库名、Questions 数量和 Runs 数量。评测器只写 `question_gold_profiles` 与 `evaluations`，不修改原始 Questions 或 Runs 的 answer/reasoning。API Key 只在内存中使用，不会写入证据。

## 测试

```bash
pytest qwen-evaluator/tests
```
