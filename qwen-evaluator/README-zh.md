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


## Capability Dataset Builder V1

从三个明确的 Batch 构建 250 条题目级能力记录：

```bash
python -m qwen_inference_lab.capability.cli build \
  --batch-state ../test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4/state.json \
  --batch-state ../test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4-round2/state.json \
  --batch-state ../test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4-round3/state.json
```

默认已存在的 `capability-v1` 记录会跳过；添加 `--force` 可幂等更新。构建同时写入 MongoDB `question_capabilities`，并生成：

- `artifacts/capability-dataset-v1.jsonl`
- `artifacts/capability-dataset-v1.csv`

查看已持久化数据的统计：

```bash
python -m qwen_inference_lab.capability.cli report
```

Dataset 保留 Question 文本和每次 Attempt 的结构化指标，不包含 `reference_answer`、完整模型 answer 或 reasoning。

## Embedding Dataset Builder + Retrieval Evaluation V1

为 250 条 `capability-v1` Question 构建或更新 Qwen3-Embedding-0.6B embedding：

```bash
python -m qwen_inference_lab.embedding.cli build
```

默认服务为 `http://192.168.100.233:8093/v1`，模型为 `Qwen/Qwen3-Embedding-0.6B`，Embedding 版本为 `qwen3-embedding-0.6b-v1`。可通过 `--base-url`、`--model`、`--timeout`、`--retries` 和 `--batch-size` 覆盖请求配置；添加 `--force` 可强制重新生成未变化的记录。Embedding 输入仅使用 Question 文本。

执行单题余弦相似度检索，或执行完整的 Leave-One-Out 评估：

```bash
python -m qwen_inference_lab.embedding.cli retrieve \
  --question-id MATH-001 \
  --top-k 10

python -m qwen_inference_lab.embedding.cli evaluate
python -m qwen_inference_lab.embedding.cli report
```

`evaluate` 会排除 Query 自身，并使用 NumPy 在内存中评估 Top-5/Top-10 邻域。`labelUsable=false` 的题目仍可作为检索结果展示，但不会参与 Local Success Mean 聚合。

Embedding 记录保存到 MongoDB `question_embeddings`，唯一键为 `(questionId, embeddingVersion)`。生成文件：

- `artifacts/embedding-dataset-qwen3-embedding-0.6b-v1.jsonl`（metadata 和完整 vector）
- `artifacts/embedding-dataset-qwen3-embedding-0.6b-v1.csv`（仅 metadata）
- `artifacts/retrieval-evaluation-qwen3-embedding-0.6b-v1.json`
- `artifacts/retrieval-neighborhoods-qwen3-embedding-0.6b-v1.jsonl`
- `artifacts/non-perfect-retrieval-qwen3-embedding-0.6b-v1.jsonl`

## Router V1 — KNN + OOD 离线评估与影子服务

执行固定随机种子的分层评估，或检查单道题的交叉验证预测：

```bash
python -m qwen_inference_lab.router.cli evaluate --folds 5 --seed 42
python -m qwen_inference_lab.router.cli report
python -m qwen_inference_lab.router.cli inspect --question-id MATH-151
```

Router V1 保留离线评估核心，并提供内部影子运行服务。决策特征只使用 embedding 和 Reference Corpus 的 Capability 标签，不使用 `difficulty`、`mathType`、`tag`、答案或 reasoning。每个 held-out Query 都会从当前 Fold 的 Reference Corpus 中排除。未知标签仍可出现在检索结果中，但不会参与 Ground Truth 指标以及 success/risk 聚合。

完整 OOF threshold sweep 仅用于探索。报告中的候选工作点使用 nested CV：每个外层 Fold 的参数只通过其 200 道训练题上的内部交叉验证选择。`falseLocalRate` 定义为 `FP / 实际 unsafe 数`。

默认开发档使用 `knn_ood`、`k=10`、`scoreThreshold=1.0`、`weightPower=1` 和 `oodThreshold=0.6`。其 OOF 指标仅用于流程开发；`conservativeRecommended` 继续保留嵌套交叉验证选择的 95% Precision 保守策略。

在仓库根目录运行 `npm run router:runtime` 可启动 Python Runtime；运行 `npm run dev:router` 可同时启动 Runtime、Fastify 和 Vite。公开接口 `POST /api/router/preview` 仅执行影子预测，不会启动模型推理；每次预测写入共享 MongoDB 的 `router_predictions`。可通过 `ROUTER_RUNTIME_URL` 和 `ROUTER_RUNTIME_TIMEOUT_MS` 覆盖 Fastify 到 Python 的连接配置。

生成文件：

- `artifacts/router-v1-offline-results.json`
- `artifacts/router-v1-threshold-sweep.csv`
- `artifacts/router-v1-false-local.jsonl`
- `artifacts/router-v1-non-perfect-analysis.jsonl`

## 安全与持久化

所有写操作前都会 ping MongoDB，并核验数据库名、Questions 数量和 Runs 数量。工具只写 `question_gold_profiles`、`evaluations`、`question_capabilities` 与 `question_embeddings` 和 `router_predictions`，不修改原始 Questions 或 Runs 的 answer/reasoning。API Key 只在内存中使用，不会写入证据。

## 测试

```bash
pytest qwen-evaluator/tests
```
