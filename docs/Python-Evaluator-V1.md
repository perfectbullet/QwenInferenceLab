# Python Evaluator V1

## 架构

Evaluator 将运行事实与数学判断分离：Runtime Precheck 后由 Gold Profile 决定路由。当前暂不使用 Level 1 Math-Verify；多小问进入 Level 2 LLM Judge，其余可评测答案统一进入 Level 3 语义 Judge。

```text
Run -> Runtime Precheck -> Gold Profile
                         +-> math_verify -> Level 1 -> verdict / Level 3 fallback
                         +-> multi_part  -> Level 2 Judge
                         +-> semantic   -> Level 3 Judge
```

多小问不会被拆分后交给 Math-Verify。运行失败只得到数学 `unresolved`；truncated 如果已有 answer，仍继续数学评测。

## Math-Verify（暂时停用）

依赖仍固定为 `math-verify==0.9.0`，以便保留兼容性和历史证据；当前 Evaluation Pipeline 不调用 `parse()`/`verify()` 判分，原 `math_verify` 路由统一改由 Level 3 语义 Judge 处理。

## LLM Judge

Level 2 与 Level 3 使用不同的版本化 prompt：`multi-part-v1` 和 `judge-v1`。Judge 输入只含 question、reference_answer、candidate_answer，不含候选模型身份、难度、标签、耗时或 token。默认精确查找 `deepseek-ai/DeepSeek-V4-Flash`；配置缺失或不唯一时保留 unresolved。输出使用严格 JSON Schema 和 Pydantic 校验；confidence 低于 0.85 自动降级 review。请求有 timeout、有限重试和单条失败隔离。

评测证据保存 Judge modelName、baseUrl、promptVersion、judgeMode、结构化响应和错误；不保存 API Key。

## MongoDB

读取现有 `questions`、`runs` 和 `model_configs`，新增：

- `question_gold_profiles`，唯一键 `(questionId, adapterVersion)`；
- `evaluations`，唯一键 `(runId, pipelineVersion)`。

默认跳过已有最终 `evaluator-v1`；使用更高 level 时可以续跑 `MAX_LEVEL_REACHED` 占位；`--force` 明确覆盖同一唯一键，不产生重复文档。任何写操作前都会完成 connect、ping、数据库名、questions count 和 runs count 验证。

## CLI

安装与完整命令见 [qwen-evaluator/README-zh.md](../qwen-evaluator/README-zh.md)。核心命令：

```bash
python -m qwen_inference_lab.evaluator.cli audit-gold
python -m qwen_inference_lab.evaluator.cli evaluate-run --run-id <run-id>
python -m qwen_inference_lab.evaluator.cli evaluate --batch-state <state.json>
python -m qwen_inference_lab.evaluator.cli report --batch-state <state.json>
```

`evaluate` 支持重复 `--batch-state`、`--judge-model-config-id`、`--max-level`、`--force` 与 `--concurrency`。并发仅用于 Judge。当前暂不使用 Math-Verify。`report` 输出逐批和 All Batches 统计，并可生成约 50 条人工审计 manifest。

## 版本

- Pipeline：`evaluator-v1`
- Gold Adapter：`gold-v1`
- Math-Verify：`0.9.0`
- Semantic Judge Prompt：`judge-v1`
- Multi-part Prompt：`multi-part-v1`

## 人工 Audit

`evaluation-audit-manifest.json` 是待人工复核的分层样本，不是自动 Gold。审核者检查题目、参考答案、候选答案、路由、解析和 Judge 证据，再填写 `humanVerdict` 与 `humanNotes`。建议重点检查 incorrect、review、跨 level 样本以及不同 math_type/difficulty/tag。

## Known Limitations

- Gold Adapter 采用保守启发式分类，ambiguous 样本需要语义 Judge 或人工确认。
- LLM Judge 仍可能产生一致但错误的判断，必须结合人工 manifest 估计可靠性。
- 当前报告不推导 Router label、localSuccessRate 或路由策略。
- Batch state 不包含 concurrency 字段；已知命令记录为 Batch 1 concurrency=4、Batch 2/3 concurrency=8，但 Evaluator 不据此做因果分析。
