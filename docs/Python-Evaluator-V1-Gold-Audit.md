# Python Evaluator V1 — Gold Audit

审计日期：2026-09-23。数据来自 `qwen_math_lab` 的 250 道真实 Questions 与三个已完成批次的 750 个唯一 Run；未改写原始 Questions、Run answer 或 reasoning。

## Gold Profile

| 状态 | 数量 |
| --- | ---: |
| total | 250 |
| resolved | 71 |
| multi_part | 150 |
| semantic_required | 1 |
| ambiguous | 28 |
| parse_failed | 0 |

多小问题直接路由 Level 2，不调用 Math-Verify。完整参考解答提取出多个候选表达式时保守标记 `ambiguous`，进入 Level 3，不任意挑选 Gold。

## Phase 5 unresolved 分析

首次 `--max-level 1` 执行后共有 537 条未进入高层：

| 原因 | 数量 |
| --- | ---: |
| multi_part，等待 Level 2 | 450 |
| semantic_required，等待 Level 3 | 3 |
| ambiguous/reference formatting，等待 Level 3 | 84 |
| prediction parse failure | 0 |
| Gold parse failure | 0 |
| Math-Verify timeout | 0 |

Level 1 共 213 条：109 correct、104 incorrect。这里的 unresolved 是路由占位，不等于数学错误。

## 最终三个批次

| 批次 | Level 1 | Level 2 | Level 3 | correct | incorrect | review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Batch 1 | 71 | 150 | 29 | 207 | 39 | 4 |
| Batch 2 | 71 | 150 | 29 | 209 | 39 | 2 |
| Batch 3 (`round3`) | 71 | 150 | 29 | 201 | 47 | 2 |
| All Batches | 213 | 450 | 87 | 617 | 125 | 8 |

最终无 `MAX_LEVEL_REACHED` 或 `JUDGE_FAILED`。8 条低置信度结果按阈值降级为 review；1 条多小问因遗漏必要小问记录为 `INCOMPLETE_SUBQUESTIONS`。

## 人工审计清单

`qwen-evaluator/evaluation-audit-manifest.json` 包含 50 条分层样本，覆盖不同 level、verdict、运行状态及题目元数据。`humanVerdict` 和 `humanNotes` 均为空，等待人工核验，程序没有伪造人工标签。
