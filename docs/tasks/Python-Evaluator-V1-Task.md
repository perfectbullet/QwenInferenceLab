# QwenInferenceLab — Python Evaluator V1 执行任务书

## 一、任务背景

当前仓库：

`perfectbullet/QwenInferenceLab`

现有系统已经能够：

* 从 MongoDB `questions` 读取数学题及 `reference_answer`
* 调用本地/远程 OpenAI-Compatible LLM
* 将每次推理结果保存到 MongoDB `runs`
* 保存 answer、reasoning、status、finishReason、模型配置、耗时、token 等信息
* 批量运行数学题

目标本地模型：

```text
nvidia/Qwen3.6-35B-A3B-NVFP4
```

当前针对该模型的批量推理工作已经结束。用户确认已执行三个批次，执行命令记录如下：

### Batch 1

```bash
BATCH_CONCURRENCY=4 node --import tsx scripts/batch-test.ts test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4
```

### Batch 2

```bash
BATCH_CONCURRENCY=8 node --import tsx scripts/batch-test.ts test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4-round2
```

### Batch 3

用户提供的第三次执行命令同样记录为：

```bash
BATCH_CONCURRENCY=8 node --import tsx scripts/batch-test.ts test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4-round2
```

注意：Batch 2 与 Batch 3 在用户提供的命令中使用了相同目录名。不要自行假定第三批目录一定叫 `round3`，也不要自行重命名或伪造路径。

在 Phase 0 必须检查实际 `test-results/` 目录、各批次 `state.json` 以及 MongoDB 中对应 Run，确认三个实际批次及其 runId 集合。如果第三批确实复用了 `round2` 目录，则按实际持久化数据处理，并在最终报告中明确说明；如果文件系统中存在独立第三批目录，则以实际目录为准。

本任务的评价范围是：**这三个已经完成的批次中能够通过 batch state / runId 明确归属的最终 Run**。

后续计划是基于这些已完成的运行结果建立本地模型能力数据，并最终用于 Local / Cloud Router。

Router 开发之前，首先需要一个可靠的：

# Python Evaluator V1

其职责是：

> 根据 question、reference_answer 和模型 answer，判断一次模型运行的数学答案是否正确，并保存完整、可重现、可审计的评测证据。

---

# 二、本任务范围

本次只实现：

```text
Python Evaluator V1
```

不要实现 Router。

不要实现：

* Embedding
* KNN
* RouteLLM
* IRT
* Local/Cloud Router
* `/api/route-run`
* Capability Dataset Builder
* localSuccessRate 聚合
* Router Policy
* 前端 Router UI

这些属于后续阶段。

---

# 三、开发环境约束

当前已经为 Evaluator 准备好独立 Python 环境和现有 MongoDB 配置：

```text
Conda Env: qwen-evaluator
Python: 3.11
MongoDB 配置来源: 仓库根目录 .env
```

执行 Python 相关任务前：

```bash
conda activate qwen-evaluator
python --version
```

预期：

```text
Python 3.11.x
```

请直接使用该环境。

不要：

```text
重新创建 Conda 环境
创建 venv
修改系统 Python
修改其他已有 Conda 环境
修改根目录 .env
新建 qwen-evaluator/.env
```

Python 依赖必须统一维护在：

```text
qwen-evaluator/pyproject.toml
```

安装项目使用：

```bash
pip install -e ./qwen-evaluator
```

如果需要开发依赖，可设计：

```bash
pip install -e './qwen-evaluator[dev]'
```

不要仅执行临时 `pip install xxx` 而不更新 `pyproject.toml`。

Python Evaluator 必须复用仓库根目录 `.env` 中现有 MongoDB 配置。任何 MongoDB 写操作前，必须先完成只读连接验证：

```text
1. connect / ping
2. 确认 MONGODB_DATABASE
3. 读取 questions count
4. 读取 runs count
```

确认连接到正确数据库后，才允许创建或写入：

```text
question_gold_profiles
evaluations
```

---

# 四、MongoDB 环境约束

Python Evaluator 必须复用当前 QwenInferenceLab 已有 MongoDB。

不要：

```text
创建新数据库
复制一份数据库
新建独立 Python .env
修改当前 MongoDB 配置
```

仓库根目录已有项目 `.env`。

Python 必须优先读取：

```text
QwenInferenceLab/.env
```

支持项目现有配置：

```text
MONGODB_URI
MONGODB_DATABASE
```

或者：

```text
MONGODB_HOST
MONGODB_PORT
MONGODB_USERNAME
MONGODB_PASSWORD
MONGODB_AUTH_SOURCE
MONGODB_DATABASE
```

Python 行为应尽量与：

```text
server/src/database.ts
```

保持一致。

---

## 4.1 数据库安全要求

任何写操作之前必须：

### 第一步

验证 MongoDB 可以连接。

### 第二步

执行只读查询：

```text
读取 database name
读取 questions count
读取 runs count
```

### 第三步

确认连接的是：

```text
MONGODB_DATABASE
```

指定的现有数据库。

确认无误以后，才允许创建：

```text
question_gold_profiles
evaluations
```

---

## 4.2 Secret

严禁：

```text
日志输出 MongoDB 密码
日志输出 API Key
evaluations 保存 API Key
Gold Profile 保存 API Key
CLI 输出 API Key
```

---

# 五、工作区与现有数据保护

开始工作前：

```bash
git status
```

本地可能存在尚未提交的并发 batch 相关修改和运行结果，必须保留用户已有工作。

必须：

* 不 reset
* 不 checkout 覆盖用户已有修改
* 不 clean
* 不删除 `test-results/`
* 不覆盖无关的 `scripts/batch-test.ts` 修改
* 不修改历史 `runs.answer`
* 不修改历史 `runs.reasoning`
* 不重写现有推理主流程
* 不重新执行或覆盖已经完成的三个 batch，除非用户明确要求

允许阅读仓库中已有 benchmark / test / script 代码用于理解现有数据结构和工程约定，但 **Python Evaluator 的数学判题主逻辑必须以 Math-Verify + 本任务定义的 LLM Judge 流程为准**，不要把旧有启发式字符串判题逻辑作为 Ground Truth 核心。

---

# 六、仓库职责划分

本项目以后保持：

```text
QwenInferenceLab/
│
├── server/          # TypeScript / Fastify / 系统编排
├── web/             # React 前端
├── shared/          # TS 公共类型
├── scripts/         # batch / benchmark / 运维脚本
│
├── qwen-evaluator/              # AI / 算法层
│   ├── Evaluator
│   └── Future Router
│
├── docs/
└── test-results/
```

其中：

```text
server/
```

负责：

```text
API
推理编排
模型调用
SSE
MongoDB 主业务
```

而：

```text
qwen-evaluator/
```

负责：

```text
数学评测
Ground Truth
Embedding（未来）
KNN Router（未来）
OOD（未来）
模型路由算法（未来）
```

---

# 七、Python 工程目录

新增：

```text
QwenInferenceLab/
│
├── qwen-evaluator/
│   ├── pyproject.toml
│   ├── README.md
│   │
│   ├── src/
│   │   └── qwen_inference_lab/
│   │       ├── __init__.py
│   │       │
│   │       ├── common/
│   │       │   ├── __init__.py
│   │       │   ├── config.py
│   │       │   └── mongodb.py
│   │       │
│   │       └── evaluator/
│   │           ├── __init__.py
│   │           ├── models.py
│   │           ├── repository.py
│   │           ├── gold_adapter.py
│   │           ├── math_verify_adapter.py
│   │           ├── multi_part_judge.py
│   │           ├── llm_judge.py
│   │           ├── pipeline.py
│   │           ├── report.py
│   │           └── cli.py
│   │
│   └── tests/
│       └── evaluator/
│           ├── test_math_verify_adapter.py
│           ├── test_gold_adapter.py
│           ├── test_pipeline.py
│           ├── test_repository.py
│           └── test_llm_judge.py
│
└── ...
```

未来 Router 放入：

```text
qwen-evaluator/src/qwen_inference_lab/router/
```

不要创建：

```text
router/
```

作为另一个独立 Python 工程。

---

# 八、设计原则

Evaluator 使用“先分类、再选择判题器”的策略：

```text
Runtime Precheck
        ↓
Gold Adapter / Question Route
        ↓
        ├── 单一、可数学表达的最终答案
        │       ↓
        │   Math-Verify
        │       ↓
        │   correct / incorrect / unresolved
        │
        ├── 多小问 / 多问题
        │       ↓
        │   直接 LLM Judge
        │       ↓
        │   correct / incorrect / review
        │
        └── 证明 / 开放式 / 语义型 / Math-Verify unresolved
                ↓
            LLM Judge
                ↓
            correct / incorrect / review
```

核心约束：

1. **多小问题不得使用 Math-Verify 进行最终判题。**
2. 多小问题应把 `question + reference_answer + candidate_answer` 作为整体交给 Judge 模型判断完整性和正确性。
3. Math-Verify 只负责适合符号/数值等价比较的单答案数学题。
4. Math-Verify parse failure 不等于 incorrect。
5. Runtime failure 不等于 math incorrect。
6. LLM Judge 不确定时返回 review，不要伪造确定性。
7. 不要创建另一套 LaTeX → SymPy 数学解析系统。

不要自行重复实现：

```text
LaTeX → SymPy parser
复杂数学表达式 simplify 比较器
大量手写数学等价规则
```

Math-Verify 承担适用范围内的数学表达式解析和等价验证。

---

# 九、Evaluator 总体 Pipeline

实现：

```text
Run
 │
 ▼
Level 0: Runtime Precheck
 │
 ▼
Gold Adapter / Evaluation Route
 │
 ├────────────── multi_part ──────────────┐
 │                                        ▼
 │                               Level 2: Multi-part
 │                               Direct LLM Judge
 │                                        │
 │                                        ▼
 │                         correct / incorrect / review
 │
 ├──────────── semantic / proof ──────────┐
 │                                        ▼
 │                               Level 3: Semantic
 │                               Direct LLM Judge
 │                                        │
 │                                        ▼
 │                         correct / incorrect / review
 │
 └──────── single_math ────────┐
                               ▼
                      Level 1: Math-Verify
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
               correct      incorrect    unresolved
                                             │
                                             ▼
                                  Level 3: LLM Judge
```

核心原则：

```text
parse failure != incorrect
timeout != incorrect
runtime failure != math incorrect
unknown != incorrect
multi_part != Math-Verify
```

Level 2 与 Level 3 都可以使用同一个外部 Judge 模型，但必须使用不同、版本化的 prompt，并在 evaluation evidence 中保存 `judgeMode`。

---

# 十、Python 依赖

`qwen-evaluator/pyproject.toml` 至少包含：

```text
Python >= 3.10
```

本环境实际使用：

```text
Python 3.11
```

依赖：

```text
math-verify==0.9.0
math-verify[antlr4_13_2]
pymongo
python-dotenv
pydantic>=2
openai
typer
```

开发：

```text
pytest
pytest-mock
```

如果 extras 与版本固定需要调整写法，可以根据 Python Packaging 规范实现，但必须保证：

```text
Math-Verify 明确固定版本
ANTLR runtime 明确固定版本
```

不要依赖：

```text
latest
```

不要 vendor Math-Verify 源码。

---

# 十一、先阅读 Math-Verify

实现前重点阅读官方：

```text
README.md

src/math_verify/parser.py
src/math_verify/grader.py
src/math_verify/metric.py

tests/test_configs.py
tests/test_timeout.py
tests/test_all.py
```

重点确认：

```text
parse()

verify()

LatexExtractionConfig

ExprExtractionConfig

StringExtractionConfig

fallback_mode

extraction_mode

parsing_timeout

strict

float_rounding

numeric_precision

allow_set_relation_comp

timeout_seconds
```

特别注意：

```python
verify(gold, prediction)
```

参数顺序不可反。

---

# 十二、不要直接使用 math_metric() 作为核心 API

Math-Verify 官方：

```python
math_metric()
```

适合 Benchmark。

但本项目 Evaluator 必须能够区分：

```text
真正答案错误
prediction parse failure
gold parse failure
timeout
unsupported format
internal error
```

所以核心应直接使用：

```python
parse()
verify()
```

并保留中间 evidence。

---

# 十三、MongoDB Collection

现有：

```text
questions
runs
model_configs
model_settings
```

新增：

```text
question_gold_profiles
evaluations
```

不要修改：

```text
questions 原始字段
runs.answer
runs.reasoning
```

---

# 十四、question_gold_profiles

目的：

> 同一道题的 reference_answer 是固定的，不能每一次模型 Run 都重新分析一遍。

建议结构：

```json
{
  "questionId": "MATH-001",

  "adapterVersion": "gold-v1",
  "mathVerifyVersion": "0.9.0",

  "referenceHash": "...",

  "questionType": "single_expression",

  "evaluationRoute": "math_verify",

  "status": "resolved",

  "rawReference": "...",

  "goldCandidates": [
    "1/2"
  ],

  "parse": {
    "success": true,
    "serialized": [
      "1/2"
    ]
  },

  "reasonCode": "DIRECT_MATH_GOLD",

  "createdAt": "...",
  "updatedAt": "..."
}
```

状态建议：

```text
resolved

multi_part

semantic_required

ambiguous

parse_failed
```

并建议明确保存：

```text
evaluationRoute = math_verify | llm_multi_part | llm_semantic | unresolved
```

其中：

```text
multi_part → llm_multi_part
semantic_required / proof → llm_semantic
单一可验证数学答案 → math_verify
```

唯一索引：

```text
questionId + adapterVersion
```

---

# 十五、Gold Adapter

这是整个 Evaluator V1 最重要的项目适配层之一。

因为：

```text
questions.reference_answer
```

可能包含：

```text
完整解题过程
中间公式
多个小问
最终结论
证明过程
```

不能默认：

```text
reference_answer == clean gold
```

---

## 15.1 先 Audit，不要先写几十条规则

首先实现：

```bash
python -m qwen_inference_lab.evaluator.cli audit-gold
```

扫描当前 MongoDB：

```text
questions
```

分析：

```text
question
reference_answer
```

输出真实统计：

```text
total

resolved

multi_part

semantic_required

ambiguous

parse_failed
```

数字必须来自当前数据库。

不得提前假设。

---

## 15.2 多问题识别后的处理

Gold Adapter 一旦可靠识别：

```text
questionType = multi_part
```

必须设置：

```text
evaluationRoute = llm_multi_part
```

这类题 **不要尝试把多个小问拆成若干 Math-Verify 表达式再汇总判分**。

后续直接进入 Level 2 Multi-part LLM Judge，由模型整体比较：

```text
question
reference_answer
candidate_answer
```

重点判断：

```text
是否回答全部小问
每个小问结论是否正确
是否存在关键推理错误
是否有遗漏条件
整体答案是否可判定为正确
```

---

## 15.3 Gold Adapter 必须保守

优先：

```text
Precision > Coverage
```

无法确定：

```text
ambiguous
```

需要自然语言/证明判断：

```text
semantic_required
```

不要猜。

---

## 15.4 禁止自行解题生成 Gold

Gold Adapter 不允许：

```text
自己计算题目答案
```

唯一正确来源：

```text
reference_answer
```

如果 Reference 本身无法可靠提取：

```text
review
或
semantic_required
```

---

# 十六、Level 0 — Runtime Precheck

读取：

```text
status
finishReason
answer
error
```

运行状态和数学能力分开。

例如：

```text
failed
interrupted
cancelled
```

结果：

```text
runtime status = 原状态
math verdict = unresolved
```

不要：

```text
failed → incorrect
```

---

## truncated

允许继续评价现有：

```text
answer
```

因此：

```text
runtimeStatus = truncated
mathVerdict = correct
```

是合法组合。

Evaluator 只保存事实。

本任务：

```text
不要决定 truncated 在 Router 中算不算 LocalSuccess。
```

---

# 十七、Level 1 — Direct Math-Verify

核心：

```text
math_verify_adapter.py
```

职责：

```text
parse gold

parse prediction

verify

保存 evidence
```

除此之外不要塞业务逻辑。

---

# 十八、Prediction Parsing

优先：

```python
LatexExtractionConfig(
    boxed_match_priority=0
)
```

同时支持：

```python
ExprExtractionConfig()
```

因为模型当前应该尽量输出：

```latex
\boxed{}
```

但不要自己写 boxed parser。

---

## 18.1 避免 Parser Fallback 掩盖失败

优先研究并合理使用：

```text
fallback_mode="no_fallback"
```

目的：

```text
真正 parse failed
```

不能由于字符串 fallback 变成：

```text
貌似 parse success
```

保存：

```text
parseSuccess

parsedValues

parseError

parseTimeout
```

---

# 十九、verify 配置

初始建议：

```python
verify(
    gold,
    prediction,
    strict=True,
    float_rounding=6,
    timeout_seconds=5
)
```

如果实际 API 参数不同，以当前 Math-Verify 0.9.0 官方实现为准。

不要为了“提高准确率”直接：

```text
strict=False
```

如果需要放宽：

必须有：

```text
真实案例
自动测试
文档说明
```

并在 Evaluation Evidence 中保存当时配置。

---

# 二十、Level 1 Verdict

Level 1 返回：

```text
correct
incorrect
unresolved
```

---

## correct

要求：

```text
gold parse 成功

prediction parse 成功

verify 正常结束

verify == true
```

---

## incorrect

只有：

```text
Gold Profile 已可靠 resolved

Prediction 成功解析

verify 正常结束

verify == false
```

而且不存在当前已知 extraction ambiguity。

---

## unresolved

包括：

```text
gold parse failed

prediction parse failed

parse timeout

verify timeout

Math-Verify exception

Gold ambiguous

Gold semantic_required
```

不要把这些情况映射成：

```text
incorrect
```

---

# 二十一、Level 2 — Multi-part Direct LLM Judge

Level 2 专门处理：

```text
多小问
多问题
需要整体检查完整性的综合题
```

核心规则：

# 多问题不使用 Math-Verify 进行最终判题。

一旦 Gold Adapter / Question Route 判定为：

```text
evaluationRoute = llm_multi_part
```

直接调用 Judge 模型。

不要：

```text
LLM 先拆小问
↓
Math-Verify 分别判断
↓
再汇总
```

本任务明确取消这条路线。

---

# 二十二、Multi-part LLM Judge

输入只包含：

```text
question
reference_answer
candidate_answer
```

Judge 必须直接完成：

```text
1. 识别题目包含多少个小问
2. 检查 candidate 是否覆盖全部小问
3. 分别核对各小问答案和关键推理
4. 检查是否存在关键性数学错误
5. 给出整体 verdict
```

输出严格 JSON，例如：

```json
{
  "verdict": "correct",
  "complete": true,
  "allSubquestionsAnswered": true,
  "hasCriticalError": false,
  "subquestions": [
    {
      "id": "1",
      "answered": true,
      "correct": true,
      "reason": "..."
    },
    {
      "id": "2",
      "answered": true,
      "correct": true,
      "reason": "..."
    }
  ],
  "confidence": 0.95,
  "reason": "...",
  "issues": []
}
```

允许：

```text
correct
incorrect
review
```

如果遗漏任意必要小问，应优先：

```text
verdict = incorrect
complete = false
```

并保存类似：

```text
reasonCode = INCOMPLETE_SUBQUESTIONS
```

如果题目/参考答案本身存在歧义，或 Judge 无法可靠判定：

```text
verdict = review
```

---

# 二十三、Level 3 — Semantic LLM Judge

Level 3 用于：

```text
证明题
开放式推导
几何论证
自然语言数学结论
Gold Adapter 标记 semantic_required
单答案 Math-Verify parse/verify 无法可靠完成的 unresolved 样本
```

Level 3 同样直接使用 Judge 模型，不再强制返回 Math-Verify。

对于单答案题，只有 Math-Verify 得到 `unresolved` 时才升级到 Level 3；Math-Verify 已可靠得到 `incorrect` 时不需要为了“翻案”再调用 LLM。

---

# 二十四、LLM Judge 输入限制

无论 Level 2 还是 Level 3，Judge 输入只允许包含：

```text
question
reference_answer
candidate_answer
```

不得向 Judge 提供：

```text
candidate model name
Qwen / DeepSeek 身份
difficulty
tag
math_type
GPU
耗时
tokens
candidate baseUrl
candidate modelConfigId
```

避免模型身份及运行指标影响判题。

注意：Judge 自身使用哪个模型可以记录在 evaluation metadata 中，但不能在判题 prompt 中告诉 Judge 候选答案来自哪个模型。

---

# 二十五、LLM Judge 输出与置信度

输出必须经过 Pydantic / JSON Schema 严格校验。

通用结构至少包括：

```json
{
  "verdict": "correct",
  "complete": true,
  "hasCriticalError": false,
  "confidence": 0.94,
  "reason": "...",
  "issues": []
}
```

Level 2 可以额外包含：

```text
allSubquestionsAnswered
subquestions[]
```

允许 verdict：

```text
correct
incorrect
review
```

建议：

```text
confidence < 0.85
```

自动降级为：

```text
review
```

阈值集中配置，不要散落硬编码。

---

# 二十六、默认 Judge Model

如果 Evaluator 需要调用外部模型比较模型输出与参考答案，默认从 MongoDB：

```text
model_configs
```

查找：

```text
modelName = "deepseek-ai/DeepSeek-V4-Flash"
```

使用该记录中的：

```text
baseUrl
modelName
apiKey
```

作为 Level 2 / Level 3 Judge。

规则：

1. 精确匹配 `modelName`。
2. 若恰好找到一条，直接作为默认 Judge。
3. 若找不到，Level 2 / Level 3 返回 `unresolved/review` 并明确报告缺少 Judge 配置，不得随意选择其他模型。
4. 若找到多条同名配置，不得随机选择；优先要求 CLI `--judge-model-config-id` 明确指定，并在报告中说明。
5. CLI `--judge-model-config-id` 作为显式 override，其优先级高于默认 modelName 查找。
6. API Key 只存在运行时内存中，不得输出或写入 evaluation。

---

# 二十七、Judge 调用方式

Judge 必须通过 `model_configs` 中配置的 OpenAI-Compatible endpoint 调用。

不得：

```text
硬编码 DeepSeek URL
硬编码 API Key
从代码中写死凭据
随意调用其他外部模型
```

必须实现：

```text
请求 timeout
有限重试
结构化输出校验
非法 JSON fallback
低 confidence → review
网络失败 → unresolved/review
```

不要因为某一条 Judge 请求失败导致整个批量 Evaluator 崩溃。

---

# 二十八、无可用 Judge 时的行为

若默认：

```text
deepseek-ai/DeepSeek-V4-Flash
```

配置不存在，且用户未提供 `--judge-model-config-id`：

```text
Level 1 Math-Verify 正常工作
multi_part → unresolved/review
semantic_required → unresolved/review
Math-Verify unresolved → unresolved/review
```

程序必须继续处理其他记录，并在最终 report 中统计：

```text
judge_unavailable
```

不得自动替换成其他模型。

---

# 二十九、evaluations Collection

每次机器评测写入：

```text
evaluations
```

不要修改：

```text
runs.answer

runs.reasoning
```

第一版也：

```text
不要自动覆盖 runs.evaluation
```

保持：

```text
runs
=
推理原始事实 + 当前人工评价

evaluations
=
机器评测事实
```

---

# 三十、Evaluation 数据结构

建议：

```json
{
  "id": "...",

  "runId": "...",
  "questionId": "...",

  "pipelineVersion": "evaluator-v1",
  "goldAdapterVersion": "gold-v1",
  "mathVerifyVersion": "0.9.0",

  "runtime": {
    "status": "completed",
    "finishReason": "stop"
  },

  "math": {
    "verdict": "correct",
    "level": 1,
    "method": "math_verify",
    "confidence": 1.0,
    "reasonCode": "SYMBOLIC_EQUIVALENT"
  },

  "extraction": {
    "goldParseSuccess": true,
    "predictionParseSuccess": true,

    "goldParsed": [
      "1/2"
    ],

    "predictionParsed": [
      "1/2"
    ],

    "error": null
  },

  "verifyConfig": {
    "strict": true,
    "floatRounding": 6,
    "timeoutSeconds": 5
  },

  "judge": null,

  "createdAt": "...",
  "updatedAt": "..."
}
```

---

# 三十一、Level 2 / Level 3 Evidence

如果使用 Judge，保存：

```text
judge modelName

judge baseUrl

promptVersion

structured response

confidence

reason
```

不要保存：

```text
apiKey
```

---

# 三十二、版本常量

集中管理：

```text
PIPELINE_VERSION = "evaluator-v1"

GOLD_ADAPTER_VERSION = "gold-v1"

LLM_JUDGE_PROMPT_VERSION = "judge-v1"

MULTI_PART_JUDGE_PROMPT_VERSION = "multi-part-v1"
```

不要散落硬编码。

---

# 三十三、Evaluator 不生成 Router Label

禁止：

```text
localSuccessRate

P(local)

RouterLabel

routeToLocal

routeToCloud
```

Evaluator 只产生：

```text
Runtime Facts

Math Verdict

Evidence
```

后续：

```text
Capability Dataset Builder
```

再决定如何把：

```text
failed

truncated

interrupted

cancelled

math correct
```

组合成 Router Ground Truth。

---

# 三十四、幂等性

索引：

```text
evaluations
→ unique(runId, pipelineVersion)
```

```text
question_gold_profiles
→ unique(questionId, adapterVersion)
```

默认：

```text
已有 evaluator-v1
→ skip
```

支持：

```bash
--force
```

进行重新评价。

不能重复执行一次就产生一批重复文档。

---

# 三十五、CLI

至少实现：

```text
audit-gold

evaluate-run

evaluate

report
```

---

## 35.1 audit-gold

```bash
python -m qwen_inference_lab.evaluator.cli audit-gold
```

功能：

```text
读取全部 questions

构建 Gold Profile

写入 question_gold_profiles

输出统计
```

---

## 35.2 evaluate-run

```bash
python -m qwen_inference_lab.evaluator.cli evaluate-run \
  --run-id <run-id>
```

用于：

```text
单条调试
人工复核
测试 Parser
```

支持显式覆盖：

```bash
--judge-model-config-id <id>
```

若未提供该参数，则自动在 `model_configs` 中精确查找：

```text
modelName = deepseek-ai/DeepSeek-V4-Flash
```

---

## 35.3 evaluate

必须至少支持单个 `--batch-state`，建议实现为可重复参数，以便一次处理三个已完成批次。

单批示例：

```bash
python -m qwen_inference_lab.evaluator.cli evaluate \
  --batch-state \
  ../test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4/state.json
```

若实际确认了三个独立 state 文件，可支持：

```bash
python -m qwen_inference_lab.evaluator.cli evaluate \
  --batch-state <batch1-state.json> \
  --batch-state <batch2-state.json> \
  --batch-state <batch3-state.json>
```

支持：

```text
--judge-model-config-id
--force
--max-level 1
--max-level 2
--max-level 3
```

默认：

```text
max-level = 3
```

未显式提供 Judge Model 时，应自动查找：

```text
modelName = deepseek-ai/DeepSeek-V4-Flash
```

仍不可用时不能让整个任务报错。

---

## 35.4 report

单批示例：

```bash
python -m qwen_inference_lab.evaluator.cli report \
  --batch-state \
  ../test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4/state.json
```

对三个已完成批次，report 应支持逐批统计和汇总统计。

输出至少包含：

```text
Batch 1
Batch 2
Batch 3（若可可靠识别）
All Batches

Runtime
--------------------------------
completed                       xxx
truncated                       xxx
failed                          xxx
cancelled                       xxx
interrupted                     xxx

Evaluation
--------------------------------
Level 1 Math-Verify             xxx
Level 2 Multi-part LLM Judge    xxx
Level 3 Semantic LLM Judge      xxx
Review                          xxx
Unresolved                      xxx

Math Verdict
--------------------------------
correct                         xxx
incorrect                       xxx
review                          xxx
unresolved                      xxx
```

不得伪造统计。

---

# 三十六、三个已完成 Batch 的识别与评价范围

当前三个批次均已执行完成。

用户提供的命令记录：

```bash
# Batch 1
BATCH_CONCURRENCY=4 node --import tsx scripts/batch-test.ts   test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4

# Batch 2
BATCH_CONCURRENCY=8 node --import tsx scripts/batch-test.ts   test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4-round2

# Batch 3（用户提供的记录与 Batch 2 使用相同目录名）
BATCH_CONCURRENCY=8 node --import tsx scripts/batch-test.ts   test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4-round2
```

必须首先查看本地真实：

```text
test-results/
各目录 state.json
worker log（如有）
MongoDB runs
```

确认三个实际批次。

## Batch 识别原则

优先依据：

```text
state.json 中明确记录的 runId
```

以及能够与 MongoDB Run 唯一对应的 batch state 信息。

不要仅依赖：

```text
startedAt
modelName
日期范围
```

猜测批次归属。

因为用户提供的 Batch 2 / Batch 3 路径相同，必须特别处理：

* 如果实际存在第三个独立目录，以实际目录为准；
* 如果第三次确实复用了同一个 `round2` 目录，检查 state 是否包含第三次 Run 的明确记录；
* 不要擅自把第三批命名为 `round3`；
* 不要人为制造不存在的 runId；
* 如果历史 state 已被覆盖而无法可靠恢复第三批归属，应明确报告该限制，而不是用时间窗口猜测；
* 只评价能够可靠归属到三个批次的 Run。

三个批次已经完成，因此本任务不再使用“只评价当前已完成部分”的增量逻辑。

Evaluator report 应同时提供：

```text
Batch 1 独立统计
Batch 2 独立统计
Batch 3 独立统计（如果可可靠识别）
All Batches 汇总统计
```

注意 Batch 1 concurrency=4，后续批次 concurrency=8。Evaluator 不负责分析并发度对模型能力的因果影响，但 report 必须保留 batch/concurrency 元数据，不能把它丢失。

---

# 三十七、测试

必须提供自动测试。

---

## 37.1 Math equivalence

覆盖：

```text
1/2 == 0.5

(x+1)^2 == x^2 + 2x + 1

集合等价

区间

方程

不等式

明显错误表达式
```

---

## 37.2 verify 参数方向

必须测试 wrapper 永远：

```python
verify(gold, prediction)
```

禁止反过来。

---

## 37.3 Parse failure

测试：

```text
prediction parse failed
```

结果必须：

```text
unresolved
```

不能：

```text
incorrect
```

---

## 37.4 Runtime failure

测试：

```text
status = failed
```

不得变成：

```text
math incorrect
```

---

## 37.5 Truncated

```text
status = truncated
```

仍然尝试评价已有：

```text
answer
```

---

## 37.6 Multi-part

必须覆盖：

```text
Reference = 2 个小问
Candidate = 只回答 1 个
```

使用 mock Judge，结果：

```text
incorrect
reasonCode = INCOMPLETE_SUBQUESTIONS
```

并增加关键测试：

```text
questionType = multi_part
→ Math-Verify adapter 不得被调用
→ 直接调用 Multi-part LLM Judge
```

这是本版本的强制验收条件。

---

## 37.7 LLM Judge

必须使用 mock client。

覆盖：

```text
正常 JSON

非法 JSON

请求失败

timeout

低 confidence

review
```

Unit Test 不允许真的访问 DeepSeek / OpenAI / 其他云端模型。

---

## 37.8 Mongo Repository

Repository 和 Pipeline 必须解耦。

核心：

```text
Gold Adapter tests

Math Verify tests

Pipeline tests
```

不得依赖真实 MongoDB。

Mongo integration test 单独处理。

---

# 三十八、Evaluator Gold Set

完成基础 Pipeline 后，从真实 R1 数据构建：

```text
约 50 条
```

人工审计 manifest。

不是纯随机。

分层覆盖：

```text
Level 1 correct

Level 1 incorrect

Level 2 Multi-part LLM Judge

Level 3 Semantic LLM Judge

review

truncated

不同 math_type

不同 difficulty

不同 tag
```

生成：

```text
qwen-evaluator/evaluation-audit-manifest.json
```

只准备待人工核验样本。

不要自动给这些样本写“人工 Gold”。

---

# 三十九、执行阶段

Codex 必须按阶段执行。

---

## Phase 0 — Inspect

首先：

```bash
git status
```

然后：

1. 查看本地未提交修改
2. 确认当前 Conda 环境
3. 查看根目录 `.env` 的字段名，但不要输出 secret
4. 查看 `server/src/database.ts`
5. 查看 Questions / Runs / ModelConfig 数据结构
6. 枚举 `test-results/` 并确认三个已完成 Batch 的真实目录、state schema 和 runId
7. 特别核实用户提供的 Batch 2 / Batch 3 同名目录问题
8. 阅读 Math-Verify 官方实现
9. 检查 `model_configs` 中 `modelName=deepseek-ai/DeepSeek-V4-Flash` 的配置是否唯一可用，但不要输出 API Key

---

## Phase 1 — Environment & Skeleton

确认：

```bash
conda activate qwen-evaluator
python --version
```

必须是：

```text
Python 3.11.x
```

然后创建：

```text
qwen-evaluator/pyproject.toml

qwen-evaluator/src/...

qwen-evaluator/tests/...
```

执行：

```bash
pip install -e ./qwen-evaluator
```

测试：

```bash
python -m qwen_inference_lab.evaluator.cli --help
pytest qwen-evaluator/tests
```

---

# 四十、Phase 2 — MongoDB Read-Only Validation

在任何 DB 写操作前：

```text
connect
↓
ping
↓
读取 database
↓
questions.count
↓
runs.count
```

输出：

```text
Connected database: <database>

Questions: xxx
Runs: xxx
```

不得显示任何 secret。

确认数据库正确后才能继续。

---

# 四十一、Phase 3 — Gold Audit

实现：

```text
Gold Adapter
question_gold_profiles
audit-gold
```

执行：

```bash
python -m qwen_inference_lab.evaluator.cli audit-gold
```

对实际题库扫描。

生成：

```text
docs/Python-Evaluator-V1-Gold-Audit.md
```

必须写真实结果：

```text
实际题数

resolved

multi_part

semantic_required

ambiguous

parse_failed

典型失败类型
```

不得编造。

---

# 四十二、Phase 4 — Level 1

实现：

```text
Math-Verify adapter

parse evidence

verify evidence

timeout

error categorization
```

先在 Batch 1 上完成 Level 1 验证，然后对三个已完成 Batch 中适合 Math-Verify 的单答案题执行 Level 1。

输出：

```text
correct

incorrect

unresolved
```

真实统计。

---

# 四十三、Phase 5 — Analyze Unresolved

在实现大量 Level 2 逻辑前：

必须先分析真实：

```text
Level 1 unresolved
```

将原因分类：

```text
multi_part

proof

reference formatting

prediction formatting

missing subquestion

semantic answer

Math-Verify unsupported

timeout

其他
```

更新：

```text
docs/Python-Evaluator-V1-Gold-Audit.md
```

---

# 四十四、Phase 6 — Level 2 Multi-part LLM Judge

根据 Gold Audit / Question Route 中真实识别出的：

```text
multi_part
```

样本实现 Level 2。

明确：

```text
multi_part
↓
直接 DeepSeek-V4-Flash Judge
↓
correct / incorrect / review
```

不要再设计：

```text
Structured Extraction → Math-Verify
```

路线。

默认 Judge：

```text
model_configs.modelName = deepseek-ai/DeepSeek-V4-Flash
```

先确认配置唯一并可调用，再对多问题进行评价。

重点验证：

```text
所有小问是否回答
各小问结论是否正确
关键推理是否有错误
是否遗漏必要条件
```

---

# 四十五、Phase 7 — Level 3 Semantic LLM Judge

实现证明题、开放题和 Level 1 unresolved 的语义 Judge。

同样默认使用：

```text
model_configs.modelName = deepseek-ai/DeepSeek-V4-Flash
```

实现：

```text
Structured Output validation
confidence threshold
review fallback
timeout / retry
单条失败隔离
```

如果 Judge 配置不存在或调用失败：

```text
不得自动换模型
不得整批失败
对应样本保留 review / unresolved
```

单元测试必须继续使用 mock，不允许测试阶段真实消耗外部 API。

集成执行阶段可以使用 MongoDB 中已有 DeepSeek-V4-Flash 配置进行真实 Judge。

---

---

# 四十六、Phase 8 — Report & Audit

完成：

```text
audit-gold

evaluate-run

evaluate

report
```

并生成：

```text
evaluation-audit-manifest.json
```

---

# 四十七、代码质量

要求：

```text
完整 Python 类型注解

Pydantic Models

小函数

清晰 Module Boundary

结构化 logging

异常分类

Repository / Domain 分离

无 secret logging

Idempotent DB Writes
```

不要：

```text
一个超大 evaluator.py

大量 catch Exception: pass

静默吞掉 Parser Error

全局 MongoClient

魔法常量散落

重复 Math-Verify 核心逻辑
```

---

# 四十八、文档

新增：

```text
docs/Python-Evaluator-V1.md
```

至少包含：

```text
Evaluator 架构

Runtime Precheck

Gold Adapter

三级判题

Math-Verify 的职责

Level 2 Multi-part LLM Judge

Level 3 LLM Judge

MongoDB Collections

CLI

运行方式

Judge Model 配置

版本化

Known Limitations

人工 Audit 流程
```

---

同时：

```text
docs/Python-Evaluator-V1-Gold-Audit.md
```

必须来源于真实数据。

---

# 四十九、验收条件

## 工程

必须：

```text
pip install -e ./qwen-evaluator
```

成功。

```text
pytest qwen-evaluator/tests
```

成功。

```text
python -m qwen_inference_lab.evaluator.cli --help
```

成功。

---

## MongoDB

必须：

```text
正确读取现有数据库

不泄露 secret

不修改原始 question 内容

不修改原始 run answer/reasoning
```

---

## Gold Audit

当前所有 Questions 都必须有明确状态：

```text
resolved

multi_part

semantic_required

ambiguous

parse_failed
```

不能静默跳过。

---

## Evaluator

必须：

```text
parse failure != incorrect

runtime failure != math incorrect

truncated 可以继续 Math Evaluation

Math-Verify correct 保存 Evidence

Math-Verify mismatch 保存 Evidence

多问题绕过 Math-Verify 并由 Level 2 LLM Judge 直接判断

Level 3 无把握 → review
```

---

## 幂等

同一个：

```text
runId + pipelineVersion
```

重复执行：

```text
不会无限新增 Evaluation。
```

---

## 可审计

任意一条 Evaluation 必须能够回答：

```text
是谁评的？

哪个 pipeline version？

哪个 Math-Verify version？

使用哪个 Gold Profile？

进入哪一级？

抽出了什么 Gold？

抽出了什么 Prediction？

Math-Verify 怎么判断？

有没有调用 LLM？

为什么最后是 correct / incorrect / review / unresolved？
```

---

# 五十、明确不属于本任务的内容

不要顺手实现：

```text
Capability Profile

3-run aggregation

local pass rate

Embedding

KNN

OOD

Router threshold

Local Precision

Local Coverage

/api/route-run

Cloud fallback
```

Evaluator V1 完成后停止。

等待下一阶段任务。

---

# 五十一、最终汇报要求

任务结束不要只说：

```text
Done
```

必须输出：

## 1. Git 状态

说明：

```text
任务开始前已有修改

本次新增/修改文件
```

---

## 2. Python 工程

列出：

```text
qwen-evaluator/
```

新增目录和主要模块。

---

## 3. Database

说明：

```text
实际连接数据库

questions count

runs count

新增 indexes / collections
```

不要输出 secret。

---

## 4. Gold Audit

报告真实：

```text
总题数

resolved

multi_part

semantic_required

ambiguous

parse_failed
```

---

## 5. Evaluation

三个 Batch 已经执行完成。

必须分别报告并给出汇总：

```text
Batch 1 Runs Evaluated
Batch 2 Runs Evaluated
Batch 3 Runs Evaluated（若可可靠识别）
All Batches Runs Evaluated

Level 1 Math-Verify
Level 2 Multi-part LLM Judge
Level 3 Semantic LLM Judge

correct
incorrect
review
unresolved
```

同时报告三个批次实际识别到的：

```text
batch path
concurrency
runId count
missing / duplicate runId
```

如果第三批因为与 Batch 2 复用同一个 state 目录而无法可靠恢复独立 runId 集合，必须明确说明证据和限制，不得用时间范围或模型名猜测。

---

## 6. Tests

报告真实：

```text
pytest 数量

pass

fail

skip
```

不得说：

```text
应该能通过
```

必须实际运行。

---

## 7. Known Issues

明确列出尚未解决问题。

---

## 8. 下一步

只给：

```text
Evaluator V1 收尾
或
Evaluator Audit
```

相关建议。

不要自行启动 Router 开发。

---

# 五十二、核心原则

整个任务始终遵循：

```text
Ground Truth 质量
>
自动化覆盖率
```

以及：

```text
宁可 unresolved
不要误判 correct

宁可 review
不要伪造确定性

Math-Verify 能处理单答案数学表达式
不要重复实现

多问题
直接使用 LLM Judge，不走 Math-Verify

默认外部 Judge
使用 model_configs 中 deepseek-ai/DeepSeek-V4-Flash

解析失败
不等于答案错误

运行失败
不等于数学错误

Evaluator 保存事实
Router 后续决定策略
```

---

# 五十三、开始执行

不要只分析方案。

按照：

```text
Phase 0
↓
Phase 1
↓
Phase 2
↓
Phase 3
↓
Phase 4
↓
Phase 5
↓
Phase 6
↓
Phase 7
↓
Phase 8
```

实际修改仓库、运行测试、检查真实数据，并最终给出完整执行报告。

如果某一个阶段发现现有真实数据与本任务书假设不一致：

1. 不要强行套用假设；
2. 先根据实际仓库和数据修正实现；
3. 保持本文档定义的核心边界；
4. 在最终报告中说明偏差和处理方式。
