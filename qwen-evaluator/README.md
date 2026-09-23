# QwenInferenceLab Python Evaluator

English | [中文](README-zh.md)

`qwen-evaluator/` provides auditable mathematical evaluation for QwenInferenceLab. It reuses the repository-root `.env` and the existing MongoDB database.

## Install

Run from the repository root:

```bash
conda activate qwen-evaluator
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple pip install -e "./qwen-evaluator[dev]"
```

Python 3.11 is required. Math-Verify is pinned to 0.9.0 and the ANTLR runtime to 4.13.2.

## Commands

```bash
python -m qwen_inference_lab.evaluator.cli --help
python -m qwen_inference_lab.evaluator.cli audit-gold
python -m qwen_inference_lab.evaluator.cli evaluate-run --run-id <run-id>
python -m qwen_inference_lab.evaluator.cli evaluate --batch-state <state.json>
python -m qwen_inference_lab.evaluator.cli report --batch-state <state.json>
```

Repeat `--batch-state` to process several batches. `evaluate` supports `--max-level 1|2|3`, `--force`, `--judge-model-config-id`, and bounded `--concurrency`. Math-Verify is temporarily disabled: multi-part questions use the Level 2 Judge and all other evaluable answers use the Level 3 semantic Judge. `--judge-timeout` and `--judge-retries` control per-request handling. Normal resume retries `JUDGE_FAILED`, `JUDGE_UNAVAILABLE`, and `MAX_LEVEL_REACHED` without overwriting final verdicts.

Generate a stratified manifest for human review:

```bash
python -m qwen_inference_lab.evaluator.cli report \
  --batch-state <state.json> \
  --audit-manifest qwen-evaluator/evaluation-audit-manifest.json \
  --audit-size 50
```

The manifest deliberately leaves `humanVerdict` and `humanNotes` empty. Reviewers fill them manually.

## Safety and persistence

Every command validates MongoDB with ping/database name/question count/run count before writes. Evaluator data is stored only in `question_gold_profiles` and `evaluations`; original Questions and Run answers/reasoning are not modified. API keys remain in memory and are never persisted in evidence.

## Tests

```bash
pytest qwen-evaluator/tests
```
