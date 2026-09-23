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


## Capability Dataset Builder V1

Build 250 question-level capability records from the three exact batches:

```bash
python -m qwen_inference_lab.capability.cli build \
  --batch-state ../test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4/state.json \
  --batch-state ../test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4-round2/state.json \
  --batch-state ../test-results/batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4-round3/state.json
```

Existing `capability-v1` records are skipped by default; pass `--force` for an idempotent update. The command writes MongoDB `question_capabilities` and exports:

- `artifacts/capability-dataset-v1.jsonl`
- `artifacts/capability-dataset-v1.csv`

Report persisted records with:

```bash
python -m qwen_inference_lab.capability.cli report
```

The dataset retains Question text and structured attempt metrics, but excludes `reference_answer`, full model answers, and reasoning.

## Safety and persistence

Every command validates MongoDB with ping/database name/question count/run count before writes. Tool data is stored only in `question_gold_profiles`, `evaluations`, and `question_capabilities`; original Questions and Run answers/reasoning are not modified. API keys remain in memory and are never persisted in evidence.

## Tests

```bash
pytest qwen-evaluator/tests
```
