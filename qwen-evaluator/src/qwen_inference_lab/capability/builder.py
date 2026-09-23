"""Build validated question-level capability records from exact batch states."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from qwen_inference_lab.evaluator import PIPELINE_VERSION

from . import CAPABILITY_VERSION
from .models import AttemptDetail, CapabilityRecord, RuntimeCounts

_BATCH_METADATA = {
    "batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4": ("round1", 4),
    "batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4-round2": ("round2", 8),
    "batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4-round3": ("round3", 8),
}
_RUNTIME_STATUSES = ("completed", "truncated", "failed", "cancelled", "interrupted")
_MATH_VERDICTS = ("correct", "incorrect", "review", "unresolved")


@dataclass(frozen=True)
class BatchSpec:
    path: Path
    label: str
    concurrency: int
    run_ids: tuple[str, ...]


def load_batch_states(paths: list[Path], *, expected_batch_size: int = 250) -> list[BatchSpec]:
    if not paths:
        raise ValueError("At least one --batch-state is required")
    specs: list[BatchSpec] = []
    all_ids: list[str] = []
    seen_names: set[str] = set()
    for path in paths:
        name = path.parent.name
        if name not in _BATCH_METADATA:
            raise ValueError(f"Unsupported batch directory: {name}")
        if name in seen_names:
            raise ValueError(f"Duplicate batch state: {name}")
        seen_names.add(name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Missing results list: {path}")
        run_ids = [item.get("runId") for item in results if isinstance(item, dict)]
        if len(run_ids) != expected_batch_size or any(not isinstance(value, str) or not value for value in run_ids):
            raise ValueError(f"Expected {expected_batch_size} valid runIds in {path}, got {len(run_ids)}")
        if len(set(run_ids)) != len(run_ids):
            raise ValueError(f"Duplicate runId inside {path}")
        label, concurrency = _BATCH_METADATA[name]
        specs.append(BatchSpec(path=path, label=label, concurrency=concurrency, run_ids=tuple(run_ids)))
        all_ids.extend(run_ids)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate runId across batch states")
    return specs


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 7) if denominator else None


def build_capability_records(
    batch_specs: list[BatchSpec],
    questions_by_id: dict[str, dict],
    runs_by_id: dict[str, dict],
    evaluations_by_run_id: dict[str, dict],
    *,
    expected_attempts: int = 3,
    expected_questions: int | None = None,
    pipeline_version: str = PIPELINE_VERSION,
) -> list[CapabilityRecord]:
    ordered_attempts: dict[str, list[tuple[BatchSpec, str]]] = defaultdict(list)
    seen_run_ids: set[str] = set()
    for batch in batch_specs:
        for run_id in batch.run_ids:
            if run_id in seen_run_ids:
                raise ValueError(f"Duplicate runId: {run_id}")
            seen_run_ids.add(run_id)
            run = runs_by_id.get(run_id)
            if run is None:
                raise ValueError(f"Missing Run: {run_id}")
            question_id = run.get("questionId")
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"Run has no questionId: {run_id}")
            ordered_attempts[question_id].append((batch, run_id))

    if expected_questions is not None and len(ordered_attempts) != expected_questions:
        raise ValueError(f"Expected {expected_questions} questions, got {len(ordered_attempts)}")

    records: list[CapabilityRecord] = []
    for question_id in sorted(ordered_attempts):
        attempts = ordered_attempts[question_id]
        if len(attempts) != expected_attempts:
            raise ValueError(f"Question {question_id} has {len(attempts)} attempts, expected {expected_attempts}")
        question = questions_by_id.get(question_id)
        if question is None:
            raise ValueError(f"Missing Question: {question_id}")

        details: list[AttemptDetail] = []
        runtime_counts: Counter[str] = Counter()
        verdict_counts: Counter[str] = Counter()
        for batch, run_id in attempts:
            run = runs_by_id[run_id]
            evaluation = evaluations_by_run_id.get(run_id)
            if evaluation is None:
                raise ValueError(f"Missing Evaluation: {run_id}")
            if evaluation.get("pipelineVersion") != pipeline_version:
                raise ValueError(f"Wrong pipelineVersion for {run_id}: {evaluation.get('pipelineVersion')}")
            if evaluation.get("questionId") != question_id:
                raise ValueError(f"Evaluation questionId mismatch: {run_id}")
            runtime_status = str(run.get("status"))
            if runtime_status not in _RUNTIME_STATUSES:
                raise ValueError(f"Unsupported runtime status for {run_id}: {runtime_status}")
            math = evaluation.get("math") or {}
            verdict = math.get("verdict")
            if verdict not in _MATH_VERDICTS:
                raise ValueError(f"Unsupported math verdict for {run_id}: {verdict}")
            runtime_counts[runtime_status] += 1
            verdict_counts[verdict] += 1
            details.append(AttemptDetail(
                runId=run_id,
                batch=batch.label,
                concurrency=batch.concurrency,
                runtimeStatus=runtime_status,
                mathVerdict=verdict,
                evaluationLevel=int(math.get("level")),
                evaluationMethod=str(math.get("method")),
                confidence=float(math.get("confidence", 0.0)),
            ))

        attempts_count = len(details)
        correct = verdict_counts["correct"]
        incorrect = verdict_counts["incorrect"]
        gradable = correct + incorrect
        local_success = sum(
            detail.runtime_status == "completed" and detail.math_verdict == "correct"
            for detail in details
        )
        runtime = RuntimeCounts(**{status: runtime_counts[status] for status in _RUNTIME_STATUSES})
        records.append(CapabilityRecord(
            questionId=question_id,
            question=str(question.get("question", "")),
            tag=question.get("tag"),
            mathType=question.get("math_type"),
            difficulty=question.get("difficulty"),
            capabilityVersion=CAPABILITY_VERSION,
            evaluatorPipelineVersion=pipeline_version,
            attempts=attempts_count,
            gradableAttempts=gradable,
            mathCorrect=correct,
            mathIncorrect=incorrect,
            review=verdict_counts["review"],
            unresolved=verdict_counts["unresolved"],
            runtime=runtime,
            runtimeSuccessRate=_rate(runtime.completed, attempts_count) or 0.0,
            mathPassRate=_rate(correct, gradable),
            localSuccessCount=local_success,
            localSuccessRate=_rate(local_success, attempts_count) or 0.0,
            labelUsable=gradable == attempts_count,
            attemptDetails=details,
        ))
    return records
