"""Batch-state loading, statistics, and human-audit manifest generation."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
from .repository import EvaluatorRepository

KNOWN_CONCURRENCY = {
    "batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4": 4,
    "batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4-round2": 8,
    "batch-20260922-nvidia-Qwen3.6-35B-A3B-NVFP4-round3": 8,
}

def load_batch_state(path: Path) -> dict:
    state = json.loads(path.read_text(encoding="utf-8"))
    run_ids = [item["runId"] for item in state.get("results", []) if isinstance(item.get("runId"), str)]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError(f"Duplicate runId in {path}")
    return {"path": str(path), "name": path.parent.name, "concurrency": KNOWN_CONCURRENCY.get(path.parent.name, "unknown"), "startedAt": state.get("startedAt"), "status": state.get("status"), "model": state.get("model"), "params": state.get("params"), "runIds": run_ids}

def summarize(evaluations: list[dict]) -> dict:
    return {
        "runtime": dict(Counter(item.get("runtime", {}).get("status", "unknown") for item in evaluations)),
        "levels": dict(Counter(str(item.get("math", {}).get("level", "unknown")) for item in evaluations)),
        "methods": dict(Counter(item.get("math", {}).get("method", "unknown") for item in evaluations)),
        "verdicts": dict(Counter(item.get("math", {}).get("verdict", "unresolved") for item in evaluations)),
        "reasonCodes": dict(Counter(item.get("math", {}).get("reasonCode", "unknown") for item in evaluations)),
        "evaluated": len(evaluations),
    }

def render_report(batch_reports: list[tuple[dict, dict]]) -> str:
    lines: list[str] = []
    for batch, summary in batch_reports:
        lines.extend([batch["name"], f"  state: {batch[chr(112)+chr(97)+chr(116)+chr(104)]}", f"  startedAt: {batch.get(chr(115)+chr(116)+chr(97)+chr(114)+chr(116)+chr(101)+chr(100)+chr(65)+chr(116))}", f"  status: {batch.get(chr(115)+chr(116)+chr(97)+chr(116)+chr(117)+chr(115))}", f"  concurrency: {batch.get(chr(99)+chr(111)+chr(110)+chr(99)+chr(117)+chr(114)+chr(114)+chr(101)+chr(110)+chr(99)+chr(121), chr(117)+chr(110)+chr(107)+chr(110)+chr(111)+chr(119)+chr(110))}", f"  runIds: {len(batch[chr(114)+chr(117)+chr(110)+chr(73)+chr(100)+chr(115)])}", f"  evaluated: {summary[chr(101)+chr(118)+chr(97)+chr(108)+chr(117)+chr(97)+chr(116)+chr(101)+chr(100)]}"])
        for heading, key in (("Runtime", "runtime"), ("Evaluation level", "levels"), ("Method", "methods"), ("Math verdict", "verdicts"), ("Reason code", "reasonCodes")):
            lines.append(f"  {heading}:")
            lines.extend(f"    {name}: {count}" for name, count in sorted(summary[key].items()))
    return "\n".join(lines)

def build_audit_manifest(repository: EvaluatorRepository, evaluations: list[dict], output: Path, target: int = 50) -> int:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for evaluation in evaluations:
        question = repository.get_question(evaluation["questionId"]) or {}
        run = repository.get_run(evaluation["runId"]) or {}
        key = (evaluation.get("math", {}).get("level"), evaluation.get("math", {}).get("verdict"), run.get("status"), question.get("math_type"), question.get("difficulty"), question.get("tag"))
        buckets[key].append({
            "runId": evaluation["runId"], "questionId": evaluation["questionId"],
            "question": question.get("question"), "referenceAnswer": question.get("reference_answer"),
            "candidateAnswer": run.get("answer"), "runtimeStatus": run.get("status"),
            "mathType": question.get("math_type"), "difficulty": question.get("difficulty"), "tag": question.get("tag"),
            "evaluation": evaluation.get("math"), "extraction": evaluation.get("extraction"), "judge": evaluation.get("judge"),
            "humanVerdict": None, "humanNotes": None,
        })
    selected: list[dict] = []
    ordered = [buckets[key] for key in sorted(buckets, key=lambda value: tuple(str(part) for part in value))]
    while len(selected) < target and any(ordered):
        for bucket in ordered:
            if bucket and len(selected) < target:
                selected.append(bucket.pop(0))
    payload = {"schemaVersion": "evaluation-audit-v1", "generatedAt": datetime.now(UTC).isoformat(), "target": target, "actual": len(selected), "instructions": "Human reviewers must fill humanVerdict and humanNotes. Machine output is evidence, not human Gold.", "samples": selected}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(selected)
