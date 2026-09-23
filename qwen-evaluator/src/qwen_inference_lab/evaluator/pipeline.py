"""Evaluator routing pipeline with auditable evidence."""
from __future__ import annotations
from importlib.metadata import version
from uuid import uuid4
from . import GOLD_ADAPTER_VERSION, LLM_JUDGE_PROMPT_VERSION, MULTI_PART_JUDGE_PROMPT_VERSION, PIPELINE_VERSION
from .gold_adapter import build_gold_profile
from .llm_judge import judge_answer
from .multi_part_judge import judge_multi_part
from .repository import EvaluatorRepository

FAILED_RUNTIME = {"failed", "cancelled", "interrupted"}

def select_judge_config(repository: EvaluatorRepository, config_id: str | None) -> dict | None:
    configs = repository.find_judge_configs(config_id)
    return configs[0] if len(configs) == 1 else None

def judge_evidence(result, config: dict | None, mode: str) -> dict:
    return {
        "modelName": config.get("modelName") if config else None,
        "baseUrl": config.get("baseUrl") if config else None,
        "promptVersion": MULTI_PART_JUDGE_PROMPT_VERSION if mode == "multi_part" else LLM_JUDGE_PROMPT_VERSION,
        "judgeMode": mode,
        "structuredResponse": result.response.model_dump(by_alias=True) if result.response else None,
        "error": result.error,
    }

def evaluate_run(repository: EvaluatorRepository, run_id: str, *, judge_model_config_id: str | None = None, max_level: int = 3, force: bool = False, judge_retries: int = 2, judge_timeout: float = 60) -> tuple[str, dict | None]:
    existing = repository.get_evaluation(run_id)
    if existing and not force:
        math = existing.get("math", {})
        can_upgrade = math.get("reasonCode") in {"MAX_LEVEL_REACHED", "JUDGE_FAILED", "JUDGE_UNAVAILABLE"} and max_level >= int(math.get("level", 99))
        if not can_upgrade:
            return "skipped", None
    run = repository.get_run(run_id)
    if not run:
        raise ValueError(f"Run not found: {run_id}")
    question_id = str(run["questionId"])
    question = repository.get_question(question_id)
    if not question:
        raise ValueError(f"Question not found: {question_id}")
    profile = repository.get_gold_profile(question_id)
    if not profile:
        built = build_gold_profile(question)
        repository.upsert_gold_profile(built)
        profile = built.model_dump(by_alias=True)
    document = {
        "id": str(uuid4()), "runId": run_id, "questionId": question_id,
        "pipelineVersion": PIPELINE_VERSION, "goldAdapterVersion": GOLD_ADAPTER_VERSION,
        "mathVerifyVersion": version("math-verify"),
        "runtime": {"status": run.get("status"), "finishReason": run.get("finishReason"), "error": run.get("error")},
        "verifyConfig": None,
        "extraction": None, "judge": None,
    }
    if run.get("status") in FAILED_RUNTIME:
        document["math"] = {"verdict": "unresolved", "level": 0, "method": "runtime_precheck", "confidence": 0.0, "reasonCode": "RUNTIME_FAILURE"}
        repository.upsert_evaluation(document)
        return "evaluated", document
    route = profile["evaluationRoute"]
    config = select_judge_config(repository, judge_model_config_id)
    if route == "llm_multi_part":
        mode = "multi_part"
        judge = judge_multi_part(question["question"], profile["rawReference"], str(run.get("answer") or ""), config, retries=judge_retries, timeout=judge_timeout) if max_level >= 2 else None
        level = 2
    else:
        mode = "semantic"
        judge = judge_answer(mode=mode, question=question["question"], reference=profile["rawReference"], candidate=str(run.get("answer") or ""), config=config, retries=judge_retries, timeout=judge_timeout) if max_level >= 3 else None
        level = 3
    if judge is None:
        document["math"] = {"verdict": "unresolved", "level": level, "method": "level_disabled", "confidence": 0.0, "reasonCode": "MAX_LEVEL_REACHED"}
    else:
        document["math"] = {
            "verdict": judge.verdict, "level": level,
            "method": "llm_multi_part" if mode == "multi_part" else "llm_semantic",
            "confidence": judge.response.confidence if judge.response else 0.0,
            "reasonCode": judge.reason_code,
        }
        document["judge"] = judge_evidence(judge, config, mode)
    repository.upsert_evaluation(document)
    return "evaluated", document
