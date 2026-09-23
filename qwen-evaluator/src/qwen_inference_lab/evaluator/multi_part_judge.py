"""Explicit Level 2 boundary: multi-part questions bypass Math-Verify."""
from .llm_judge import judge_answer

def judge_multi_part(question: str, reference: str, candidate: str, config: dict | None, *, retries: int = 2, timeout: float = 60):
    return judge_answer(mode="multi_part", question=question, reference=reference, candidate=candidate, config=config, retries=retries, timeout=timeout)
