"""Conservative reference-answer routing; it never solves a question."""

from __future__ import annotations

from hashlib import sha256
import re
from importlib.metadata import version

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse

from . import GOLD_ADAPTER_VERSION
from .models import EvaluationRoute, GoldProfile, GoldStatus

_MULTI_PART = re.compile(r"(?:\([a-zA-Z]\)|（[a-zA-Z]）|\([0-9]+\)|（[0-9]+）|(?:^|\n)\s*(?:[12][\.)]))")
_SEMANTIC = re.compile(r"证明|prove|explain|justify|why|show that|论证", re.IGNORECASE)


def build_gold_profile(question: dict[str, object]) -> GoldProfile:
    """Classify only from question/reference content, prioritising precision."""
    question_id = str(question["id"])
    prompt = str(question.get("question", ""))
    reference = str(question.get("reference_answer", "")).strip()
    base = dict(question_id=question_id, adapter_version=GOLD_ADAPTER_VERSION,
                math_verify_version=version("math-verify"),
                reference_hash=sha256(reference.encode()).hexdigest(), raw_reference=reference)
    if _MULTI_PART.search(prompt):
        return GoldProfile(**base, question_type="multi_part", evaluation_route=EvaluationRoute.LLM_MULTI_PART,
                           status=GoldStatus.MULTI_PART, parse={"success": False, "serialized": []}, reason_code="MULTI_PART_QUESTION")
    if _SEMANTIC.search(prompt):
        return GoldProfile(**base, question_type="semantic", evaluation_route=EvaluationRoute.LLM_SEMANTIC,
                           status=GoldStatus.SEMANTIC_REQUIRED, parse={"success": False, "serialized": []}, reason_code="SEMANTIC_QUESTION")
    if not reference:
        return GoldProfile(**base, question_type="unknown", evaluation_route=EvaluationRoute.UNRESOLVED,
                           status=GoldStatus.PARSE_FAILED, parse={"success": False, "serialized": []}, reason_code="EMPTY_REFERENCE")
    try:
        parsed = parse(reference, [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()], fallback_mode="no_fallback", extraction_mode="any_match", parsing_timeout=5, raise_on_error=True)
    except Exception as error:
        return GoldProfile(**base, question_type="unknown", evaluation_route=EvaluationRoute.UNRESOLVED, status=GoldStatus.PARSE_FAILED, parse={"success": False, "serialized": [], "error": type(error).__name__}, reason_code="GOLD_PARSE_ERROR")
    serialized = [str(item) for item in parsed]
    if len(parsed) != 1:
        return GoldProfile(**base, question_type="ambiguous", evaluation_route=EvaluationRoute.UNRESOLVED, status=GoldStatus.AMBIGUOUS, parse={"success": bool(parsed), "serialized": serialized}, reason_code="NON_SINGLE_GOLD_EXTRACTION")
    return GoldProfile(**base, question_type="single_expression", evaluation_route=EvaluationRoute.MATH_VERIFY, status=GoldStatus.RESOLVED, gold_candidates=serialized, parse={"success": True, "serialized": serialized}, reason_code="DIRECT_MATH_GOLD")
