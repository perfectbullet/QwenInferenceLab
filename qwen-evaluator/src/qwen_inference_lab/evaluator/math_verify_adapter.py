"""Thin evidence-preserving adapter around Math-Verify 0.9.0."""

from __future__ import annotations

from dataclasses import dataclass
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


@dataclass(frozen=True)
class MathVerifyResult:
    verdict: str
    gold_parsed: list[str]
    prediction_parsed: list[str]
    error: str | None = None


def evaluate_single_answer(gold_text: str, prediction_text: str) -> MathVerifyResult:
    config = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
    try:
        gold = parse(gold_text, config, fallback_mode="no_fallback", extraction_mode="any_match", parsing_timeout=5, raise_on_error=True)
        prediction = parse(prediction_text, config, fallback_mode="no_fallback", extraction_mode="any_match", parsing_timeout=5, raise_on_error=True)
    except Exception as error:
        return MathVerifyResult("unresolved", [], [], type(error).__name__)
    if not gold or not prediction:
        return MathVerifyResult("unresolved", [str(value) for value in gold], [str(value) for value in prediction], "PARSE_EMPTY")
    try:
        equivalent = verify(gold, prediction, strict=True, float_rounding=6, timeout_seconds=5, raise_on_error=True)
    except Exception as error:
        return MathVerifyResult("unresolved", [str(value) for value in gold], [str(value) for value in prediction], type(error).__name__)
    return MathVerifyResult("correct" if equivalent else "incorrect", [str(value) for value in gold], [str(value) for value in prediction])
