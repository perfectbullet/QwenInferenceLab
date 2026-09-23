"""Versioned OpenAI-compatible semantic judge with strict validation."""
from __future__ import annotations
import json
import time
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from . import LLM_JUDGE_PROMPT_VERSION, MULTI_PART_JUDGE_PROMPT_VERSION

CONFIDENCE_THRESHOLD = 0.85
JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "incorrect", "review"]},
        "complete": {"type": "boolean"},
        "hasCriticalError": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "allSubquestionsAnswered": {"type": ["boolean", "null"]},
        "subquestions": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"id": {"type": "string"}, "answered": {"type": "boolean"}, "correct": {"type": ["boolean", "null"]}, "reason": {"type": "string"}}, "required": ["id", "answered", "correct", "reason"]}},
    },
    "required": ["verdict", "complete", "hasCriticalError", "confidence", "reason", "issues", "allSubquestionsAnswered", "subquestions"],
}

class JudgeSubquestion(BaseModel):
    id: str
    answered: bool
    correct: bool | None = None
    reason: str

class JudgeResponse(BaseModel):
    verdict: Literal["correct", "incorrect", "review"]
    complete: bool
    has_critical_error: bool = Field(alias="hasCriticalError")
    confidence: float = Field(ge=0, le=1)
    reason: str
    issues: list[str] = Field(default_factory=list)
    all_subquestions_answered: bool | None = Field(default=None, alias="allSubquestionsAnswered")
    subquestions: list[JudgeSubquestion] = Field(default_factory=list)
    model_config = {"populate_by_name": True}

class JudgeResult(BaseModel):
    verdict: Literal["correct", "incorrect", "review", "unresolved"]
    reason_code: str
    response: JudgeResponse | None = None
    error: str | None = None

def build_messages(mode: Literal["multi_part", "semantic"], question: str, reference: str, candidate: str) -> list[dict[str, str]]:
    version = MULTI_PART_JUDGE_PROMPT_VERSION if mode == "multi_part" else LLM_JUDGE_PROMPT_VERSION
    requirement = "Identify every subquestion, verify all parts, and mark omitted required parts incorrect." if mode == "multi_part" else "Judge correctness and completeness; choose review when ambiguous."
    system = f"You are a conservative math judge. Prompt version: {version}. {requirement} Return only JSON with verdict, complete, hasCriticalError, confidence, reason, issues, allSubquestionsAnswered, and subquestions."
    user = json.dumps({"question": question, "reference_answer": reference, "candidate_answer": candidate}, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def judge_answer(*, mode: Literal["multi_part", "semantic"], question: str, reference: str, candidate: str, config: dict | None, retries: int = 2, timeout: float = 60) -> JudgeResult:
    if not config:
        return JudgeResult(verdict="unresolved", reason_code="JUDGE_UNAVAILABLE", error="No unique judge configuration")
    last_error = "unknown"
    for attempt in range(retries + 1):
        try:
            client = OpenAI(base_url=config["baseUrl"], api_key=config.get("apiKey") or "not-required", timeout=timeout)
            response = client.chat.completions.create(model=config["modelName"], messages=build_messages(mode, question, reference, candidate), response_format={"type": "json_schema", "json_schema": {"name": "math_judge", "strict": True, "schema": JUDGE_SCHEMA}}, temperature=0)
            parsed = JudgeResponse.model_validate_json(response.choices[0].message.content or "")
            if parsed.confidence < CONFIDENCE_THRESHOLD:
                parsed.verdict = "review"
                return JudgeResult(verdict="review", reason_code="LOW_CONFIDENCE", response=parsed)
            if mode == "multi_part" and parsed.all_subquestions_answered is False:
                parsed.verdict = "incorrect"
                parsed.complete = False
                return JudgeResult(verdict="incorrect", reason_code="INCOMPLETE_SUBQUESTIONS", response=parsed)
            return JudgeResult(verdict=parsed.verdict, reason_code="LLM_JUDGE_VERDICT", response=parsed)
        except (ValidationError, json.JSONDecodeError) as error:
            last_error = f"INVALID_JUDGE_RESPONSE:{type(error).__name__}"
        except Exception as error:
            last_error = f"JUDGE_REQUEST_FAILED:{type(error).__name__}"
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))
    return JudgeResult(verdict="unresolved", reason_code="JUDGE_FAILED", error=last_error)
