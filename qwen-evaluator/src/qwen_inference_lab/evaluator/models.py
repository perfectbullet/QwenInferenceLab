"""Typed domain records for auditable evaluator persistence."""

from __future__ import annotations

from enum import StrEnum
from pydantic import BaseModel, Field


class GoldStatus(StrEnum):
    RESOLVED = "resolved"
    MULTI_PART = "multi_part"
    SEMANTIC_REQUIRED = "semantic_required"
    AMBIGUOUS = "ambiguous"
    PARSE_FAILED = "parse_failed"


class EvaluationRoute(StrEnum):
    MATH_VERIFY = "math_verify"
    LLM_MULTI_PART = "llm_multi_part"
    LLM_SEMANTIC = "llm_semantic"
    UNRESOLVED = "unresolved"


class GoldProfile(BaseModel):
    question_id: str = Field(alias="questionId")
    adapter_version: str = Field(alias="adapterVersion")
    math_verify_version: str = Field(alias="mathVerifyVersion")
    reference_hash: str = Field(alias="referenceHash")
    question_type: str = Field(alias="questionType")
    evaluation_route: EvaluationRoute = Field(alias="evaluationRoute")
    status: GoldStatus
    raw_reference: str = Field(alias="rawReference")
    gold_candidates: list[str] = Field(alias="goldCandidates", default_factory=list)
    parse: dict[str, object]
    reason_code: str = Field(alias="reasonCode")

    model_config = {"populate_by_name": True}
