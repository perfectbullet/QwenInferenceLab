"""Typed capability dataset records."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RuntimeCounts(BaseModel):
    completed: int = 0
    truncated: int = 0
    failed: int = 0
    cancelled: int = 0
    interrupted: int = 0


class AttemptDetail(BaseModel):
    run_id: str = Field(alias="runId")
    batch: str
    concurrency: int
    runtime_status: str = Field(alias="runtimeStatus")
    math_verdict: str = Field(alias="mathVerdict")
    evaluation_level: int = Field(alias="evaluationLevel")
    evaluation_method: str = Field(alias="evaluationMethod")
    confidence: float

    model_config = {"populate_by_name": True}


class CapabilityRecord(BaseModel):
    question_id: str = Field(alias="questionId")
    question: str
    tag: str | None = None
    math_type: str | None = Field(alias="mathType", default=None)
    difficulty: str | None = None
    capability_version: str = Field(alias="capabilityVersion")
    evaluator_pipeline_version: str = Field(alias="evaluatorPipelineVersion")
    attempts: int
    gradable_attempts: int = Field(alias="gradableAttempts")
    math_correct: int = Field(alias="mathCorrect")
    math_incorrect: int = Field(alias="mathIncorrect")
    review: int
    unresolved: int
    runtime: RuntimeCounts
    runtime_success_rate: float = Field(alias="runtimeSuccessRate")
    math_pass_rate: float | None = Field(alias="mathPassRate")
    local_success_count: int = Field(alias="localSuccessCount")
    local_success_rate: float = Field(alias="localSuccessRate")
    label_usable: bool = Field(alias="labelUsable")
    attempt_details: list[AttemptDetail] = Field(alias="attemptDetails")

    model_config = {"populate_by_name": True}
