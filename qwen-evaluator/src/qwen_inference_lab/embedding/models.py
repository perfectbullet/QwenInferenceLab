"""Typed records for embeddings and retrieval neighborhoods."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingRecord(BaseModel):
    question_id: str = Field(alias="questionId")
    embedding_version: str = Field(alias="embeddingVersion")
    model: str
    base_url: str = Field(alias="baseUrl")
    dimension: int
    text_hash: str = Field(alias="textHash")
    question: str
    tag: str | None = None
    math_type: str | None = Field(alias="mathType", default=None)
    difficulty: str | None = None
    capability_version: str = Field(alias="capabilityVersion")
    local_success_rate: float = Field(alias="localSuccessRate")
    math_pass_rate: float | None = Field(alias="mathPassRate")
    label_usable: bool = Field(alias="labelUsable")
    embedding: list[float]

    model_config = {"populate_by_name": True}


class Neighbor(BaseModel):
    question_id: str = Field(alias="questionId")
    similarity: float
    math_type: str | None = Field(alias="mathType", default=None)
    difficulty: str | None = None
    tag: str | None = None
    local_success_rate: float = Field(alias="localSuccessRate")
    label_usable: bool = Field(alias="labelUsable")

    model_config = {"populate_by_name": True}


class Neighborhood(BaseModel):
    question_id: str = Field(alias="questionId")
    math_type: str | None = Field(alias="mathType", default=None)
    difficulty: str | None = None
    tag: str | None = None
    local_success_rate: float = Field(alias="localSuccessRate")
    label_usable: bool = Field(alias="labelUsable")
    top5: list[Neighbor]
    top10: list[Neighbor]
    top5_usable_neighbors: int = Field(alias="top5UsableNeighbors")
    top5_local_success_mean: float | None = Field(alias="top5LocalSuccessMean")
    top10_usable_neighbors: int = Field(alias="top10UsableNeighbors")
    top10_local_success_mean: float | None = Field(alias="top10LocalSuccessMean")

    model_config = {"populate_by_name": True}
