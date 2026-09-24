"""Typed records for offline router experiments."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RouterQuestion(BaseModel):
    question_id: str = Field(alias="questionId")
    question: str
    embedding: list[float]
    dimension: int
    local_success_rate: float = Field(alias="localSuccessRate")
    label_usable: bool = Field(alias="labelUsable")
    math_type: str | None = Field(alias="mathType", default=None)
    difficulty: str | None = None
    tag: str | None = None

    model_config = {"populate_by_name": True}

    @property
    def safe_local(self) -> bool | None:
        if not self.label_usable:
            return None
        return self.local_success_rate == 1.0


class RouterNeighbor(BaseModel):
    question_id: str = Field(alias="questionId")
    similarity: float
    local_success_rate: float = Field(alias="localSuccessRate")
    label_usable: bool = Field(alias="labelUsable")
    safe_local: bool | None = Field(alias="safeLocal")

    model_config = {"populate_by_name": True}


class RouterFeatures(BaseModel):
    k: int
    top1_similarity: float = Field(alias="top1Similarity")
    top_k_mean_similarity: float = Field(alias="topKMeanSimilarity")
    top_k_min_similarity: float = Field(alias="topKMinSimilarity")
    similarity_std: float = Field(alias="similarityStd")
    usable_neighbor_count: int = Field(alias="usableNeighborCount")
    neighbor_success_mean: float | None = Field(alias="neighborSuccessMean")
    neighbor_success_std: float | None = Field(alias="neighborSuccessStd")
    weighted_success: float | None = Field(alias="weightedSuccess")
    weighted_success_squared: float | None = Field(alias="weightedSuccessSquared")
    non_perfect_neighbor_count: int = Field(alias="nonPerfectNeighborCount")
    non_perfect_neighbor_rate: float | None = Field(alias="nonPerfectNeighborRate")
    weighted_non_perfect_rate: float | None = Field(alias="weightedNonPerfectRate")
    nearest_non_perfect_similarity: float | None = Field(alias="nearestNonPerfectSimilarity")
    nearest_safe_similarity: float | None = Field(alias="nearestSafeSimilarity")
    safe_vs_risk_margin: float | None = Field(alias="safeVsRiskMargin")

    model_config = {"populate_by_name": True}


class PolicyConfig(BaseModel):
    policy: str
    k: int
    score_threshold: float = Field(alias="scoreThreshold")
    weight_power: int = Field(alias="weightPower", default=1)
    ood_threshold: float | None = Field(alias="oodThreshold", default=None)
    max_non_perfect_rate: float | None = Field(alias="maxNonPerfectRate", default=None)
    max_nearest_non_perfect_similarity: float | None = Field(
        alias="maxNearestNonPerfectSimilarity", default=None
    )
    min_safe_vs_risk_margin: float | None = Field(alias="minSafeVsRiskMargin", default=None)

    model_config = {"populate_by_name": True}


class RouterPrediction(BaseModel):
    question_id: str = Field(alias="questionId")
    fold: int
    local_success_rate: float = Field(alias="localSuccessRate")
    label_usable: bool = Field(alias="labelUsable")
    safe_local: bool | None = Field(alias="safeLocal")
    decision: str
    score: float | None
    features: RouterFeatures
    neighbors: list[RouterNeighbor]
    reason_codes: list[str] = Field(alias="reasonCodes")

    model_config = {"populate_by_name": True}
