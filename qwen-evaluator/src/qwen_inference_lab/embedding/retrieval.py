"""In-memory NumPy cosine retrieval and leave-one-out evaluation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .models import Neighbor, Neighborhood


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(list(left), dtype=np.float64)
    b = np.asarray(list(right), dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("Vectors must be non-empty and have the same shape")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")
    value = float(np.dot(a, b) / denominator)
    if not math.isfinite(value):
        raise ValueError("Cosine similarity is not finite")
    return value


def _validated_matrix(records: list[dict]) -> tuple[np.ndarray, dict[str, int]]:
    if not records:
        raise ValueError("No embedding records")
    ids = [record.get("questionId") for record in records]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Embedding record has no questionId")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate questionId in embedding records")
    dimensions = {int(record.get("dimension", 0)) for record in records}
    if len(dimensions) != 1 or next(iter(dimensions)) <= 0:
        raise ValueError(f"Invalid embedding dimensions: {dimensions}")
    matrix = np.asarray([record.get("embedding") for record in records], dtype=np.float64)
    dimension = next(iter(dimensions))
    if matrix.shape != (len(records), dimension):
        raise ValueError("Embedding matrix shape does not match dimensions")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains NaN or infinity")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embedding matrix contains empty/zero vectors")
    normalized = matrix / norms[:, None]
    return normalized, {question_id: index for index, question_id in enumerate(ids)}


def _neighbor(record: dict, similarity: float) -> Neighbor:
    return Neighbor(
        questionId=record["questionId"],
        similarity=round(float(similarity), 8),
        mathType=record.get("mathType"),
        difficulty=record.get("difficulty"),
        tag=record.get("tag"),
        localSuccessRate=float(record["localSuccessRate"]),
        labelUsable=bool(record["labelUsable"]),
    )


def retrieve_one(records: list[dict], question_id: str, top_k: int = 10, *, exclude_self: bool = True) -> list[Neighbor]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    matrix, index_by_id = _validated_matrix(records)
    if question_id not in index_by_id:
        raise ValueError(f"Unknown questionId: {question_id}")
    query_index = index_by_id[question_id]
    similarities = matrix @ matrix[query_index]
    candidates = [
        index for index in range(len(records))
        if not (exclude_self and index == query_index)
    ]
    candidates.sort(key=lambda index: (-float(similarities[index]), str(records[index]["questionId"])))
    return [_neighbor(records[index], similarities[index]) for index in candidates[:top_k]]


def _usable_mean(neighbors: list[Neighbor]) -> tuple[int, float | None]:
    values = [neighbor.local_success_rate for neighbor in neighbors if neighbor.label_usable]
    return len(values), round(float(np.mean(values)), 7) if values else None


def build_neighborhoods(records: list[dict]) -> list[Neighborhood]:
    _validated_matrix(records)
    neighborhoods: list[Neighborhood] = []
    for record in records:
        top10 = retrieve_one(records, record["questionId"], 10, exclude_self=True)
        top5 = top10[:5]
        top5_count, top5_mean = _usable_mean(top5)
        top10_count, top10_mean = _usable_mean(top10)
        neighborhoods.append(Neighborhood(
            questionId=record["questionId"],
            mathType=record.get("mathType"),
            difficulty=record.get("difficulty"),
            tag=record.get("tag"),
            localSuccessRate=float(record["localSuccessRate"]),
            labelUsable=bool(record["labelUsable"]),
            top5=top5,
            top10=top10,
            top5UsableNeighbors=top5_count,
            top5LocalSuccessMean=top5_mean,
            top10UsableNeighbors=top10_count,
            top10LocalSuccessMean=top10_mean,
        ))
    return neighborhoods


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    names = ("min", "p25", "median", "p75", "p90", "p95", "max")
    quantiles = np.quantile(array, (0, 0.25, 0.5, 0.75, 0.90, 0.95, 1))
    return {name: round(float(value), 8) for name, value in zip(names, quantiles, strict=True)}


def _match_rate(neighborhoods: list[Neighborhood], field: str, top_k: int, *, require_query_value: bool = False) -> float | None:
    matches = 0
    total = 0
    for neighborhood in neighborhoods:
        query_value = getattr(neighborhood, field)
        if require_query_value and not query_value:
            continue
        for neighbor in neighborhood.top10[:top_k]:
            matches += getattr(neighbor, field) == query_value
            total += 1
    return round(matches / total, 7) if total else None


def evaluate_leave_one_out(records: list[dict]) -> tuple[dict, list[Neighborhood]]:
    neighborhoods = build_neighborhoods(records)
    top1 = [item.top10[0].similarity for item in neighborhoods]
    top5_mean = [float(np.mean([neighbor.similarity for neighbor in item.top5])) for item in neighborhoods]
    top10_mean = [float(np.mean([neighbor.similarity for neighbor in item.top10])) for item in neighborhoods]
    self_matches = sum(
        neighbor.question_id == item.question_id
        for item in neighborhoods
        for neighbor in item.top10
    )
    summary = {
        "questions": len(records),
        "dimension": int(records[0]["dimension"]) if records else 0,
        "nanVectors": sum(any(not math.isfinite(float(value)) for value in record["embedding"]) for record in records),
        "emptyVectors": sum(not bool(record.get("embedding")) for record in records),
        "leaveOneOutSelfMatches": self_matches,
        "similarityDistribution": {
            "top1": _distribution(top1),
            "top5Mean": _distribution(top5_mean),
            "top10Mean": _distribution(top10_mean),
        },
        "top5MathTypeMatchRate": _match_rate(neighborhoods, "math_type", 5),
        "top10MathTypeMatchRate": _match_rate(neighborhoods, "math_type", 10),
        "top5TagMatchRate": _match_rate(neighborhoods, "tag", 5, require_query_value=True),
        "top10TagMatchRate": _match_rate(neighborhoods, "tag", 10, require_query_value=True),
        "top5DifficultyMatchRate": _match_rate(neighborhoods, "difficulty", 5),
        "top10DifficultyMatchRate": _match_rate(neighborhoods, "difficulty", 10),
        "nonPerfectQuestions": sum(item.local_success_rate < 1.0 for item in neighborhoods),
    }
    return summary, neighborhoods
