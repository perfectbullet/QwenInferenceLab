"""Failure-aware features computed only from a fold's reference corpus."""

from __future__ import annotations

import numpy as np

from .models import RouterFeatures, RouterNeighbor


def similarity_weight(similarity: float, power: int) -> float:
    if power not in {1, 2}:
        raise ValueError("weight power must be 1 or 2")
    return max(float(similarity), 0.0) ** power


def weighted_success(neighbors: list[RouterNeighbor], power: int = 1) -> float | None:
    usable = [neighbor for neighbor in neighbors if neighbor.label_usable]
    weights = [similarity_weight(neighbor.similarity, power) for neighbor in usable]
    denominator = sum(weights)
    if not usable or denominator == 0:
        return None
    return float(sum(
        weight * neighbor.local_success_rate
        for weight, neighbor in zip(weights, usable, strict=True)
    ) / denominator)


def weighted_non_perfect_rate(neighbors: list[RouterNeighbor]) -> float | None:
    usable = [neighbor for neighbor in neighbors if neighbor.label_usable]
    weights = [similarity_weight(neighbor.similarity, 1) for neighbor in usable]
    denominator = sum(weights)
    if not usable or denominator == 0:
        return None
    return float(sum(
        weight * (neighbor.safe_local is False)
        for weight, neighbor in zip(weights, usable, strict=True)
    ) / denominator)


def compute_features(all_neighbors: list[RouterNeighbor], k: int) -> tuple[RouterFeatures, list[RouterNeighbor]]:
    if k < 1:
        raise ValueError("k must be positive")
    if len(all_neighbors) < k:
        raise ValueError(f"Need at least {k} neighbors, got {len(all_neighbors)}")
    neighbors = all_neighbors[:k]
    similarities = np.asarray([neighbor.similarity for neighbor in neighbors], dtype=np.float64)
    usable = [neighbor for neighbor in neighbors if neighbor.label_usable]
    successes = np.asarray(
        [neighbor.local_success_rate for neighbor in usable],
        dtype=np.float64,
    )
    non_perfect = [neighbor for neighbor in usable if neighbor.safe_local is False]

    nearest_safe = next(
        (neighbor.similarity for neighbor in all_neighbors if neighbor.safe_local is True),
        None,
    )
    nearest_non_perfect = next(
        (neighbor.similarity for neighbor in all_neighbors if neighbor.safe_local is False),
        None,
    )
    margin = (
        nearest_safe - nearest_non_perfect
        if nearest_safe is not None and nearest_non_perfect is not None
        else None
    )
    features = RouterFeatures(
        k=k,
        top1Similarity=round(float(similarities[0]), 8),
        topKMeanSimilarity=round(float(np.mean(similarities)), 8),
        topKMinSimilarity=round(float(np.min(similarities)), 8),
        similarityStd=round(float(np.std(similarities)), 8),
        usableNeighborCount=len(usable),
        neighborSuccessMean=round(float(np.mean(successes)), 8) if len(successes) else None,
        neighborSuccessStd=round(float(np.std(successes)), 8) if len(successes) else None,
        weightedSuccess=_rounded(weighted_success(neighbors, 1)),
        weightedSuccessSquared=_rounded(weighted_success(neighbors, 2)),
        nonPerfectNeighborCount=len(non_perfect),
        nonPerfectNeighborRate=round(len(non_perfect) / len(usable), 8) if usable else None,
        weightedNonPerfectRate=_rounded(weighted_non_perfect_rate(neighbors)),
        nearestNonPerfectSimilarity=_rounded(nearest_non_perfect),
        nearestSafeSimilarity=_rounded(nearest_safe),
        safeVsRiskMargin=_rounded(margin),
    )
    return features, neighbors


def _rounded(value: float | None) -> float | None:
    return round(float(value), 8) if value is not None else None
