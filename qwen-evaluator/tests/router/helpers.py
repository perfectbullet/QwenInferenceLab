from __future__ import annotations

import math

from qwen_inference_lab.router.models import (
    RouterFeatures,
    RouterNeighbor,
    RouterQuestion,
)


def question(
    question_id: str,
    success: float,
    *,
    usable: bool = True,
    angle: float = 0.0,
) -> RouterQuestion:
    return RouterQuestion(
        questionId=question_id,
        question=f"Question {question_id}",
        embedding=[math.cos(angle), math.sin(angle)],
        dimension=2,
        localSuccessRate=success,
        labelUsable=usable,
    )


def neighbor(
    question_id: str,
    similarity: float,
    success: float,
    *,
    usable: bool = True,
) -> RouterNeighbor:
    safe = None if not usable else success == 1.0
    return RouterNeighbor(
        questionId=question_id,
        similarity=similarity,
        localSuccessRate=success,
        labelUsable=usable,
        safeLocal=safe,
    )


def features(**overrides) -> RouterFeatures:
    values = {
        "k": 3,
        "top1Similarity": 0.9,
        "topKMeanSimilarity": 0.8,
        "topKMinSimilarity": 0.7,
        "similarityStd": 0.08,
        "usableNeighborCount": 3,
        "neighborSuccessMean": 0.9,
        "neighborSuccessStd": 0.1,
        "weightedSuccess": 0.9,
        "weightedSuccessSquared": 0.91,
        "nonPerfectNeighborCount": 1,
        "nonPerfectNeighborRate": 1 / 3,
        "weightedNonPerfectRate": 0.3,
        "nearestNonPerfectSimilarity": 0.82,
        "nearestSafeSimilarity": 0.9,
        "safeVsRiskMargin": 0.08,
    }
    values.update(overrides)
    return RouterFeatures(**values)


def cv_records() -> list[RouterQuestion]:
    records = []
    for index in range(10):
        records.append(question(f"S{index}", 1.0, angle=index * 0.05))
    for index in range(5):
        records.append(question(f"R{index}", index / 12, angle=0.8 + index * 0.05))
    for index in range(5):
        records.append(question(f"U{index}", index / 12, usable=False, angle=1.3 + index * 0.05))
    return records
