"""Cosine retrieval for fold-local reference corpora."""

from __future__ import annotations

import numpy as np

from .models import RouterNeighbor, RouterQuestion


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("Vectors must be non-empty and have equal shape")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")
    return float(np.dot(a, b) / denominator)


def retrieve_neighbors(
    query: RouterQuestion,
    references: list[RouterQuestion],
    top_k: int | None = None,
) -> list[RouterNeighbor]:
    if not references:
        raise ValueError("Reference corpus is empty")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    if any(reference.question_id == query.question_id for reference in references):
        raise ValueError(f"Held-out query leaked into reference corpus: {query.question_id}")

    query_vector = np.asarray(query.embedding, dtype=np.float64)
    matrix = np.asarray([reference.embedding for reference in references], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != query_vector.size:
        raise ValueError("Embedding dimensions are inconsistent")
    query_norm = float(np.linalg.norm(query_vector))
    reference_norms = np.linalg.norm(matrix, axis=1)
    if query_norm == 0 or np.any(reference_norms == 0):
        raise ValueError("Zero vector cannot be used for cosine retrieval")
    similarities = (matrix @ query_vector) / (reference_norms * query_norm)
    if not np.isfinite(similarities).all():
        raise ValueError("Cosine similarities contain NaN or infinity")

    indices = list(range(len(references)))
    indices.sort(key=lambda index: (-float(similarities[index]), references[index].question_id))
    if top_k is not None:
        indices = indices[:top_k]
    return [
        RouterNeighbor(
            questionId=references[index].question_id,
            similarity=round(float(similarities[index]), 8),
            localSuccessRate=references[index].local_success_rate,
            labelUsable=references[index].label_usable,
            safeLocal=references[index].safe_local,
        )
        for index in indices
    ]
