"""Build embedding records without mixing labels into embedding text."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

from . import EMBEDDING_VERSION
from .models import EmbeddingRecord


@dataclass(frozen=True)
class EmbeddingBuildStats:
    generated: int
    skipped: int
    dimension: int


def text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _valid_existing(document: dict, expected_hash: str, model: str, base_url: str) -> bool:
    vector = document.get("embedding")
    dimension = document.get("dimension")
    return (
        document.get("textHash") == expected_hash
        and document.get("model") == model
        and str(document.get("baseUrl", "")).rstrip("/") == base_url.rstrip("/")
        and isinstance(vector, list)
        and bool(vector)
        and dimension == len(vector)
        and all(math.isfinite(float(value)) for value in vector)
    )


def build_embedding_records(
    capabilities: list[dict],
    existing_by_question_id: dict[str, dict],
    embedder,
    *,
    model: str,
    base_url: str,
    embedding_version: str = EMBEDDING_VERSION,
    batch_size: int = 32,
    force: bool = False,
) -> tuple[list[EmbeddingRecord], EmbeddingBuildStats]:
    by_question: dict[str, dict] = {}
    for capability in capabilities:
        question_id = capability.get("questionId")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Capability record has no questionId")
        if question_id in by_question:
            raise ValueError(f"Duplicate questionId: {question_id}")
        if capability.get("capabilityVersion") != "capability-v1":
            raise ValueError(f"Unsupported capabilityVersion for {question_id}")
        question = capability.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question text is empty: {question_id}")
        by_question[question_id] = capability

    pending_ids: list[str] = []
    vectors_by_id: dict[str, list[float]] = {}
    for question_id in sorted(by_question):
        question = by_question[question_id]["question"]
        digest = text_hash(question)
        existing = existing_by_question_id.get(question_id)
        if not force and existing and existing.get("embeddingVersion") == embedding_version and _valid_existing(existing, digest, model, base_url):
            vectors_by_id[question_id] = [float(value) for value in existing["embedding"]]
        else:
            pending_ids.append(question_id)

    generated_vectors: list[list[float]] = []
    generated_dimension = 0
    if pending_ids:
        texts = [by_question[question_id]["question"] for question_id in pending_ids]
        generated_vectors, generated_dimension = embedder.embed(texts, batch_size=batch_size)
        if len(generated_vectors) != len(pending_ids):
            raise ValueError("Embedding client returned wrong number of vectors")
        vectors_by_id.update(dict(zip(pending_ids, generated_vectors, strict=True)))

    dimensions = {len(vector) for vector in vectors_by_id.values()}
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions are inconsistent: {sorted(dimensions)}")
    dimension = dimensions.pop() if dimensions else generated_dimension
    if generated_dimension and dimension != generated_dimension:
        raise ValueError("Generated dimension does not match existing embeddings")

    records: list[EmbeddingRecord] = []
    for question_id in sorted(by_question):
        capability = by_question[question_id]
        question = capability["question"]
        records.append(EmbeddingRecord(
            questionId=question_id,
            embeddingVersion=embedding_version,
            model=model,
            baseUrl=base_url.rstrip("/"),
            dimension=dimension,
            textHash=text_hash(question),
            question=question,
            tag=capability.get("tag"),
            mathType=capability.get("mathType"),
            difficulty=capability.get("difficulty"),
            capabilityVersion=capability["capabilityVersion"],
            localSuccessRate=float(capability["localSuccessRate"]),
            mathPassRate=capability.get("mathPassRate"),
            labelUsable=bool(capability["labelUsable"]),
            embedding=vectors_by_id[question_id],
        ))
    return records, EmbeddingBuildStats(
        generated=len(pending_ids),
        skipped=len(records) - len(pending_ids),
        dimension=dimension,
    )
