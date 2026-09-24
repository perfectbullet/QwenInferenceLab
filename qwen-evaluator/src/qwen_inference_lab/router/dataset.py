"""Load and split the capability/embedding dataset without label leakage."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from pymongo.database import Database

from . import CAPABILITY_VERSION, EMBEDDING_VERSION
from .models import RouterQuestion


@dataclass(frozen=True)
class DatasetMetadata:
    questions: int
    capabilities: int
    embeddings: int
    dimension: int
    label_usable: int
    safe_local: int
    unsafe_local: int
    unknown: int


@dataclass(frozen=True)
class CVPartition:
    fold: int
    query_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]


def load_router_dataset(database: Database) -> tuple[list[RouterQuestion], DatasetMetadata]:
    question_count = database["questions"].count_documents({})
    capabilities = list(database["question_capabilities"].find(
        {"capabilityVersion": CAPABILITY_VERSION},
        {"_id": 0},
    ))
    embeddings = list(database["question_embeddings"].find(
        {"embeddingVersion": EMBEDDING_VERSION},
        {"_id": 0},
    ))
    if question_count != 250:
        raise RuntimeError(f"Expected 250 Questions, got {question_count}")
    if len(capabilities) != 250:
        raise RuntimeError(f"Expected 250 capability records, got {len(capabilities)}")
    if len(embeddings) != 250:
        raise RuntimeError(f"Expected 250 embedding records, got {len(embeddings)}")

    capability_by_id = _unique_by_question_id(capabilities, "capability")
    embedding_by_id = _unique_by_question_id(embeddings, "embedding")
    if set(capability_by_id) != set(embedding_by_id):
        missing_embedding = sorted(set(capability_by_id) - set(embedding_by_id))
        missing_capability = sorted(set(embedding_by_id) - set(capability_by_id))
        raise RuntimeError(
            f"questionId mismatch: missing embeddings={missing_embedding}, "
            f"missing capabilities={missing_capability}"
        )

    records: list[RouterQuestion] = []
    dimensions: set[int] = set()
    for question_id in sorted(capability_by_id):
        capability = capability_by_id[question_id]
        embedding = embedding_by_id[question_id]
        vector = embedding.get("embedding")
        dimension = int(embedding.get("dimension", 0))
        if not isinstance(vector, list) or not vector or len(vector) != dimension:
            raise RuntimeError(f"Invalid embedding for {question_id}")
        if any(not math.isfinite(float(value)) for value in vector):
            raise RuntimeError(f"Non-finite embedding for {question_id}")
        dimensions.add(dimension)
        records.append(RouterQuestion(
            questionId=question_id,
            question=capability["question"],
            embedding=[float(value) for value in vector],
            dimension=dimension,
            localSuccessRate=float(capability["localSuccessRate"]),
            labelUsable=bool(capability["labelUsable"]),
            mathType=capability.get("mathType"),
            difficulty=capability.get("difficulty"),
            tag=capability.get("tag"),
        ))
    if len(dimensions) != 1:
        raise RuntimeError(f"Embedding dimensions differ: {sorted(dimensions)}")
    usable = [record for record in records if record.label_usable]
    safe = [record for record in usable if record.safe_local]
    metadata = DatasetMetadata(
        questions=question_count,
        capabilities=len(capabilities),
        embeddings=len(embeddings),
        dimension=dimensions.pop(),
        label_usable=len(usable),
        safe_local=len(safe),
        unsafe_local=len(usable) - len(safe),
        unknown=len(records) - len(usable),
    )
    return records, metadata


def _unique_by_question_id(documents: list[dict], kind: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for document in documents:
        question_id = document.get("questionId")
        if not isinstance(question_id, str) or not question_id:
            raise RuntimeError(f"{kind} record has no questionId")
        if question_id in result:
            raise RuntimeError(f"Duplicate {kind} questionId: {question_id}")
        result[question_id] = document
    return result


def stratified_partitions(
    records: list[RouterQuestion],
    folds: int = 5,
    seed: int = 42,
) -> list[CVPartition]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    groups: dict[str, list[str]] = {"safe": [], "unsafe": [], "unknown": []}
    for record in records:
        key = "unknown" if record.safe_local is None else ("safe" if record.safe_local else "unsafe")
        groups[key].append(record.question_id)
    if min(len(groups["safe"]), len(groups["unsafe"])) < folds:
        raise ValueError("Each labeled class must have at least one item per fold")

    randomizer = random.Random(seed)
    fold_ids: list[list[str]] = [[] for _ in range(folds)]
    for key in ("safe", "unsafe", "unknown"):
        values = sorted(groups[key])
        randomizer.shuffle(values)
        for index, question_id in enumerate(values):
            fold_ids[index % folds].append(question_id)

    all_ids = {record.question_id for record in records}
    partitions: list[CVPartition] = []
    seen: set[str] = set()
    for fold, query_values in enumerate(fold_ids):
        query_ids = tuple(sorted(query_values))
        query_set = set(query_ids)
        reference_ids = tuple(sorted(all_ids - query_set))
        if query_set.intersection(reference_ids):
            raise AssertionError("Query leaked into reference corpus")
        seen.update(query_set)
        partitions.append(CVPartition(fold=fold, query_ids=query_ids, reference_ids=reference_ids))
    if seen != all_ids:
        raise AssertionError("Cross-validation partitions do not cover all questions")
    return partitions
