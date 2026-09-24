"""Online shadow Router runtime backed by the shared MongoDB corpus."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Protocol

from qwen_inference_lab.embedding import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    EMBEDDING_VERSION,
)
from qwen_inference_lab.embedding.client import EmbeddingClient

from . import ROUTER_VERSION
from .features import compute_features
from .knn import retrieve_neighbors
from .models import PolicyConfig, RouterQuestion
from .policy import decide


PROFILE_CONFIGS: dict[str, PolicyConfig] = {
    "development": PolicyConfig(
        policy="knn_ood",
        k=10,
        scoreThreshold=1.0,
        weightPower=1,
        oodThreshold=0.6,
    ),
    "conservative": PolicyConfig(
        policy="knn_ood",
        k=3,
        scoreThreshold=0.9,
        weightPower=1,
        oodThreshold=0.9,
    ),
}


class Embedder(Protocol):
    model: str
    base_url: str

    def embed(
        self, texts: list[str], *, batch_size: int = 32
    ) -> tuple[list[list[float]], int]: ...


@dataclass(frozen=True)
class RuntimeMetadata:
    corpus_size: int
    dimension: int
    embedding_version: str = EMBEDDING_VERSION
    router_version: str = ROUTER_VERSION


class RouterRuntime:
    """Generate an embedding and apply one fixed shadow-routing profile."""

    def __init__(
        self,
        records: list[RouterQuestion],
        embedder: Embedder,
    ) -> None:
        if not records:
            raise ValueError("Router reference corpus is empty")
        ids = [record.question_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Router reference corpus contains duplicate questionId")
        dimensions = {record.dimension for record in records}
        if len(dimensions) != 1:
            raise ValueError("Router reference embedding dimensions differ")
        self.records = sorted(records, key=lambda record: record.question_id)
        self.by_id = {record.question_id: record for record in self.records}
        self.embedder = embedder
        self.metadata = RuntimeMetadata(
            corpus_size=len(records),
            dimension=dimensions.pop(),
        )

    def preview(
        self,
        question: str,
        *,
        profile: str = "development",
        exclude_question_id: str | None = None,
    ) -> dict:
        text = question.strip()
        if not text:
            raise ValueError("question must not be empty")
        if profile not in PROFILE_CONFIGS:
            raise ValueError(f"Unknown Router profile: {profile}")
        if exclude_question_id is not None and exclude_question_id not in self.by_id:
            raise ValueError(f"Unknown questionId: {exclude_question_id}")

        started = time.perf_counter()
        vectors, dimension = self.embedder.embed([text], batch_size=1)
        if len(vectors) != 1 or dimension != self.metadata.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected "
                f"{self.metadata.dimension}, got {dimension}"
            )
        references = [
            record for record in self.records
            if record.question_id != exclude_question_id
        ]
        config = PROFILE_CONFIGS[profile]
        if len(references) < config.k:
            raise ValueError(
                f"Reference corpus needs at least {config.k} records"
            )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        query = RouterQuestion(
            questionId=f"preview:{digest}",
            question=text,
            embedding=vectors[0],
            dimension=dimension,
            localSuccessRate=0.0,
            labelUsable=False,
        )
        all_neighbors = retrieve_neighbors(query, references)
        features, neighbors = compute_features(all_neighbors, config.k)
        decision, score, reason_codes = decide(features, config)
        enriched_neighbors = []
        for neighbor in neighbors:
            reference = self.by_id[neighbor.question_id]
            enriched_neighbors.append({
                **neighbor.model_dump(by_alias=True),
                "question": reference.question,
                "mathType": reference.math_type,
                "difficulty": reference.difficulty,
                "tag": reference.tag,
            })
        return {
            "profile": profile,
            "decision": decision,
            "score": score,
            "experimental": True,
            "reasonCodes": reason_codes,
            "features": features.model_dump(by_alias=True),
            "neighbors": enriched_neighbors,
            "policyConfig": config.model_dump(by_alias=True),
            "routerVersion": ROUTER_VERSION,
            "embeddingVersion": EMBEDDING_VERSION,
            "embeddingModel": self.embedder.model,
            "corpusSize": len(references),
            "dimension": dimension,
            "textHash": digest,
            "durationMs": round((time.perf_counter() - started) * 1000, 2),
        }


def build_runtime_from_database(database) -> tuple[RouterRuntime, EmbeddingClient]:
    """Load the verified corpus from the same MongoDB used by the Node server."""
    from .dataset import load_router_dataset

    records, metadata = load_router_dataset(database)
    if metadata.embeddings != 250:
        raise RuntimeError(
            f"Expected 250 embedding records, got {metadata.embeddings}"
        )
    embedding_client = EmbeddingClient(
        DEFAULT_BASE_URL,
        DEFAULT_MODEL,
        timeout=60,
        retries=2,
    )
    return RouterRuntime(records, embedding_client), embedding_client
