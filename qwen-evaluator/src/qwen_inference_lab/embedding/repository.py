"""MongoDB persistence for question embeddings."""

from __future__ import annotations

from datetime import UTC, datetime
from pymongo.database import Database

from . import EMBEDDING_VERSION
from .models import EmbeddingRecord


class EmbeddingRepository:
    def __init__(self, database: Database):
        self.capabilities = database["question_capabilities"]
        self.embeddings = database["question_embeddings"]

    def ensure_indexes(self) -> None:
        self.embeddings.create_index(
            [("questionId", 1), ("embeddingVersion", 1)],
            unique=True,
        )

    def capability_records(self, capability_version: str = "capability-v1") -> list[dict]:
        return list(self.capabilities.find(
            {"capabilityVersion": capability_version},
            {"_id": 0},
        ).sort("questionId", 1))

    def existing_embeddings(self, question_ids: list[str], embedding_version: str = EMBEDDING_VERSION) -> dict[str, dict]:
        documents = self.embeddings.find(
            {"questionId": {"$in": question_ids}, "embeddingVersion": embedding_version},
            {"_id": 0},
        )
        return {document["questionId"]: document for document in documents}

    def upsert_records(self, records: list[EmbeddingRecord], *, force: bool = False) -> dict[str, int]:
        counts = {"inserted": 0, "updated": 0, "skipped": 0}
        now = datetime.now(UTC)
        for record in records:
            query = {"questionId": record.question_id, "embeddingVersion": record.embedding_version}
            existing = self.embeddings.find_one(query, {"_id": 0, "textHash": 1, "model": 1, "baseUrl": 1})
            unchanged = (
                existing
                and existing.get("textHash") == record.text_hash
                and existing.get("model") == record.model
                and str(existing.get("baseUrl", "")).rstrip("/") == record.base_url.rstrip("/")
            )
            if unchanged and not force:
                counts["skipped"] += 1
                continue
            document = record.model_dump(by_alias=True)
            self.embeddings.update_one(
                query,
                {"$set": {**document, "updatedAt": now}, "$setOnInsert": {"createdAt": now}},
                upsert=True,
            )
            counts["updated" if existing else "inserted"] += 1
        return counts

    def all_embeddings(self, embedding_version: str = EMBEDDING_VERSION) -> list[dict]:
        return list(self.embeddings.find(
            {"embeddingVersion": embedding_version},
            {"_id": 0},
        ).sort("questionId", 1))
