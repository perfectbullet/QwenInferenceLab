"""MongoDB persistence for question capability records."""

from __future__ import annotations

from datetime import UTC, datetime
from pymongo.database import Database

from qwen_inference_lab.evaluator import PIPELINE_VERSION

from . import CAPABILITY_VERSION
from .models import CapabilityRecord


class CapabilityRepository:
    def __init__(self, database: Database):
        self.questions_collection = database["questions"]
        self.runs_collection = database["runs"]
        self.evaluations_collection = database["evaluations"]
        self.capabilities = database["question_capabilities"]

    def ensure_indexes(self) -> None:
        self.capabilities.create_index(
            [("questionId", 1), ("capabilityVersion", 1)],
            unique=True,
        )

    def get_questions(self, question_ids: list[str]) -> dict[str, dict]:
        documents = self.questions_collection.find({"id": {"$in": question_ids}}, {"_id": 0})
        return {document["id"]: document for document in documents}

    def get_runs(self, run_ids: list[str]) -> dict[str, dict]:
        documents = self.runs_collection.find({"id": {"$in": run_ids}}, {"_id": 0})
        return {document["id"]: document for document in documents}

    def get_evaluations(self, run_ids: list[str], pipeline_version: str = PIPELINE_VERSION) -> dict[str, dict]:
        documents = self.evaluations_collection.find(
            {"runId": {"$in": run_ids}, "pipelineVersion": pipeline_version},
            {"_id": 0},
        )
        return {document["runId"]: document for document in documents}

    def upsert_records(self, records: list[CapabilityRecord], *, force: bool = False) -> dict[str, int]:
        counts = {"inserted": 0, "updated": 0, "skipped": 0}
        now = datetime.now(UTC)
        for record in records:
            query = {"questionId": record.question_id, "capabilityVersion": record.capability_version}
            exists = self.capabilities.find_one(query, {"_id": 1}) is not None
            if exists and not force:
                counts["skipped"] += 1
                continue
            document = record.model_dump(by_alias=True)
            self.capabilities.update_one(
                query,
                {"$set": {**document, "updatedAt": now}, "$setOnInsert": {"createdAt": now}},
                upsert=True,
            )
            counts["updated" if exists else "inserted"] += 1
        return counts

    def all_records(self, capability_version: str = CAPABILITY_VERSION) -> list[dict]:
        return list(self.capabilities.find(
            {"capabilityVersion": capability_version},
            {"_id": 0},
        ).sort("questionId", 1))
