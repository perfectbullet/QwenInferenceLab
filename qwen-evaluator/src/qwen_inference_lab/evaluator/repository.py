"""Persistence boundary for evaluator collections."""

from __future__ import annotations

from datetime import UTC, datetime
from pymongo.database import Database
from . import GOLD_ADAPTER_VERSION, PIPELINE_VERSION
from .models import GoldProfile


class EvaluatorRepository:
    def __init__(self, database: Database):
        self.database = database
        self.gold_profiles = database["question_gold_profiles"]
        self.evaluations = database["evaluations"]

    def ensure_indexes(self) -> None:
        self.gold_profiles.create_index([("questionId", 1), ("adapterVersion", 1)], unique=True)
        self.evaluations.create_index([("runId", 1), ("pipelineVersion", 1)], unique=True)

    def upsert_gold_profile(self, profile: GoldProfile) -> None:
        now = datetime.now(UTC)
        document = profile.model_dump(by_alias=True)
        self.gold_profiles.update_one({"questionId": profile.question_id, "adapterVersion": profile.adapter_version}, {"$set": {**document, "updatedAt": now}, "$setOnInsert": {"createdAt": now}}, upsert=True)

    def questions(self):
        return self.database["questions"].find({}, {"_id": 0})

    def get_question(self, question_id: str) -> dict | None:
        return self.database["questions"].find_one({"id": question_id}, {"_id": 0})

    def get_run(self, run_id: str) -> dict | None:
        return self.database["runs"].find_one({"id": run_id}, {"_id": 0})

    def get_gold_profile(self, question_id: str) -> dict | None:
        return self.gold_profiles.find_one({"questionId": question_id, "adapterVersion": GOLD_ADAPTER_VERSION}, {"_id": 0})

    def find_judge_configs(self, config_id: str | None = None) -> list[dict]:
        query = {"id": config_id} if config_id else {"modelName": "deepseek-ai/DeepSeek-V4-Flash"}
        return list(self.database["model_configs"].find(query, {"_id": 0}))

    def has_evaluation(self, run_id: str) -> bool:
        return self.evaluations.count_documents({"runId": run_id, "pipelineVersion": PIPELINE_VERSION}, limit=1) > 0

    def upsert_evaluation(self, document: dict) -> None:
        now = datetime.now(UTC)
        self.evaluations.update_one({"runId": document["runId"], "pipelineVersion": document["pipelineVersion"]}, {"$set": {**document, "updatedAt": now}, "$setOnInsert": {"createdAt": now}}, upsert=True)

    def evaluations_for_runs(self, run_ids: list[str]) -> list[dict]:
        return list(self.evaluations.find({"runId": {"$in": run_ids}, "pipelineVersion": PIPELINE_VERSION}, {"_id": 0}))

    def get_evaluation(self, run_id: str) -> dict | None:
        return self.evaluations.find_one({"runId": run_id, "pipelineVersion": PIPELINE_VERSION}, {"_id": 0})
