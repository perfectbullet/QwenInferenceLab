"""MongoDB access with explicit, short-lived clients and safe validation."""

from __future__ import annotations

from dataclasses import dataclass

from pymongo import MongoClient
from pymongo.database import Database

from .config import MongoSettings, load_mongo_settings


@dataclass(frozen=True)
class DatabaseValidation:
    database: str
    questions: int
    runs: int


def create_client(settings: MongoSettings | None = None) -> MongoClient:
    settings = settings or load_mongo_settings()
    return MongoClient(
        settings.uri,
        appname="qwen-inference-lab-evaluator",
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=15000,
    )


def validate_read_only(client: MongoClient, settings: MongoSettings | None = None) -> tuple[Database, DatabaseValidation]:
    """Ping and count existing business collections before any evaluator write."""
    settings = settings or load_mongo_settings()
    database = client[settings.database]
    database.command("ping")
    validation = DatabaseValidation(
        database=database.name,
        questions=database["questions"].count_documents({}),
        runs=database["runs"].count_documents({}),
    )
    if validation.database != settings.database:
        raise RuntimeError("Connected database does not match MONGODB_DATABASE")
    return database, validation
