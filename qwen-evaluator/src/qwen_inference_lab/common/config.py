"""Safe loading of the repository MongoDB configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
from urllib.parse import quote_plus

from dotenv import load_dotenv


@dataclass(frozen=True)
class MongoSettings:
    uri: str
    database: str


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_mongo_settings() -> MongoSettings:
    """Load the root .env using the same connection rules as server/database.ts."""
    load_dotenv(repository_root() / ".env", override=False)
    database = os.getenv("MONGODB_DATABASE", "")
    if not database or re.search(r'[\s/\\."$]', database):
        raise ValueError("MONGODB_DATABASE is missing or invalid")
    explicit_uri = os.getenv("MONGODB_URI")
    if explicit_uri:
        return MongoSettings(uri=explicit_uri, database=database)
    username, password = os.getenv("MONGODB_USERNAME"), os.getenv("MONGODB_PASSWORD")
    if bool(username) != bool(password):
        raise ValueError("MONGODB_USERNAME and MONGODB_PASSWORD must be configured together")
    host = os.getenv("MONGODB_HOST", "127.0.0.1")
    if re.search(r"[\s/@?#]", host):
        raise ValueError("MONGODB_HOST is invalid")
    port = int(os.getenv("MONGODB_PORT", "27017"))
    if not 1 <= port <= 65535:
        raise ValueError("MONGODB_PORT is invalid")
    auth = ""
    if username and password:
        auth_source = os.getenv("MONGODB_AUTH_SOURCE", database)
        auth = f"{quote_plus(username)}:{quote_plus(password)}@"
        return MongoSettings(uri=f"mongodb://{auth}{host}:{port}/?authSource={quote_plus(auth_source)}", database=database)
    return MongoSettings(uri=f"mongodb://{host}:{port}", database=database)
