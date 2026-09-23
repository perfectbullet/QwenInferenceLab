from qwen_inference_lab.embedding.models import EmbeddingRecord
from qwen_inference_lab.embedding.repository import EmbeddingRepository


class Collection:
    def __init__(self):
        self.documents = {}
        self.indexes = []
    def create_index(self, keys, unique=False):
        self.indexes.append((keys, unique))
    def find_one(self, query, projection=None):
        return self.documents.get((query["questionId"], query["embeddingVersion"]))
    def update_one(self, query, update, upsert=False):
        key = (query["questionId"], query["embeddingVersion"])
        self.documents[key] = {**self.documents.get(key, {}), **update["$set"]}


class Database:
    def __init__(self):
        self.collections = {}
    def __getitem__(self, name):
        return self.collections.setdefault(name, Collection())


def record(text_hash="hash"):
    return EmbeddingRecord(
        questionId="Q1", embeddingVersion="bge-m3-v1", model="BAAI/bge-m3",
        baseUrl="http://embedding.test/v1", dimension=2, textHash=text_hash,
        question="question", tag="tag", mathType="algebra", difficulty="medium",
        capabilityVersion="capability-v1", localSuccessRate=1.0, mathPassRate=1.0,
        labelUsable=True, embedding=[1.0, 0.0],
    )


def test_mongodb_unique_index_and_idempotent_upsert():
    database = Database()
    repository = EmbeddingRepository(database)
    repository.ensure_indexes()
    assert database["question_embeddings"].indexes == [([("questionId", 1), ("embeddingVersion", 1)], True)]
    assert repository.upsert_records([record()]) == {"inserted": 1, "updated": 0, "skipped": 0}
    assert repository.upsert_records([record()]) == {"inserted": 0, "updated": 0, "skipped": 1}
    assert repository.upsert_records([record("changed")]) == {"inserted": 0, "updated": 1, "skipped": 0}
    assert len(database["question_embeddings"].documents) == 1
