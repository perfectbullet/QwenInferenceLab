from qwen_inference_lab.capability.models import CapabilityRecord, RuntimeCounts
from qwen_inference_lab.capability.repository import CapabilityRepository


class Collection:
    def __init__(self):
        self.documents = {}
        self.indexes = []
    def create_index(self, keys, unique=False):
        self.indexes.append((keys, unique))
    def find_one(self, query, projection=None):
        return self.documents.get((query["questionId"], query["capabilityVersion"]))
    def update_one(self, query, update, upsert=False):
        key = (query["questionId"], query["capabilityVersion"])
        self.documents[key] = {**self.documents.get(key, {}), **update["$set"]}


class Database:
    def __init__(self):
        self.collections = {}
    def __getitem__(self, name):
        return self.collections.setdefault(name, Collection())


def record(question="question"):
    return CapabilityRecord(
        questionId="Q1", question=question, tag="tag", mathType="algebra", difficulty="medium",
        capabilityVersion="capability-v1", evaluatorPipelineVersion="evaluator-v1",
        attempts=3, gradableAttempts=3, mathCorrect=2, mathIncorrect=1, review=0, unresolved=0,
        runtime=RuntimeCounts(completed=3), runtimeSuccessRate=1.0, mathPassRate=0.6666667,
        localSuccessCount=2, localSuccessRate=0.6666667, labelUsable=True, attemptDetails=[],
    )


def test_unique_index_and_idempotent_upsert():
    database = Database()
    repository = CapabilityRepository(database)
    repository.ensure_indexes()
    assert database["question_capabilities"].indexes == [([("questionId", 1), ("capabilityVersion", 1)], True)]
    assert repository.upsert_records([record()]) == {"inserted": 1, "updated": 0, "skipped": 0}
    assert repository.upsert_records([record("changed")]) == {"inserted": 0, "updated": 0, "skipped": 1}
    assert len(database["question_capabilities"].documents) == 1
    assert repository.upsert_records([record("changed")], force=True) == {"inserted": 0, "updated": 1, "skipped": 0}
    assert len(database["question_capabilities"].documents) == 1
