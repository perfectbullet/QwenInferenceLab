from qwen_inference_lab.evaluator.models import EvaluationRoute, GoldProfile, GoldStatus
from qwen_inference_lab.evaluator.repository import EvaluatorRepository

class Collection:
    def __init__(self):
        self.indexes = []
        self.updates = []
    def create_index(self, keys, unique=False): self.indexes.append((keys, unique))
    def update_one(self, query, update, upsert=False): self.updates.append((query, update, upsert))

class Database:
    def __init__(self): self.collections = {}
    def __getitem__(self, name): return self.collections.setdefault(name, Collection())

def test_unique_indexes_and_gold_upsert():
    database = Database()
    repository = EvaluatorRepository(database)
    repository.ensure_indexes()
    assert database["question_gold_profiles"].indexes[0][1] is True
    assert database["evaluations"].indexes[0][1] is True
    profile = GoldProfile(questionId="Q1", adapterVersion="gold-v1", mathVerifyVersion="0.9.0", referenceHash="hash", questionType="single_expression", evaluationRoute=EvaluationRoute.MATH_VERIFY, status=GoldStatus.RESOLVED, rawReference="1", goldCandidates=["1"], parse={"success": True}, reasonCode="DIRECT_MATH_GOLD")
    repository.upsert_gold_profile(profile)
    query, update, upsert = database["question_gold_profiles"].updates[0]
    assert query == {"questionId": "Q1", "adapterVersion": "gold-v1"}
    assert upsert is True
    assert "$setOnInsert" in update
