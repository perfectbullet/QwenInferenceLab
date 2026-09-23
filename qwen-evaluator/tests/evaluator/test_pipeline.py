from qwen_inference_lab.evaluator import pipeline
from qwen_inference_lab.evaluator.llm_judge import JudgeResponse, JudgeResult

class FakeRepository:
    def __init__(self, route="llm_multi_part", status="completed"):
        self.route = route
        self.status = status
        self.saved = None
    def has_evaluation(self, run_id): return False
    def get_evaluation(self, run_id): return None
    def get_run(self, run_id): return {"id": run_id, "questionId": "Q1", "status": self.status, "finishReason": "stop", "answer": "candidate"}
    def get_question(self, question_id): return {"id": question_id, "question": "(1) a (2) b", "reference_answer": "reference"}
    def get_gold_profile(self, question_id): return {"evaluationRoute": self.route, "rawReference": "reference"}
    def find_judge_configs(self, config_id=None): return [{"modelName": "judge", "baseUrl": "https://judge.invalid", "apiKey": "secret"}]
    def upsert_evaluation(self, document): self.saved = document

def incomplete():
    response = JudgeResponse(verdict="incorrect", complete=False, hasCriticalError=True, confidence=0.98, reason="missing", issues=[], allSubquestionsAnswered=False)
    return JudgeResult(verdict="incorrect", reason_code="INCOMPLETE_SUBQUESTIONS", response=response)

def test_multi_part_bypasses_math_verify(monkeypatch):
    repository = FakeRepository()
    monkeypatch.setattr(pipeline, "judge_multi_part", lambda *args, **kwargs: incomplete())
    _, document = pipeline.evaluate_run(repository, "R1")
    assert document["math"]["level"] == 2
    assert document["math"]["reasonCode"] == "INCOMPLETE_SUBQUESTIONS"

def test_runtime_failure_is_not_math_incorrect(monkeypatch):
    repository = FakeRepository(status="failed")
    _, document = pipeline.evaluate_run(repository, "R1")
    assert document["math"]["verdict"] == "unresolved"
    assert document["math"]["level"] == 0

def test_math_verify_route_temporarily_uses_semantic_judge(monkeypatch):
    repository = FakeRepository(route="math_verify", status="truncated")
    monkeypatch.setattr(pipeline, "judge_answer", lambda *args, **kwargs: incomplete())
    _, document = pipeline.evaluate_run(repository, "R1", max_level=3)
    assert document["math"]["method"] == "llm_semantic"
    assert document["math"]["level"] == 3
    assert document["verifyConfig"] is None
    assert document["runtime"]["status"] == "truncated"
