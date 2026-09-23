from types import SimpleNamespace
import pytest
from qwen_inference_lab.evaluator import llm_judge

class FakeCompletions:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
    def create(self, **kwargs):
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])

class FakeOpenAI:
    completions = None
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=self.completions)

def install(monkeypatch, content=None, error=None):
    FakeOpenAI.completions = FakeCompletions(content, error)
    monkeypatch.setattr(llm_judge, "OpenAI", FakeOpenAI)

def config():
    return {"baseUrl": "https://judge.invalid/v1", "modelName": "judge", "apiKey": "secret"}

def test_normal_json(monkeypatch):
    install(monkeypatch, "{\"verdict\":\"correct\",\"complete\":true,\"hasCriticalError\":false,\"confidence\":0.95,\"reason\":\"ok\",\"issues\":[]}")
    result = llm_judge.judge_answer(mode="semantic", question="q", reference="r", candidate="c", config=config(), retries=0)
    assert result.verdict == "correct"

def test_invalid_json_is_unresolved(monkeypatch):
    install(monkeypatch, "not json")
    assert llm_judge.judge_answer(mode="semantic", question="q", reference="r", candidate="c", config=config(), retries=0).verdict == "unresolved"

def test_request_failure_is_unresolved(monkeypatch):
    install(monkeypatch, error=TimeoutError())
    assert llm_judge.judge_answer(mode="semantic", question="q", reference="r", candidate="c", config=config(), retries=0).verdict == "unresolved"

def test_low_confidence_becomes_review(monkeypatch):
    install(monkeypatch, "{\"verdict\":\"correct\",\"complete\":true,\"hasCriticalError\":false,\"confidence\":0.4,\"reason\":\"unsure\",\"issues\":[]}")
    assert llm_judge.judge_answer(mode="semantic", question="q", reference="r", candidate="c", config=config(), retries=0).verdict == "review"

def test_missing_config_is_unresolved():
    assert llm_judge.judge_answer(mode="semantic", question="q", reference="r", candidate="c", config=None).reason_code == "JUDGE_UNAVAILABLE"
