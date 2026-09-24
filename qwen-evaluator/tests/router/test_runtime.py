from __future__ import annotations

from qwen_inference_lab.router.models import RouterQuestion
from qwen_inference_lab.router.runtime import PROFILE_CONFIGS, RouterRuntime


class FakeEmbedder:
    model = "embedding-test"
    base_url = "http://embedding.test/v1"

    def __init__(self, vector: list[float]):
        self.vector = vector
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], *, batch_size: int = 32):
        self.calls.append(texts)
        return [self.vector for _ in texts], len(self.vector)


def corpus() -> list[RouterQuestion]:
    records = []
    for index in range(12):
        records.append(RouterQuestion(
            questionId=f"Q{index:02d}",
            question=f"question {index}",
            embedding=[1.0, index / 100],
            dimension=2,
            localSuccessRate=1.0 if index != 11 else 2 / 3,
            labelUsable=True,
            mathType="algebra",
            difficulty="medium",
            tag="test",
        ))
    return records


def test_runtime_development_preview_returns_explainable_decision():
    embedder = FakeEmbedder([1.0, 0.0])
    runtime = RouterRuntime(corpus(), embedder)

    result = runtime.preview(" new question ", profile="development")

    assert result["profile"] == "development"
    assert result["decision"] == "local"
    assert result["experimental"] is True
    assert result["dimension"] == 2
    assert len(result["neighbors"]) == PROFILE_CONFIGS["development"].k
    assert result["neighbors"][0]["question"] == "question 0"
    assert embedder.calls == [["new question"]]


def test_runtime_excludes_known_question_from_neighbors():
    runtime = RouterRuntime(corpus(), FakeEmbedder([1.0, 0.0]))

    result = runtime.preview(
        "question 0",
        profile="conservative",
        exclude_question_id="Q00",
    )

    assert all(item["questionId"] != "Q00" for item in result["neighbors"])
    assert result["corpusSize"] == 11


def test_runtime_rejects_unknown_profile_and_dimension_mismatch():
    runtime = RouterRuntime(corpus(), FakeEmbedder([1.0, 0.0]))
    try:
        runtime.preview("question", profile="invalid")
    except ValueError as error:
        assert "Unknown Router profile" in str(error)
    else:
        raise AssertionError("unknown profile was accepted")

    mismatch = RouterRuntime(corpus(), FakeEmbedder([1.0]))
    try:
        mismatch.preview("question")
    except ValueError as error:
        assert "dimension mismatch" in str(error)
    else:
        raise AssertionError("dimension mismatch was accepted")
