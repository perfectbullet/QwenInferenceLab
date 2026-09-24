from __future__ import annotations

from fastapi.testclient import TestClient

from qwen_inference_lab.router.models import RouterQuestion
from qwen_inference_lab.router.runtime import RouterRuntime
from qwen_inference_lab.router.service import create_app


class FakeEmbedder:
    model = "embedding-test"
    base_url = "http://embedding.test/v1"

    def embed(self, texts: list[str], *, batch_size: int = 32):
        return [[1.0, 0.0] for _ in texts], 2


def runtime_factory():
    records = [
        RouterQuestion(
            questionId=f"Q{index:02d}",
            question=f"question {index}",
            embedding=[1.0, index / 100],
            dimension=2,
            localSuccessRate=1.0,
            labelUsable=True,
        )
        for index in range(12)
    ]
    return RouterRuntime(records, FakeEmbedder()), lambda: None


def test_health_and_preview_api():
    with TestClient(create_app(runtime_factory)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["corpusSize"] == 12

        response = client.post("/preview", json={
            "question": "new question",
            "profile": "development",
        })
        assert response.status_code == 200
        assert response.json()["decision"] == "local"
        assert response.json()["experimental"] is True


def test_preview_validates_input():
    with TestClient(create_app(runtime_factory)) as client:
        response = client.post("/preview", json={
            "question": "",
            "profile": "development",
        })
        assert response.status_code == 422
