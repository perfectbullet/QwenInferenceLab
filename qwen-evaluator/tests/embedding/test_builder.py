import pytest

from qwen_inference_lab.embedding.builder import build_embedding_records, text_hash


class Embedder:
    def __init__(self):
        self.calls = []
    def embed(self, texts, batch_size=32):
        self.calls.append((list(texts), batch_size))
        return [[float(index + 1), 1.0] for index in range(len(texts))], 2


def capability(question="question", question_id="Q1"):
    return {
        "questionId": question_id,
        "question": question,
        "tag": "tag",
        "mathType": "algebra",
        "difficulty": "medium",
        "capabilityVersion": "capability-v1",
        "localSuccessRate": 2 / 3,
        "mathPassRate": 2 / 3,
        "labelUsable": True,
    }


def existing(question="question"):
    return {
        "questionId": "Q1",
        "embeddingVersion": "bge-m3-v1",
        "model": "BAAI/bge-m3",
        "baseUrl": "http://embedding.test/v1",
        "dimension": 2,
        "textHash": text_hash(question),
        "embedding": [1.0, 0.0],
    }


def test_same_text_hash_skips_embedding_generation():
    embedder = Embedder()
    records, stats = build_embedding_records(
        [capability()], {"Q1": existing()}, embedder,
        model="BAAI/bge-m3", base_url="http://embedding.test/v1",
    )
    assert embedder.calls == []
    assert stats.generated == 0
    assert stats.skipped == 1
    assert records[0].embedding == [1.0, 0.0]


def test_question_change_triggers_embedding_generation():
    embedder = Embedder()
    records, stats = build_embedding_records(
        [capability("changed")], {"Q1": existing()}, embedder,
        model="BAAI/bge-m3", base_url="http://embedding.test/v1",
    )
    assert embedder.calls == [(["changed"], 32)]
    assert stats.generated == 1
    assert records[0].text_hash == text_hash("changed")


def test_duplicate_question_id_is_rejected():
    with pytest.raises(ValueError, match="Duplicate questionId"):
        build_embedding_records(
            [capability(), capability()],
            {}, Embedder(), model="BAAI/bge-m3", base_url="http://embedding.test/v1",
        )
