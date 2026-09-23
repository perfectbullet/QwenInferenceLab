import pytest

from qwen_inference_lab.embedding.retrieval import (
    build_neighborhoods,
    cosine_similarity,
    retrieve_one,
)


def record(question_id, vector, *, local=1.0, usable=True):
    return {
        "questionId": question_id,
        "dimension": 2,
        "embedding": vector,
        "mathType": "algebra",
        "difficulty": "medium",
        "tag": "tag",
        "localSuccessRate": local,
        "labelUsable": usable,
    }


def records():
    return [
        record("A", [1.0, 0.0]),
        record("B", [0.9, 0.1], local=0.0, usable=False),
        record("C", [0.5, 0.5], local=0.25),
        record("D", [0.0, 1.0], local=0.75),
    ]


def test_cosine_similarity_is_correct():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0)


def test_leave_one_out_excludes_query():
    neighbors = retrieve_one(records(), "A", top_k=3, exclude_self=True)
    assert all(item.question_id != "A" for item in neighbors)


def test_top_k_is_sorted_by_similarity():
    neighbors = retrieve_one(records(), "A", top_k=3, exclude_self=True)
    assert [item.question_id for item in neighbors] == ["B", "C", "D"]
    assert [item.similarity for item in neighbors] == sorted([item.similarity for item in neighbors], reverse=True)


def test_unusable_neighbor_does_not_enter_capability_mean():
    neighborhood = next(item for item in build_neighborhoods(records()) if item.question_id == "A")
    assert neighborhood.top5[0].question_id == "B"
    assert neighborhood.top5[0].label_usable is False
    assert neighborhood.top5_usable_neighbors == 2
    assert neighborhood.top5_local_success_mean == 0.5
