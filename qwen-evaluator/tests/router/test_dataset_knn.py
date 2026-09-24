from __future__ import annotations

import pytest

from qwen_inference_lab.router.dataset import stratified_partitions
from qwen_inference_lab.router.evaluation import build_cv_features
from qwen_inference_lab.router.knn import cosine_similarity, retrieve_neighbors

from .helpers import cv_records, question


def test_safe_local_label_definition_is_exact_three_of_three():
    assert question("safe", 1.0).safe_local is True
    assert question("risk", 2 / 3).safe_local is False
    assert question("unknown", 0.0, usable=False).safe_local is None


def test_self_neighbor_is_rejected():
    query = question("Q", 1.0)
    with pytest.raises(ValueError, match="leaked"):
        retrieve_neighbors(query, [query, question("N", 1.0, angle=0.2)])


def test_cosine_retrieval_is_sorted_correctly():
    query = question("Q", 1.0, angle=0.0)
    near = question("near", 1.0, angle=0.1)
    far = question("far", 1.0, angle=1.0)
    neighbors = retrieve_neighbors(query, [far, near])
    assert [item.question_id for item in neighbors] == ["near", "far"]
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_held_out_queries_never_enter_reference_corpus():
    partitions = stratified_partitions(cv_records(), folds=5, seed=42)
    for partition in partitions:
        assert set(partition.query_ids).isdisjoint(partition.reference_ids)


def test_stratified_folds_preserve_safe_risk_and_unknown_counts():
    records = cv_records()
    by_id = {record.question_id: record for record in records}
    for partition in stratified_partitions(records, folds=5, seed=42):
        queries = [by_id[question_id] for question_id in partition.query_ids]
        assert sum(item.safe_local is True for item in queries) == 2
        assert sum(item.safe_local is False for item in queries) == 1
        assert sum(item.safe_local is None for item in queries) == 1


def test_cv_is_deterministic():
    records = cv_records()
    first = stratified_partitions(records, folds=5, seed=42)
    second = stratified_partitions(records, folds=5, seed=42)
    assert first == second


def test_cv_features_contain_no_self_neighbor_or_cross_fold_label_leakage():
    records = cv_records()
    partitions, rows = build_cv_features(records, folds=5, seed=42)
    partition_by_fold = {item.fold: item for item in partitions}
    for row in rows:
        allowed = set(partition_by_fold[row.fold].reference_ids)
        for neighbors in row.neighbors_by_k.values():
            assert row.record.question_id not in {item.question_id for item in neighbors}
            assert {item.question_id for item in neighbors}.issubset(allowed)
