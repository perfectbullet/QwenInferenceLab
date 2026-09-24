from __future__ import annotations

import pytest

from qwen_inference_lab.router.evaluation import (
    DecisionRow,
    FeatureRow,
    calculate_metrics,
    cross_fitted_result,
    decisions_for_config,
    threshold_sweep,
)
from qwen_inference_lab.router.models import PolicyConfig

from .helpers import cv_records, features, neighbor, question


def row(question_id: str, success: float, fold: int, score: float, *, usable: bool = True) -> FeatureRow:
    record = question(question_id, success, usable=usable)
    value = features(
        neighborSuccessMean=score,
        weightedSuccess=score,
        weightedSuccessSquared=score,
    )
    return FeatureRow(
        record=record,
        fold=fold,
        features_by_k={3: value},
        neighbors_by_k={3: [
            neighbor("N1", 0.9, 1.0),
            neighbor("N2", 0.8, 1.0),
            neighbor("N3", 0.7, 1.0),
        ]},
    )


def config(threshold: float) -> PolicyConfig:
    return PolicyConfig(policy="neighbor_mean", k=3, scoreThreshold=threshold)


def test_unknown_label_is_excluded_from_ground_truth_metrics():
    rows = [
        row("safe", 1.0, 0, 0.9),
        row("unknown", 0.0, 0, 0.9, usable=False),
    ]
    metrics = calculate_metrics(decisions_for_config(rows, config(0.5)))
    assert metrics["queryCount"] == 2
    assert metrics["groundTruthCount"] == 1
    assert metrics["unknownCount"] == 1
    assert metrics["localPrecision"] == 1.0


def test_false_local_precision_and_coverage_metrics():
    rows = [
        row("tp", 1.0, 0, 0.9),
        row("fp", 2 / 3, 0, 0.9),
        row("fn", 1.0, 0, 0.1),
        row("tn", 0.0, 0, 0.1),
    ]
    metrics = calculate_metrics(decisions_for_config(rows, config(0.5)))
    assert metrics["falseLocalCount"] == 1
    assert metrics["falseLocalRate"] == pytest.approx(0.5)
    assert metrics["localPrecision"] == pytest.approx(0.5)
    assert metrics["localCoverage"] == pytest.approx(0.5)
    assert metrics["confusionMatrix"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_threshold_sweep_changes_local_coverage():
    rows = [
        row("safe", 1.0, 0, 0.9),
        row("risk", 0.0, 1, 0.6),
    ]
    sweep = threshold_sweep(rows, folds=2, configs=[config(0.5), config(0.8)])
    assert sweep[0]["metrics"]["localCoverage"] == 1.0
    assert sweep[1]["metrics"]["localCoverage"] == 0.5
    assert sweep[1]["metrics"]["falseLocalCount"] == 0


def test_cross_fitted_selection_never_uses_current_fold_metrics():
    rows = [
        row("safe0", 1.0, 0, 0.9),
        row("risk0", 0.0, 0, 0.7),
        row("safe1", 1.0, 1, 0.9),
        row("risk1", 0.0, 1, 0.4),
    ]
    sweep = threshold_sweep(rows, folds=2, configs=[config(0.5), config(0.8)])
    summary, decisions = cross_fitted_result(
        rows,
        sweep,
        folds=2,
        target_precision=1.0,
    )
    assert len(decisions) == 4
    assert summary["selectedConfigs"][0]["selectionMetricsOnOtherFolds"]["falseLocalCount"] == 0
    assert summary["selectedConfigs"][1]["selectionMetricsOnOtherFolds"]["falseLocalCount"] == 0
    assert all(
        item["selectedConfig"]["scoreThreshold"] == 0.8 for item in summary["selectedConfigs"]
    )

def test_nested_cv_selects_only_inside_outer_training_corpus():
    from qwen_inference_lab.router.dataset import stratified_partitions
    from qwen_inference_lab.router.evaluation import build_cv_features, nested_cv_candidates

    records = cv_records()
    partitions = stratified_partitions(records, folds=5, seed=42)
    _, outer_rows = build_cv_features(records, folds=5, seed=42)
    results = nested_cv_candidates(
        records,
        partitions,
        outer_rows,
        folds=5,
        seed=42,
        configs=[
            PolicyConfig(
                policy=policy,
                k=k,
                scoreThreshold=0.5,
                oodThreshold=0.0,
            )
            for policy in ("neighbor_mean", "weighted_knn", "knn_ood", "failure_aware")
            for k in (3, 5, 10)
        ],
    )
    selected = results["selection-target:0.95"]["summary"]["selectedConfigs"]
    assert len(selected) == 5
    assert all(item["outerTrainingCount"] == 16 for item in selected)
    assert all(item["innerFolds"] == 4 for item in selected)
