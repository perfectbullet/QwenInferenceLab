from __future__ import annotations

import pytest

from qwen_inference_lab.router.features import compute_features, weighted_success
from qwen_inference_lab.router.models import PolicyConfig
from qwen_inference_lab.router.ood import passes_ood_gate
from qwen_inference_lab.router.policy import decide

from .helpers import features, neighbor


def test_weighted_success_for_similarity_and_squared_similarity():
    neighbors = [
        neighbor("A", 1.0, 1.0),
        neighbor("B", 0.5, 0.0),
    ]
    assert weighted_success(neighbors, 1) == pytest.approx(2 / 3)
    assert weighted_success(neighbors, 2) == pytest.approx(0.8)


def test_non_perfect_rate_ignores_unusable_neighbor():
    all_neighbors = [
        neighbor("safe", 0.9, 1.0),
        neighbor("risk", 0.8, 2 / 3),
        neighbor("unknown", 0.7, 0.0, usable=False),
        neighbor("safe2", 0.6, 1.0),
    ]
    result, _ = compute_features(all_neighbors, 3)
    assert result.usable_neighbor_count == 2
    assert result.non_perfect_neighbor_count == 1
    assert result.non_perfect_neighbor_rate == pytest.approx(0.5)


def test_nearest_non_perfect_and_safe_margin_use_full_reference_ranking():
    all_neighbors = [
        neighbor("safe", 0.92, 1.0),
        neighbor("unknown", 0.90, 0.0, usable=False),
        neighbor("risk", 0.84, 2 / 3),
        neighbor("safe2", 0.70, 1.0),
    ]
    result, _ = compute_features(all_neighbors, 2)
    assert result.nearest_non_perfect_similarity == pytest.approx(0.84)
    assert result.nearest_safe_similarity == pytest.approx(0.92)
    assert result.safe_vs_risk_margin == pytest.approx(0.08)


def test_ood_gate_and_policy_force_cloud():
    value = features(top1Similarity=0.69)
    assert passes_ood_gate(value, 0.70) is False
    config = PolicyConfig(
        policy="knn_ood",
        k=3,
        scoreThreshold=0.8,
        oodThreshold=0.70,
    )
    decision, score, reasons = decide(value, config)
    assert decision == "cloud"
    assert score == pytest.approx(0.9)
    assert reasons == ["OOD_LOW_TOP1_SIMILARITY"]


def test_failure_aware_policy_uses_risk_features():
    value = features(
        nonPerfectNeighborRate=0.4,
        nearestNonPerfectSimilarity=0.9,
        safeVsRiskMargin=-0.01,
    )
    config = PolicyConfig(
        policy="failure_aware",
        k=3,
        scoreThreshold=0.8,
        oodThreshold=0.7,
        maxNonPerfectRate=0.34,
        maxNearestNonPerfectSimilarity=0.85,
        minSafeVsRiskMargin=0.0,
    )
    decision, _, reasons = decide(value, config)
    assert decision == "cloud"
    assert set(reasons) == {
        "NON_PERFECT_NEIGHBOR_RATE_HIGH",
        "NEAREST_NON_PERFECT_TOO_SIMILAR",
        "SAFE_VS_RISK_MARGIN_LOW",
    }
