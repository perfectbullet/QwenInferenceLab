"""Rule-based Router V1 policies."""

from __future__ import annotations

from .models import PolicyConfig, RouterFeatures
from .ood import passes_ood_gate


POLICIES = {
    "neighbor_mean",
    "weighted_knn",
    "knn_ood",
    "failure_aware",
}


def policy_score(features: RouterFeatures, config: PolicyConfig) -> float | None:
    if config.policy == "neighbor_mean":
        return features.neighbor_success_mean
    if config.weight_power == 2:
        return features.weighted_success_squared
    return features.weighted_success


def decide(features: RouterFeatures, config: PolicyConfig) -> tuple[str, float | None, list[str]]:
    if config.policy not in POLICIES:
        raise ValueError(f"Unknown policy: {config.policy}")
    if features.k != config.k:
        raise ValueError("Policy K does not match feature K")
    score = policy_score(features, config)
    reasons: list[str] = []

    if score is None:
        return "cloud", None, ["NO_USABLE_NEIGHBORS"]
    if score < config.score_threshold:
        reasons.append("SCORE_BELOW_THRESHOLD")

    if config.policy in {"knn_ood", "failure_aware"} and not passes_ood_gate(
        features, config.ood_threshold
    ):
        reasons.append("OOD_LOW_TOP1_SIMILARITY")

    if config.policy == "failure_aware":
        if (
            config.max_non_perfect_rate is not None
            and (
                features.non_perfect_neighbor_rate is None
                or features.non_perfect_neighbor_rate > config.max_non_perfect_rate
            )
        ):
            reasons.append("NON_PERFECT_NEIGHBOR_RATE_HIGH")
        if (
            config.max_nearest_non_perfect_similarity is not None
            and (
                features.nearest_non_perfect_similarity is None
                or features.nearest_non_perfect_similarity
                > config.max_nearest_non_perfect_similarity
            )
        ):
            reasons.append("NEAREST_NON_PERFECT_TOO_SIMILAR")
        if (
            config.min_safe_vs_risk_margin is not None
            and (
                features.safe_vs_risk_margin is None
                or features.safe_vs_risk_margin < config.min_safe_vs_risk_margin
            )
        ):
            reasons.append("SAFE_VS_RISK_MARGIN_LOW")

    if reasons:
        return "cloud", score, reasons
    return "local", score, ["LOCAL_THRESHOLDS_PASSED"]
