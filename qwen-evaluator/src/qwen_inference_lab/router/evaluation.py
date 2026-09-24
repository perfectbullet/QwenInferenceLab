"""Stratified cross-validation, threshold sweeps, and safety metrics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from statistics import mean, pstdev
from typing import Iterable

from .dataset import CVPartition, stratified_partitions
from .features import compute_features
from .knn import retrieve_neighbors
from .models import (
    PolicyConfig,
    RouterFeatures,
    RouterNeighbor,
    RouterPrediction,
    RouterQuestion,
)
from .ood import OOD_THRESHOLDS
from .policy import decide


K_VALUES = (3, 5, 10)
SCORE_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
NON_PERFECT_RATE_THRESHOLDS = (0.0, 0.20, 0.34, 0.50)
NEAREST_RISK_THRESHOLDS = (0.75, 0.85, 0.95, 1.00)
MARGIN_THRESHOLDS = (-0.10, 0.0, 0.05)
PRECISION_TARGETS = (0.95, 0.97, 0.99)


@dataclass(frozen=True)
class FeatureRow:
    record: RouterQuestion
    fold: int
    features_by_k: dict[int, RouterFeatures]
    neighbors_by_k: dict[int, list[RouterNeighbor]]


@dataclass(frozen=True)
class DecisionRow:
    row: FeatureRow
    decision: str
    score: float | None
    reason_codes: list[str]
    config: PolicyConfig


def build_cv_features(
    records: list[RouterQuestion],
    folds: int,
    seed: int,
) -> tuple[list[CVPartition], list[FeatureRow]]:
    partitions = stratified_partitions(records, folds=folds, seed=seed)
    by_id = {record.question_id: record for record in records}
    rows: list[FeatureRow] = []
    for partition in partitions:
        queries = [by_id[question_id] for question_id in partition.query_ids]
        references = [by_id[question_id] for question_id in partition.reference_ids]
        reference_ids = {record.question_id for record in references}
        for query in queries:
            if query.question_id in reference_ids:
                raise AssertionError("Held-out query entered its reference corpus")
            all_neighbors = retrieve_neighbors(query, references)
            features_by_k: dict[int, RouterFeatures] = {}
            neighbors_by_k: dict[int, list[RouterNeighbor]] = {}
            for k in K_VALUES:
                features, neighbors = compute_features(all_neighbors, k)
                features_by_k[k] = features
                neighbors_by_k[k] = neighbors
            rows.append(FeatureRow(
                record=query,
                fold=partition.fold,
                features_by_k=features_by_k,
                neighbors_by_k=neighbors_by_k,
            ))
    if len(rows) != len(records) or len({row.record.question_id for row in rows}) != len(records):
        raise AssertionError("Cross-validation did not yield one prediction row per Question")
    return partitions, sorted(rows, key=lambda row: row.record.question_id)


def decisions_for_config(
    rows: Iterable[FeatureRow],
    config: PolicyConfig,
) -> list[DecisionRow]:
    decisions: list[DecisionRow] = []
    for row in rows:
        features = row.features_by_k[config.k]
        decision, score, reasons = decide(features, config)
        decisions.append(DecisionRow(
            row=row,
            decision=decision,
            score=score,
            reason_codes=reasons,
            config=config,
        ))
    return decisions


def metric_counts(decisions: Iterable[DecisionRow]) -> dict[str, int]:
    values = list(decisions)
    labeled = [item for item in values if item.row.record.safe_local is not None]
    tp = sum(item.decision == "local" and item.row.record.safe_local is True for item in labeled)
    fp = sum(item.decision == "local" and item.row.record.safe_local is False for item in labeled)
    tn = sum(item.decision == "cloud" and item.row.record.safe_local is False for item in labeled)
    fn = sum(item.decision == "cloud" and item.row.record.safe_local is True for item in labeled)
    return {
        "queryCount": len(values),
        "groundTruthCount": len(labeled),
        "positiveCount": tp + fn,
        "negativeCount": fp + tn,
        "unknownCount": len(values) - len(labeled),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def metrics_from_counts(counts: dict[str, int]) -> dict:
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    ground_truth = tp + fp + tn + fn
    local = tp + fp
    cloud = tn + fn
    positive = tp + fn
    negative = fp + tn
    return {
        **counts,
        "localDecisionCount": local,
        "cloudDecisionCount": cloud,
        "localPrecision": _ratio(tp, local),
        "localCoverage": _ratio(local, ground_truth),
        "falseLocalCount": fp,
        "falseLocalRate": _ratio(fp, negative),
        "cloudRate": _ratio(cloud, ground_truth),
        "safeLocalRecall": _ratio(tp, positive),
        "confusionMatrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def calculate_metrics(decisions: Iterable[DecisionRow]) -> dict:
    return metrics_from_counts(metric_counts(decisions))


def summarize_decisions(decisions: list[DecisionRow], folds: int) -> dict:
    pooled = calculate_metrics(decisions)
    fold_metrics: list[dict] = []
    for fold in range(folds):
        current = [item for item in decisions if item.row.fold == fold]
        fold_metrics.append({"fold": fold, **calculate_metrics(current)})
    aggregate: dict[str, dict[str, float | None]] = {}
    for field in (
        "localPrecision",
        "localCoverage",
        "falseLocalCount",
        "falseLocalRate",
        "cloudRate",
        "safeLocalRecall",
    ):
        values = [item[field] for item in fold_metrics if item[field] is not None]
        aggregate[field] = {
            "mean": round(mean(values), 8) if values else None,
            "std": round(pstdev(values), 8) if values else None,
        }
    return {"pooled": pooled, "foldMetrics": fold_metrics, "foldMeanStd": aggregate}


def policy_configs() -> list[PolicyConfig]:
    configs: list[PolicyConfig] = []
    for k in K_VALUES:
        for threshold in SCORE_THRESHOLDS:
            configs.append(PolicyConfig(
                policy="neighbor_mean", k=k, scoreThreshold=threshold,
            ))
            for power in (1, 2):
                configs.append(PolicyConfig(
                    policy="weighted_knn",
                    k=k,
                    scoreThreshold=threshold,
                    weightPower=power,
                ))
                for ood_threshold in OOD_THRESHOLDS:
                    configs.append(PolicyConfig(
                        policy="knn_ood",
                        k=k,
                        scoreThreshold=threshold,
                        weightPower=power,
                        oodThreshold=ood_threshold,
                    ))
        for threshold in SCORE_THRESHOLDS:
            for ood_threshold in OOD_THRESHOLDS:
                for max_rate in NON_PERFECT_RATE_THRESHOLDS:
                    for max_risk in NEAREST_RISK_THRESHOLDS:
                        for margin in MARGIN_THRESHOLDS:
                            configs.append(PolicyConfig(
                                policy="failure_aware",
                                k=k,
                                scoreThreshold=threshold,
                                weightPower=1,
                                oodThreshold=ood_threshold,
                                maxNonPerfectRate=max_rate,
                                maxNearestNonPerfectSimilarity=max_risk,
                                minSafeVsRiskMargin=margin,
                            ))
    return configs


def threshold_sweep(
    rows: list[FeatureRow],
    folds: int,
    configs: list[PolicyConfig] | None = None,
) -> list[dict]:
    results: list[dict] = []
    for config in configs or policy_configs():
        decisions = decisions_for_config(rows, config)
        summary = summarize_decisions(decisions, folds)
        results.append({
            "config": config.model_dump(by_alias=True),
            "metrics": summary["pooled"],
            "foldMetrics": summary["foldMetrics"],
        })
    return results


def cross_fitted_result(
    rows: list[FeatureRow],
    sweep: list[dict],
    folds: int,
    *,
    target_precision: float,
    policy: str | None = None,
    k: int | None = None,
) -> tuple[dict, list[DecisionRow]]:
    candidates = [
        item for item in sweep
        if (policy is None or item["config"]["policy"] == policy)
        and (k is None or item["config"]["k"] == k)
    ]
    if not candidates:
        raise ValueError("No policy configurations match selection filter")
    selected_decisions: list[DecisionRow] = []
    selected_configs: list[dict] = []
    for fold in range(folds):
        scored: list[tuple[dict, dict]] = []
        for item in candidates:
            training_folds = [value for value in item["foldMetrics"] if value["fold"] != fold]
            counts = _sum_metric_counts(training_folds)
            scored.append((item, metrics_from_counts(counts)))
        selected_item, training_metrics = _select_operating_point(scored, target_precision)
        config = PolicyConfig.model_validate(selected_item["config"])
        fold_rows = [row for row in rows if row.fold == fold]
        selected_decisions.extend(decisions_for_config(fold_rows, config))
        selected_configs.append({
            "fold": fold,
            "selectedConfig": config.model_dump(by_alias=True),
            "selectionMetricsOnOtherFolds": training_metrics,
        })
    summary = summarize_decisions(selected_decisions, folds)
    summary.update({
        "targetPrecision": target_precision,
        "selectionMethod": "cross-fitted: each fold configured using the other folds only",
        "selectedConfigs": selected_configs,
    })
    return summary, selected_decisions


def nested_cv_candidates(
    records: list[RouterQuestion],
    outer_partitions: list[CVPartition],
    outer_rows: list[FeatureRow],
    folds: int,
    seed: int,
    configs: list[PolicyConfig],
) -> dict[str, dict]:
    """Select policy parameters inside each outer training corpus only."""
    if folds < 3:
        raise ValueError("Nested CV requires at least 3 outer folds")
    by_id = {record.question_id: record for record in records}
    specs: list[tuple[str, float, str | None, int | None]] = []
    for policy in ("neighbor_mean", "weighted_knn", "knn_ood", "failure_aware"):
        for k in K_VALUES:
            specs.append((f"representative:{policy}:k={k}", 0.95, policy, k))
    for target in PRECISION_TARGETS:
        specs.append((f"selection-target:{target:.2f}", target, None, None))

    accumulators = {
        source: {"decisions": [], "selectedConfigs": []}
        for source, _, _, _ in specs
    }
    for partition in outer_partitions:
        outer_training = [by_id[question_id] for question_id in partition.reference_ids]
        inner_folds = folds - 1
        _, inner_rows = build_cv_features(
            outer_training,
            folds=inner_folds,
            seed=seed + 1000 + partition.fold,
        )
        scored: list[tuple[dict, dict]] = []
        for config in configs:
            metrics = calculate_metrics(decisions_for_config(inner_rows, config))
            scored.append(({"config": config.model_dump(by_alias=True)}, metrics))

        outer_fold_rows = [row for row in outer_rows if row.fold == partition.fold]
        for source, target, policy, k in specs:
            filtered = [
                item for item in scored
                if (policy is None or item[0]["config"]["policy"] == policy)
                and (k is None or item[0]["config"]["k"] == k)
            ]
            selected_item, selection_metrics = _select_operating_point(filtered, target)
            selected_config = PolicyConfig.model_validate(selected_item["config"])
            accumulator = accumulators[source]
            accumulator["decisions"].extend(
                decisions_for_config(outer_fold_rows, selected_config)
            )
            accumulator["selectedConfigs"].append({
                "fold": partition.fold,
                "selectedConfig": selected_config.model_dump(by_alias=True),
                "selectionMetricsOnInnerCV": selection_metrics,
                "outerTrainingCount": len(outer_training),
                "innerFolds": inner_folds,
            })

    results: dict[str, dict] = {}
    for source, target, _, _ in specs:
        accumulator = accumulators[source]
        summary = summarize_decisions(accumulator["decisions"], folds)
        summary.update({
            "targetPrecision": target,
            "selectionMethod": (
                "nested CV: each outer fold configured by inner CV on its "
                "outer training corpus only"
            ),
            "selectedConfigs": accumulator["selectedConfigs"],
        })
        results[source] = {
            "source": source,
            "summary": summary,
            "decisions": accumulator["decisions"],
        }
    return results


def exploratory_operating_point(sweep: list[dict], target_precision: float) -> dict | None:
    eligible = [
        item for item in sweep
        if item["metrics"]["localDecisionCount"] > 0
        and item["metrics"]["localPrecision"] is not None
        and item["metrics"]["localPrecision"] >= target_precision
    ]
    if not eligible:
        return None
    return max(eligible, key=_coverage_key)


def precision_coverage_frontier(sweep: list[dict]) -> list[dict]:
    by_point: dict[tuple[float, float], dict] = {}
    for item in sweep:
        metrics = item["metrics"]
        precision = metrics["localPrecision"]
        coverage = metrics["localCoverage"]
        if precision is None or metrics["localDecisionCount"] == 0:
            continue
        key = (precision, coverage)
        current = by_point.get(key)
        if current is None or metrics["falseLocalCount"] < current["metrics"]["falseLocalCount"]:
            by_point[key] = item
    ordered = sorted(
        by_point.values(),
        key=lambda item: (-item["metrics"]["localCoverage"], -item["metrics"]["localPrecision"]),
    )
    frontier: list[dict] = []
    best_precision = -1.0
    for item in ordered:
        precision = item["metrics"]["localPrecision"]
        if precision > best_precision:
            frontier.append(item)
            best_precision = precision
    return sorted(frontier, key=lambda item: item["metrics"]["localPrecision"])


def predictions(decisions: Iterable[DecisionRow]) -> list[RouterPrediction]:
    return [
        RouterPrediction(
            questionId=item.row.record.question_id,
            fold=item.row.fold,
            localSuccessRate=item.row.record.local_success_rate,
            labelUsable=item.row.record.label_usable,
            safeLocal=item.row.record.safe_local,
            decision=item.decision,
            score=item.score,
            features=item.row.features_by_k[item.config.k],
            neighbors=item.row.neighbors_by_k[item.config.k],
            reasonCodes=item.reason_codes,
        )
        for item in decisions
    ]


def _select_operating_point(
    scored: list[tuple[dict, dict]],
    target_precision: float,
) -> tuple[dict, dict]:
    eligible = [
        item for item in scored
        if item[1]["localDecisionCount"] > 0
        and item[1]["localPrecision"] is not None
        and item[1]["localPrecision"] >= target_precision
    ]
    pool = eligible or [
        item for item in scored
        if item[1]["localDecisionCount"] > 0 and item[1]["localPrecision"] is not None
    ]
    if not pool:
        raise RuntimeError("Every policy configuration abstains on every labeled query")
    if eligible:
        return max(
            pool,
            key=lambda item: _coverage_key({
                "metrics": item[1],
                "config": item[0]["config"],
            }),
        )
    return max(
        pool,
        key=lambda item: (
            item[1]["localPrecision"],
            item[1]["localCoverage"],
            -item[1]["falseLocalCount"],
            _config_tie_breaker(item[0]["config"]),
        ),
    )


def _coverage_key(item: dict) -> tuple:
    metrics = item["metrics"]
    config = item.get("config", {})
    return (
        metrics["localCoverage"],
        -metrics["falseLocalCount"],
        metrics["localPrecision"],
        _config_tie_breaker(config),
    )


def _config_tie_breaker(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _sum_metric_counts(metrics: list[dict]) -> dict[str, int]:
    fields = (
        "queryCount",
        "groundTruthCount",
        "positiveCount",
        "negativeCount",
        "unknownCount",
        "tp",
        "fp",
        "tn",
        "fn",
    )
    return {field: sum(int(item[field]) for item in metrics) for field in fields}


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 8) if denominator else None
