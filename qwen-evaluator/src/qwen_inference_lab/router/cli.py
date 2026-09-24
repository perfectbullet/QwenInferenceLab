"""CLI for Router V1 KNN + OOD offline evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from qwen_inference_lab.common.mongodb import create_client, validate_read_only

from . import DEFAULT_FOLDS, DEFAULT_SEED, ROUTER_VERSION
from .dataset import load_router_dataset
from .evaluation import (
    K_VALUES,
    PRECISION_TARGETS,
    build_cv_features,
    nested_cv_candidates,
    decisions_for_config,
    exploratory_operating_point,
    policy_configs,
    precision_coverage_frontier,
    threshold_sweep,
    summarize_decisions,
)
from .models import PolicyConfig
from .report import (
    render_report,
    write_false_locals,
    write_non_perfect_analysis,
    write_results,
    write_sweep_csv,
)


DEFAULT_RESULTS = Path("artifacts/router-v1-offline-results.json")
DEFAULT_SWEEP = Path("artifacts/router-v1-threshold-sweep.csv")
DEFAULT_FALSE_LOCAL = Path("artifacts/router-v1-false-local.jsonl")
DEFAULT_NON_PERFECT = Path("artifacts/router-v1-non-perfect-analysis.jsonl")

app = typer.Typer(help="Offline-only KNN + OOD Router V1 research.")


def _load_dataset():
    client = create_client()
    database, validation = validate_read_only(client)
    records, metadata = load_router_dataset(database)
    return client, validation, records, metadata


@app.command("evaluate")
def evaluate(
    folds: int = typer.Option(DEFAULT_FOLDS, "--folds", min=2, max=10),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    results_output: Path = typer.Option(DEFAULT_RESULTS, "--results-output"),
    sweep_output: Path = typer.Option(DEFAULT_SWEEP, "--sweep-output"),
    false_local_output: Path = typer.Option(DEFAULT_FALSE_LOCAL, "--false-local-output"),
    non_perfect_output: Path = typer.Option(DEFAULT_NON_PERFECT, "--non-perfect-output"),
) -> None:
    """Run stratified cross-validation and the complete policy sweep."""
    client, validation, records, metadata = _load_dataset()
    try:
        partitions, rows = build_cv_features(records, folds=folds, seed=seed)
    finally:
        client.close()

    configs = policy_configs()
    sweep = threshold_sweep(rows, folds, configs)
    nested_by_source = nested_cv_candidates(
        records,
        partitions,
        rows,
        folds,
        seed,
        configs,
    )
    nested_candidates = list(nested_by_source.values())

    representative: dict[str, dict] = {}
    for policy in ("neighbor_mean", "weighted_knn", "knn_ood", "failure_aware"):
        representative[policy] = {
            str(k): nested_by_source[f"representative:{policy}:k={k}"]["summary"]
            for k in K_VALUES
        }

    target_results: dict[str, dict] = {}
    for target in PRECISION_TARGETS:
        candidate = nested_by_source[f"selection-target:{target:.2f}"]
        summary = candidate["summary"]
        actual_precision = summary["pooled"]["localPrecision"]
        target_results[f"{target:.2f}"] = {
            "achievable": (
                actual_precision is not None
                and summary["pooled"]["localDecisionCount"] > 0
                and actual_precision >= target
            ),
            "metrics": summary,
        }

    achieved_operating_points: dict[str, dict] = {}
    for target in PRECISION_TARGETS:
        eligible = [
            item for item in nested_candidates
            if item["summary"]["pooled"]["localDecisionCount"] > 0
            and item["summary"]["pooled"]["localPrecision"] is not None
            and item["summary"]["pooled"]["localPrecision"] >= target
        ]
        if not eligible:
            achieved_operating_points[f"{target:.2f}"] = {
                "achievable": False,
                "source": None,
                "metrics": None,
            }
            continue
        selected = max(
            eligible,
            key=lambda item: (
                item["summary"]["pooled"]["localCoverage"],
                -item["summary"]["pooled"]["falseLocalCount"],
            ),
        )
        achieved_operating_points[f"{target:.2f}"] = {
            "achievable": True,
            "source": selected["source"],
            "metrics": selected["summary"],
        }

    safest_candidate = min(
        (
            item for item in nested_candidates
            if item["summary"]["pooled"]["localDecisionCount"] > 0
        ),
        key=lambda item: (
            item["summary"]["pooled"]["falseLocalCount"],
            -item["summary"]["pooled"]["localCoverage"],
        ),
    )
    conservative_point = achieved_operating_points["0.95"]
    if conservative_point["achievable"]:
        conservative_candidate = nested_by_source[conservative_point["source"]]
        conservative_selection = (
            "maximum coverage among nested-CV candidates reaching "
            "95% pooled precision"
        )
    else:
        conservative_candidate = safest_candidate
        conservative_selection = (
            "95% precision was not achievable; using minimum False Local "
            "then maximum coverage"
        )
    development_config = PolicyConfig(
        policy="knn_ood", k=10, scoreThreshold=1.0,
        weightPower=1, oodThreshold=0.6,
    )
    recommended_decisions = decisions_for_config(rows, development_config)
    recommended_summary = summarize_decisions(recommended_decisions, folds)
    recommended_selection = "fixed development profile; exploratory OOF metrics"

    exploratory_targets: dict[str, dict | None] = {}
    for target in PRECISION_TARGETS:
        item = exploratory_operating_point(sweep, target)
        exploratory_targets[f"{target:.2f}"] = (
            {"config": item["config"], "metrics": item["metrics"]}
            if item is not None
            else None
        )

    usable_sweep = [
        item for item in sweep if item["metrics"]["localDecisionCount"] > 0
    ]
    least_false = min(
        usable_sweep,
        key=lambda item: (
            item["metrics"]["falseLocalCount"],
            -item["metrics"]["localCoverage"],
            -(item["metrics"]["localPrecision"] or 0),
        ),
    )
    frontier = precision_coverage_frontier(sweep)
    fold_payload = []
    by_id = {record.question_id: record for record in records}
    for partition in partitions:
        query_records = [by_id[question_id] for question_id in partition.query_ids]
        fold_payload.append({
            "fold": partition.fold,
            "queryCount": len(query_records),
            "referenceCount": len(partition.reference_ids),
            "positiveCount": sum(record.safe_local is True for record in query_records),
            "negativeCount": sum(record.safe_local is False for record in query_records),
            "unknownCount": sum(record.safe_local is None for record in query_records),
            "queryIds": list(partition.query_ids),
        })

    results = {
        "routerVersion": ROUTER_VERSION,
        "dataset": {
            "questions": metadata.questions,
            "capabilityRecords": metadata.capabilities,
            "embeddingRecords": metadata.embeddings,
            "embeddingDimension": metadata.dimension,
            "labelUsable": metadata.label_usable,
            "safeLocal": metadata.safe_local,
            "unsafeLocal": metadata.unsafe_local,
            "unknown": metadata.unknown,
        },
        "cv": {
            "folds": folds,
            "seed": seed,
            "stratifiedBy": "safeLocal with unknown labels distributed separately",
            "foldsDetail": fold_payload,
            "leakageChecks": {
                "queryExcludedFromReference": True,
                "featuresUseReferenceLabelsOnly": True,
                "nestedCVThresholdSelection": True,
            },
        },
        "sweep": {
            "configurationCount": len(sweep),
            "exploratoryOnly": True,
            "note": "Full OOF frontier and the development profile are exploratory; conservativeRecommended uses nested CV threshold selection.",
        },
        "representativeResults": representative,
        "precisionCoverageFrontier": [
            {"config": item["config"], "metrics": item["metrics"]}
            for item in frontier
        ],
        "exploratoryOperatingPoints": exploratory_targets,
        "nestedOperatingPoints": target_results,
        "achievedOperatingPoints": achieved_operating_points,
        "safestNestedPolicy": {
            "source": safest_candidate["source"],
            "metrics": safest_candidate["summary"],
            "selectionNote": "Minimum False Local, then maximum coverage among predeclared nested-CV candidates.",
        },
        "leastFalseLocalUsablePolicy": {
            "config": least_false["config"],
            "metrics": least_false["metrics"],
            "selectionNote": "Exploratory OOF: minimum FP, then maximum non-zero coverage.",
        },
        "recommended": {
            "profile": "development",
            "source": "fixed-development-profile",
            "config": development_config.model_dump(by_alias=True),
            "selection": recommended_selection,
            "metrics": recommended_summary,
        },
        "conservativeRecommended": {
            "target": 0.95,
            "targetAchieved": conservative_point["achievable"],
            "source": conservative_candidate["source"],
            "selection": conservative_selection,
            "metrics": conservative_candidate["summary"],
        },
        "metricDefinitions": {
            "localPrecision": "TP / (TP + FP)",
            "localCoverage": "(TP + FP) / labelUsable",
            "falseLocalRate": "FP / actual unsafe (FP + TN)",
            "cloudRate": "(TN + FN) / labelUsable",
            "safeLocalRecall": "TP / actual safe (TP + FN)",
        },
    }

    write_results(results, results_output)
    write_sweep_csv(sweep, sweep_output)
    false_count = write_false_locals(recommended_decisions, false_local_output)
    non_perfect_count = write_non_perfect_analysis(recommended_decisions, non_perfect_output)
    if false_count != recommended_summary["pooled"]["falseLocalCount"]:
        raise RuntimeError("False Local artifact count does not match recommended metrics")
    if non_perfect_count != 30:
        raise RuntimeError(f"Expected 30 non-perfect Questions, got {non_perfect_count}")

    typer.echo(f"Connected database: {validation.database}")
    typer.echo(render_report(results))
    typer.echo(f"False Local records: {false_count}")
    typer.echo(f"Non-perfect records: {non_perfect_count}")
    typer.echo(f"Results: {results_output.resolve()}")
    typer.echo(f"Sweep: {sweep_output.resolve()}")
    typer.echo(f"False Local: {false_local_output.resolve()}")
    typer.echo(f"Non-perfect: {non_perfect_output.resolve()}")


@app.command("report")
def report(
    results_path: Path = typer.Option(DEFAULT_RESULTS, "--results-path", exists=True, dir_okay=False),
) -> None:
    """Print the persisted Router V1 offline report."""
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    typer.echo(render_report(payload))


@app.command("inspect")
def inspect(
    question_id: str = typer.Option(..., "--question-id"),
    results_path: Path = typer.Option(DEFAULT_RESULTS, "--results-path", exists=True, dir_okay=False),
) -> None:
    """Inspect the cross-validated features and recommended decision for one Question."""
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    folds = int(payload["cv"]["folds"])
    seed = int(payload["cv"]["seed"])
    client, _, records, _ = _load_dataset()
    try:
        _, rows = build_cv_features(records, folds=folds, seed=seed)
    finally:
        client.close()
    row = next((item for item in rows if item.record.question_id == question_id), None)
    if row is None:
        raise typer.BadParameter(f"Unknown questionId: {question_id}")
    config = PolicyConfig.model_validate(payload["recommended"]["config"])
    decision = decisions_for_config([row], config)[0]
    output = {
        "questionId": row.record.question_id,
        "question": row.record.question,
        "fold": row.fold,
        "decision": decision.decision,
        "score": decision.score,
        "features": row.features_by_k[config.k].model_dump(by_alias=True),
        "neighbors": [
            item.model_dump(by_alias=True)
            for item in row.neighbors_by_k[config.k]
        ],
        "reasonCodes": decision.reason_codes,
        "policyConfig": config.model_dump(by_alias=True),
    }
    typer.echo(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
