"""Artifacts and human-readable reporting for Router V1."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .evaluation import DecisionRow
from .models import RouterPrediction


SWEEP_FIELDS = [
    "policy",
    "k",
    "weightPower",
    "scoreThreshold",
    "oodThreshold",
    "maxNonPerfectRate",
    "maxNearestNonPerfectSimilarity",
    "minSafeVsRiskMargin",
    "localPrecision",
    "localCoverage",
    "falseLocalCount",
    "falseLocalRate",
    "cloudRate",
    "safeLocalRecall",
    "localDecisionCount",
]


def write_results(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_sweep_csv(sweep: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        for item in sweep:
            config = item["config"]
            metrics = item["metrics"]
            writer.writerow({
                field: config.get(field, metrics.get(field))
                for field in SWEEP_FIELDS
            })


def write_false_locals(decisions: list[DecisionRow], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in sorted(decisions, key=lambda value: value.row.record.question_id):
            record = item.row.record
            if not (record.safe_local is False and item.decision == "local"):
                continue
            payload = _prediction_payload(item)
            payload["question"] = record.question
            payload["policyConfig"] = item.config.model_dump(by_alias=True)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_non_perfect_analysis(decisions: list[DecisionRow], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in sorted(decisions, key=lambda value: value.row.record.question_id):
            record = item.row.record
            if record.local_success_rate >= 1.0:
                continue
            payload = _prediction_payload(item)
            payload.update({
                "question": record.question,
                "groundTruthStatus": (
                    "unknown" if record.safe_local is None else "unsafe"
                ),
                "policyConfig": item.config.model_dump(by_alias=True),
            })
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def _prediction_payload(item: DecisionRow) -> dict:
    prediction = RouterPrediction(
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
    return prediction.model_dump(by_alias=True)


def render_report(results: dict) -> str:
    dataset = results["dataset"]
    recommended = results["recommended"]["metrics"]["pooled"]
    lines = [
        f"Router version: {results['routerVersion']}",
        (
            f"Dataset: {dataset['questions']} questions, "
            f"{dataset['labelUsable']} labeled "
            f"({dataset['safeLocal']} safe / {dataset['unsafeLocal']} unsafe), "
            f"{dataset['unknown']} unknown"
        ),
        f"CV: {results['cv']['folds']}-fold stratified, seed={results['cv']['seed']}",
        f"Threshold configurations: {results['sweep']['configurationCount']}",
        "Recommended nested-CV result:",
        f"  Local Precision: {recommended['localPrecision']}",
        f"  Local Coverage: {recommended['localCoverage']}",
        f"  False Local: {recommended['falseLocalCount']}",
        f"  False Local Rate: {recommended['falseLocalRate']}",
        f"  Cloud Rate: {recommended['cloudRate']}",
        f"  Safe Local Recall: {recommended['safeLocalRecall']}",
    ]
    for target, point in results["achievedOperatingPoints"].items():
        status = "achievable" if point["achievable"] else "not achievable"
        metrics = point["metrics"]["pooled"] if point["metrics"] else {}
        lines.append(
            f"Precision target {target}: {status}; "
            f"actual precision={metrics.get('localPrecision')}, "
            f"coverage={metrics.get('localCoverage')}"
        )
    return "\n".join(lines)
