"""Capability dataset exports and aggregate reporting."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Iterable

from .models import CapabilityRecord


def write_jsonl(records: Iterable[CapabilityRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(by_alias=True), ensure_ascii=False) + "\n")


def write_csv(records: Iterable[CapabilityRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "questionId", "question", "tag", "mathType", "difficulty",
        "capabilityVersion", "evaluatorPipelineVersion", "attempts", "gradableAttempts",
        "mathCorrect", "mathIncorrect", "review", "unresolved",
        "runtimeCompleted", "runtimeTruncated", "runtimeFailed", "runtimeCancelled", "runtimeInterrupted",
        "runtimeSuccessRate", "mathPassRate", "localSuccessCount", "localSuccessRate",
        "labelUsable", "attemptDetails",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            payload = record.model_dump(by_alias=True)
            runtime = payload.pop("runtime")
            details = payload.pop("attemptDetails")
            payload.update({
                "runtimeCompleted": runtime["completed"],
                "runtimeTruncated": runtime["truncated"],
                "runtimeFailed": runtime["failed"],
                "runtimeCancelled": runtime["cancelled"],
                "runtimeInterrupted": runtime["interrupted"],
                "attemptDetails": json.dumps(details, ensure_ascii=False),
            })
            writer.writerow(payload)


def _group_stats(records: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key) or "unknown")].append(record)
    result: dict[str, dict] = {}
    for name, items in sorted(groups.items()):
        correct = sum(int(item["mathCorrect"]) for item in items)
        incorrect = sum(int(item["mathIncorrect"]) for item in items)
        attempts = sum(int(item["attempts"]) for item in items)
        local_success = sum(int(item["localSuccessCount"]) for item in items)
        result[name] = {
            "questions": len(items),
            "labelUsable": sum(bool(item["labelUsable"]) for item in items),
            "mathCorrect": correct,
            "mathIncorrect": incorrect,
            "review": sum(int(item["review"]) for item in items),
            "unresolved": sum(int(item["unresolved"]) for item in items),
            "mathPassRate": round(correct / (correct + incorrect), 7) if correct + incorrect else None,
            "localSuccessRate": round(local_success / attempts, 7) if attempts else None,
        }
    return result


def summarize(records: list[dict]) -> dict:
    pass_rates = Counter("null" if item.get("mathPassRate") is None else str(item["mathPassRate"]) for item in records)
    local_distribution = Counter(f"{int(item['localSuccessCount'])}/3" for item in records)
    return {
        "capabilityRecords": len(records),
        "totalAttempts": sum(int(item["attempts"]) for item in records),
        "uniqueRunIds": len({
            detail["runId"]
            for item in records
            for detail in item.get("attemptDetails", [])
        }),
        "labelUsable": sum(bool(item["labelUsable"]) for item in records),
        "questionsWithReview": sum(int(item["review"]) > 0 for item in records),
        "questionsWithUnresolved": sum(int(item["unresolved"]) > 0 for item in records),
        "mathPassRateDistribution": dict(sorted(pass_rates.items())),
        "localSuccessRateDistribution": {
            bucket: local_distribution.get(bucket, 0)
            for bucket in ("0/3", "1/3", "2/3", "3/3")
        },
        "byDifficulty": _group_stats(records, "difficulty"),
        "byMathType": _group_stats(records, "mathType"),
        "byTag": _group_stats(records, "tag"),
    }


def render_report(summary: dict) -> str:
    lines = [
        f"Capability Records: {summary['capabilityRecords']}",
        f"Total Attempts: {summary['totalAttempts']}",
        f"Unique Run IDs: {summary['uniqueRunIds']}",
        f"labelUsable: {summary['labelUsable']}",
        f"Questions with review: {summary['questionsWithReview']}",
        f"Questions with unresolved: {summary['questionsWithUnresolved']}",
        f"mathPassRate distribution: {json.dumps(summary['mathPassRateDistribution'], ensure_ascii=False, sort_keys=True)}",
        f"localSuccessRate distribution: {json.dumps(summary['localSuccessRateDistribution'], ensure_ascii=False)}",
    ]
    for heading, key in (
        ("By difficulty", "byDifficulty"),
        ("By mathType", "byMathType"),
        ("By tag", "byTag"),
    ):
        lines.append(f"{heading}:")
        for name, stats in summary[key].items():
            lines.append(f"  {name}: {json.dumps(stats, ensure_ascii=False, sort_keys=True)}")
    return "\n".join(lines)
