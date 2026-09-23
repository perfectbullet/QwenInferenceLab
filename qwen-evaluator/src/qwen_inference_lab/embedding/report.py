"""Embedding dataset and retrieval report outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .models import EmbeddingRecord, Neighborhood


def write_embedding_jsonl(records: Iterable[EmbeddingRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(by_alias=True), ensure_ascii=False) + "\n")


def write_embedding_csv(records: Iterable[EmbeddingRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "questionId", "embeddingVersion", "model", "baseUrl", "dimension", "textHash",
        "question", "tag", "mathType", "difficulty", "capabilityVersion",
        "localSuccessRate", "mathPassRate", "labelUsable",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            payload = record.model_dump(by_alias=True)
            payload.pop("embedding")
            writer.writerow(payload)


def write_retrieval_artifacts(
    summary: dict,
    neighborhoods: list[Neighborhood],
    *,
    summary_path: Path,
    neighborhoods_path: Path,
    non_perfect_path: Path,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    neighborhoods_path.parent.mkdir(parents=True, exist_ok=True)
    non_perfect_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with neighborhoods_path.open("w", encoding="utf-8") as handle:
        for item in neighborhoods:
            handle.write(json.dumps(item.model_dump(by_alias=True), ensure_ascii=False) + "\n")
    with non_perfect_path.open("w", encoding="utf-8") as handle:
        for item in neighborhoods:
            if item.local_success_rate < 1.0:
                payload = item.model_dump(by_alias=True)
                payload.pop("top10")
                payload.pop("top10UsableNeighbors")
                payload.pop("top10LocalSuccessMean")
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def render_evaluation(summary: dict) -> str:
    return "\n".join([
        f"Questions: {summary['questions']}",
        f"Dimension: {summary['dimension']}",
        f"NaN vectors: {summary['nanVectors']}",
        f"Empty vectors: {summary['emptyVectors']}",
        f"Leave-One-Out self matches: {summary['leaveOneOutSelfMatches']}",
        f"Top1 similarity: {json.dumps(summary['similarityDistribution']['top1'], sort_keys=True)}",
        f"Top5 mean similarity: {json.dumps(summary['similarityDistribution']['top5Mean'], sort_keys=True)}",
        f"Top10 mean similarity: {json.dumps(summary['similarityDistribution']['top10Mean'], sort_keys=True)}",
        f"Top5 MathType Match Rate: {summary['top5MathTypeMatchRate']}",
        f"Top10 MathType Match Rate: {summary['top10MathTypeMatchRate']}",
        f"Top5 Tag Match Rate: {summary['top5TagMatchRate']}",
        f"Top10 Tag Match Rate: {summary['top10TagMatchRate']}",
        f"Top5 Difficulty Match Rate: {summary['top5DifficultyMatchRate']}",
        f"Top10 Difficulty Match Rate: {summary['top10DifficultyMatchRate']}",
        f"Non-perfect Questions: {summary['nonPerfectQuestions']}",
    ])
