"""CLI for Embedding Dataset Builder and Retrieval Evaluation V1."""

from __future__ import annotations

import json
from pathlib import Path
import typer

from qwen_inference_lab.common.mongodb import create_client, validate_read_only

from . import DEFAULT_BASE_URL, DEFAULT_MODEL, EMBEDDING_VERSION
from .builder import build_embedding_records
from .client import EmbeddingClient
from .report import (
    render_evaluation,
    write_embedding_csv,
    write_embedding_jsonl,
    write_retrieval_artifacts,
)
from .repository import EmbeddingRepository
from .retrieval import evaluate_leave_one_out, retrieve_one

app = typer.Typer(help="Build Qwen3 embeddings and evaluate in-memory cosine retrieval.")


def open_repository():
    client = create_client()
    database, validation = validate_read_only(client)
    return client, EmbeddingRepository(database), validation


@app.command("build")
def build(
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
    model: str = typer.Option(DEFAULT_MODEL, "--model"),
    timeout: float = typer.Option(60, "--timeout", min=1),
    retries: int = typer.Option(2, "--retries", min=0, max=10),
    batch_size: int = typer.Option(32, "--batch-size", min=1, max=250),
    force: bool = typer.Option(False, "--force"),
    jsonl_output: Path = typer.Option(Path("artifacts/embedding-dataset-qwen3-embedding-0.6b-v1.jsonl"), "--jsonl-output"),
    csv_output: Path = typer.Option(Path("artifacts/embedding-dataset-qwen3-embedding-0.6b-v1.csv"), "--csv-output"),
) -> None:
    """Generate or reuse 250 embeddings and persist/export the dataset."""
    mongo_client, repository, validation = open_repository()
    try:
        capabilities = repository.capability_records("capability-v1")
        if len(capabilities) != 250:
            raise RuntimeError(f"Expected 250 capability records, got {len(capabilities)}")
        question_ids = [record["questionId"] for record in capabilities]
        if len(set(question_ids)) != 250:
            raise RuntimeError("Capability records contain duplicate questionId")
        existing = repository.existing_embeddings(question_ids, EMBEDDING_VERSION)
        with EmbeddingClient(base_url, model, timeout=timeout, retries=retries) as embedding_client:
            models = embedding_client.list_models()
            if model not in models:
                raise RuntimeError(f"Embedding model {model!r} not found; available={models}")
            records, build_stats = build_embedding_records(
                capabilities,
                existing,
                embedding_client,
                model=model,
                base_url=base_url,
                batch_size=batch_size,
                force=force,
            )
            request_stats = embedding_client.stats
        if len(records) != 250 or len({record.question_id for record in records}) != 250:
            raise RuntimeError("Embedding build did not produce 250 unique records")
        dimensions = {record.dimension for record in records}
        if len(dimensions) != 1:
            raise RuntimeError(f"Embedding dimensions differ: {dimensions}")
        repository.ensure_indexes()
        writes = repository.upsert_records(records, force=force)
        write_embedding_jsonl(records, jsonl_output)
        write_embedding_csv(records, csv_output)
        typer.echo(f"Connected database: {validation.database}")
        typer.echo(f"Embedding version: {EMBEDDING_VERSION}")
        typer.echo(f"Model: {model}")
        typer.echo(f"Base URL: {base_url.rstrip('/')}")
        typer.echo(f"Dimension: {build_stats.dimension}")
        typer.echo(f"Generated: {build_stats.generated} Skipped generation: {build_stats.skipped}")
        typer.echo(f"API requests: {request_stats.requests} retries: {request_stats.retries} failures: {request_stats.failures}")
        typer.echo(f"MongoDB writes: {writes}")
        typer.echo(f"JSONL: {jsonl_output.resolve()}")
        typer.echo(f"CSV: {csv_output.resolve()}")
    finally:
        mongo_client.close()


@app.command("retrieve")
def retrieve(
    question_id: str = typer.Option(..., "--question-id"),
    top_k: int = typer.Option(10, "--top-k", min=1, max=249),
) -> None:
    """Retrieve cosine-nearest questions while excluding the query itself."""
    mongo_client, repository, _ = open_repository()
    try:
        records = repository.all_embeddings(EMBEDDING_VERSION)
        neighbors = retrieve_one(records, question_id, top_k, exclude_self=True)
        usable = [item.local_success_rate for item in neighbors if item.label_usable]
        typer.echo(json.dumps({
            "questionId": question_id,
            "topK": top_k,
            "neighbors": [item.model_dump(by_alias=True) for item in neighbors],
            "usableNeighbors": len(usable),
            "neighborLocalSuccessMean": round(sum(usable) / len(usable), 7) if usable else None,
        }, ensure_ascii=False, indent=2))
    finally:
        mongo_client.close()


@app.command("evaluate")
def evaluate(
    summary_output: Path = typer.Option(Path("artifacts/retrieval-evaluation-qwen3-embedding-0.6b-v1.json"), "--summary-output"),
    neighborhoods_output: Path = typer.Option(Path("artifacts/retrieval-neighborhoods-qwen3-embedding-0.6b-v1.jsonl"), "--neighborhoods-output"),
    non_perfect_output: Path = typer.Option(Path("artifacts/non-perfect-retrieval-qwen3-embedding-0.6b-v1.jsonl"), "--non-perfect-output"),
) -> None:
    """Run leave-one-out Top-5/Top-10 retrieval evaluation."""
    mongo_client, repository, _ = open_repository()
    try:
        records = repository.all_embeddings(EMBEDDING_VERSION)
        if len(records) != 250:
            raise RuntimeError(f"Expected 250 embeddings, got {len(records)}")
        summary, neighborhoods = evaluate_leave_one_out(records)
        if summary["leaveOneOutSelfMatches"] != 0:
            raise RuntimeError("Leave-One-Out validation failed")
        write_retrieval_artifacts(
            summary,
            neighborhoods,
            summary_path=summary_output,
            neighborhoods_path=neighborhoods_output,
            non_perfect_path=non_perfect_output,
        )
        typer.echo(render_evaluation(summary))
        typer.echo(f"Summary: {summary_output.resolve()}")
        typer.echo(f"Neighborhoods: {neighborhoods_output.resolve()}")
        typer.echo(f"Non-perfect: {non_perfect_output.resolve()}")
    finally:
        mongo_client.close()


@app.command("report")
def report(
    summary_path: Path = typer.Option(Path("artifacts/retrieval-evaluation-qwen3-embedding-0.6b-v1.json"), "--summary-path", exists=True, dir_okay=False),
) -> None:
    """Print persisted embedding counts and the latest retrieval evaluation."""
    mongo_client, repository, validation = open_repository()
    try:
        records = repository.all_embeddings(EMBEDDING_VERSION)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        typer.echo(f"Connected database: {validation.database}")
        typer.echo(f"Embedding Records: {len(records)}")
        typer.echo(render_evaluation(summary))
    finally:
        mongo_client.close()


if __name__ == "__main__":
    app()
