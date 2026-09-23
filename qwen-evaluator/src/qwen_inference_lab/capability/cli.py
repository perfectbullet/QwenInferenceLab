"""CLI for Capability Dataset Builder V1."""

from __future__ import annotations

from pathlib import Path
import typer

from qwen_inference_lab.common.mongodb import create_client, validate_read_only
from qwen_inference_lab.evaluator import PIPELINE_VERSION

from . import CAPABILITY_VERSION
from .builder import build_capability_records, load_batch_states
from .report import render_report, summarize, write_csv, write_jsonl
from .repository import CapabilityRepository

app = typer.Typer(help="Build question-level capability datasets from exact evaluated batches.")


def open_repository():
    client = create_client()
    database, validation = validate_read_only(client)
    return client, CapabilityRepository(database), validation


@app.command("build")
def build(
    batch_state: list[Path] = typer.Option(..., "--batch-state", exists=True, dir_okay=False),
    force: bool = typer.Option(False, "--force"),
    jsonl_output: Path = typer.Option(Path("artifacts/capability-dataset-v1.jsonl"), "--jsonl-output"),
    csv_output: Path = typer.Option(Path("artifacts/capability-dataset-v1.csv"), "--csv-output"),
) -> None:
    """Validate exact batches, persist capabilities, and export JSONL/CSV."""
    specs = load_batch_states(batch_state, expected_batch_size=250)
    run_ids = [run_id for spec in specs for run_id in spec.run_ids]
    if len(run_ids) != 750 or len(set(run_ids)) != 750:
        raise typer.BadParameter(f"Expected 750 unique runIds, got {len(set(run_ids))}")

    client, repository, validation = open_repository()
    try:
        runs = repository.get_runs(run_ids)
        question_ids = sorted({str(run.get("questionId")) for run in runs.values()})
        questions = repository.get_questions(question_ids)
        evaluations = repository.get_evaluations(run_ids, PIPELINE_VERSION)
        records = build_capability_records(
            specs,
            questions,
            runs,
            evaluations,
            expected_attempts=3,
            expected_questions=250,
            pipeline_version=PIPELINE_VERSION,
        )
        payloads = [record.model_dump(by_alias=True) for record in records]
        summary = summarize(payloads)
        if summary["capabilityRecords"] != 250 or summary["totalAttempts"] != 750 or summary["uniqueRunIds"] != 750:
            raise RuntimeError(f"Final validation failed: {summary}")

        repository.ensure_indexes()
        writes = repository.upsert_records(records, force=force)
        write_jsonl(records, jsonl_output)
        write_csv(records, csv_output)
        typer.echo(f"Connected database: {validation.database}")
        typer.echo(f"Capability version: {CAPABILITY_VERSION}")
        typer.echo(f"Evaluator pipeline: {PIPELINE_VERSION}")
        typer.echo(f"MongoDB writes: {writes}")
        typer.echo(f"JSONL: {jsonl_output.resolve()}")
        typer.echo(f"CSV: {csv_output.resolve()}")
        typer.echo(render_report(summary))
    finally:
        client.close()


@app.command("report")
def report() -> None:
    """Report persisted capability-v1 records."""
    client, repository, validation = open_repository()
    try:
        records = repository.all_records(CAPABILITY_VERSION)
        typer.echo(f"Connected database: {validation.database}")
        typer.echo(render_report(summarize(records)))
    finally:
        client.close()


if __name__ == "__main__":
    app()
