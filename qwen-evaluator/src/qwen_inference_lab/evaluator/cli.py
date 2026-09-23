"""Command-line interface for Python Evaluator V1."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import typer
from qwen_inference_lab.common.mongodb import create_client, validate_read_only
from .gold_adapter import build_gold_profile
from .pipeline import evaluate_run as run_pipeline
from .report import build_audit_manifest, load_batch_state, render_report, summarize
from .repository import EvaluatorRepository

app = typer.Typer(help="Auditable mathematical evaluation for QwenInferenceLab.")

def open_repository():
    client = create_client()
    database, validation = validate_read_only(client)
    repository = EvaluatorRepository(database)
    repository.ensure_indexes()
    return client, repository, validation

@app.command("audit-gold")
def audit_gold() -> None:
    """Build and audit all question gold profiles."""
    client, repository, validation = open_repository()
    try:
        counts: Counter[str] = Counter()
        for question in repository.questions():
            profile = build_gold_profile(question)
            repository.upsert_gold_profile(profile)
            counts[profile.status.value] += 1
        typer.echo(f"Connected database: {validation.database}")
        typer.echo(f"Questions: {validation.questions}")
        for status in ("resolved", "multi_part", "semantic_required", "ambiguous", "parse_failed"):
            typer.echo(f"{status}: {counts[status]}")
    finally:
        client.close()

@app.command("evaluate-run")
def evaluate_one(run_id: str = typer.Option(..., "--run-id"), judge_model_config_id: str | None = typer.Option(None, "--judge-model-config-id"), max_level: int = typer.Option(3, min=1, max=3), force: bool = False, judge_retries: int = typer.Option(2, min=0, max=5), judge_timeout: float = typer.Option(60, min=5, max=600)) -> None:
    """Evaluate one persisted run."""
    client, repository, _ = open_repository()
    try:
        status, document = run_pipeline(repository, run_id, judge_model_config_id=judge_model_config_id, max_level=max_level, force=force, judge_retries=judge_retries, judge_timeout=judge_timeout)
        typer.echo(f"{status}: {run_id}")
        if document:
            typer.echo(f"verdict={document[chr(109)+chr(97)+chr(116)+chr(104)][chr(118)+chr(101)+chr(114)+chr(100)+chr(105)+chr(99)+chr(116)]} level={document[chr(109)+chr(97)+chr(116)+chr(104)][chr(108)+chr(101)+chr(118)+chr(101)+chr(108)]}")
    finally:
        client.close()

@app.command("evaluate")
def evaluate_batches(batch_state: list[Path] = typer.Option(..., "--batch-state", exists=True, dir_okay=False), judge_model_config_id: str | None = typer.Option(None, "--judge-model-config-id"), max_level: int = typer.Option(3, min=1, max=3), force: bool = False, concurrency: int = typer.Option(4, min=1, max=16), local_only: bool = typer.Option(False, "--local-only"), judge_retries: int = typer.Option(2, min=0, max=5), judge_timeout: float = typer.Option(60, min=5, max=600)) -> None:
    """Evaluate runIds; Judge calls are concurrent."""
    batches = [load_batch_state(path) for path in batch_state]
    run_ids = list(dict.fromkeys(run_id for batch in batches for run_id in batch["runIds"]))
    client, repository, _ = open_repository()
    counts: Counter[str] = Counter()
    local_ids: list[str] = []
    judge_ids: list[str] = []
    for candidate_id in run_ids:
        stored_run = repository.get_run(candidate_id) or {}
        if stored_run.get("status") in {"failed", "cancelled", "interrupted"}:
            local_ids.append(candidate_id)
        else:
            judge_ids.append(candidate_id)
    if local_only:
        judge_ids = []
    total_ids = len(local_ids) + len(judge_ids)
    def work(run_id: str) -> str:
        try:
            status, _ = run_pipeline(repository, run_id, judge_model_config_id=judge_model_config_id, max_level=max_level, force=force, judge_retries=judge_retries, judge_timeout=judge_timeout)
            return status
        except Exception as error:
            return f"error:{type(error).__name__}"
    try:
        processed = 0
        for run_id in local_ids:
            counts[work(run_id)] += 1
            processed += 1
            if processed % 25 == 0:
                typer.echo(f"processed {processed}/{total_ids}")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for status in executor.map(work, judge_ids):
                counts[status] += 1
                processed += 1
                if processed % 25 == 0 or processed == len(run_ids):
                    typer.echo(f"processed {processed}/{total_ids}")
        typer.echo(" ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    finally:
        client.close()

@app.command("report")
def report_batches(batch_state: list[Path] = typer.Option(..., "--batch-state", exists=True, dir_okay=False), audit_manifest: Path | None = typer.Option(None, "--audit-manifest"), audit_size: int = typer.Option(50, min=1)) -> None:
    """Print per-batch and aggregate statistics; optionally build a human-audit manifest."""
    batches = [load_batch_state(path) for path in batch_state]
    client, repository, _ = open_repository()
    try:
        reports = [(batch, summarize(repository.evaluations_for_runs(batch["runIds"]))) for batch in batches]
        all_ids = list(dict.fromkeys(run_id for batch in batches for run_id in batch["runIds"]))
        all_evaluations = repository.evaluations_for_runs(all_ids)
        reports.append(({"name": "All Batches", "path": "-", "startedAt": "-", "status": "aggregate", "runIds": all_ids}, summarize(all_evaluations)))
        typer.echo(render_report(reports))
        if audit_manifest:
            count = build_audit_manifest(repository, all_evaluations, audit_manifest, audit_size)
            typer.echo(f"Audit manifest: {audit_manifest} ({count} samples)")
    finally:
        client.close()

if __name__ == "__main__":
    app()
