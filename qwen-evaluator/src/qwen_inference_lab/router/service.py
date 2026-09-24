"""Internal FastAPI service for online shadow Router previews."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from qwen_inference_lab.common.mongodb import create_client, validate_read_only

from .runtime import RouterRuntime, build_runtime_from_database


class PreviewRequest(BaseModel):
    question: str = Field(min_length=1, max_length=50_000)
    profile: Literal["development", "conservative"] = "development"
    question_id: str | None = Field(alias="questionId", default=None)

    model_config = {"populate_by_name": True}


RuntimeFactory = Callable[[], tuple[RouterRuntime, Callable[[], None]]]


def default_runtime_factory() -> tuple[RouterRuntime, Callable[[], None]]:
    mongo_client = create_client()
    try:
        database, _ = validate_read_only(mongo_client)
        runtime, embedding_client = build_runtime_from_database(database)
    except Exception:
        mongo_client.close()
        raise

    def close() -> None:
        embedding_client.close()
        mongo_client.close()

    return runtime, close


def create_app(runtime_factory: RuntimeFactory = default_runtime_factory) -> FastAPI:
    state: dict[str, RouterRuntime] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime, close = runtime_factory()
        state["runtime"] = runtime
        try:
            yield
        finally:
            close()
            state.clear()

    application = FastAPI(
        title="QwenInferenceLab Router Runtime",
        version="1.0.0",
        description="Internal-only shadow Router service.",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health() -> dict:
        runtime = state["runtime"]
        return {
            "ok": True,
            "corpusSize": runtime.metadata.corpus_size,
            "dimension": runtime.metadata.dimension,
            "embeddingVersion": runtime.metadata.embedding_version,
            "routerVersion": runtime.metadata.router_version,
        }

    @application.post("/preview")
    def preview(payload: PreviewRequest) -> dict:
        try:
            return state["runtime"].preview(
                payload.question,
                profile=payload.profile,
                exclude_question_id=payload.question_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.exception_handler(Exception)
    async def internal_error(_request: Request, _error: Exception):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"detail": "Router Runtime unavailable"},
        )

    return application


app = create_app()
