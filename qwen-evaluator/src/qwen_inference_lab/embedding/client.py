"""OpenAI-compatible embedding client with validation and retries."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import httpx


@dataclass
class EmbeddingClientStats:
    requests: int = 0
    retries: int = 0
    failures: int = 0


def parse_embedding_response(payload: dict[str, Any], expected_count: int) -> tuple[list[list[float]], int]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("Embedding response has no data list")
    if len(data) != expected_count:
        raise ValueError(f"Embedding count mismatch: expected {expected_count}, got {len(data)}")
    try:
        ordered = sorted(data, key=lambda item: int(item["index"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Embedding response contains invalid indices") from error
    if [int(item["index"]) for item in ordered] != list(range(expected_count)):
        raise ValueError("Embedding response indices are not contiguous")
    vectors: list[list[float]] = []
    dimension: int | None = None
    for item in ordered:
        raw = item.get("embedding")
        if not isinstance(raw, list) or not raw:
            raise ValueError("Embedding vector is empty")
        vector = [float(value) for value in raw]
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("Embedding vector contains NaN or infinity")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError("Embedding dimensions are inconsistent")
        vectors.append(vector)
    return vectors, int(dimension or 0)


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 60,
        retries: int = 2,
        api_key: str | None = None,
        retry_backoff: float = 0.25,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = http_client or httpx.Client(headers=headers)
        self._owns_client = http_client is None
        self.stats = EmbeddingClientStats()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "EmbeddingClient":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _request_json(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self.stats.requests += 1
            try:
                response = self._client.request(method, url, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("API response is not a JSON object")
                return payload
            except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as error:
                last_error = error
                retryable = not isinstance(error, httpx.HTTPStatusError) or error.response.status_code in {429, 500, 502, 503, 504}
                if not retryable or attempt >= self.retries:
                    self.stats.failures += 1
                    raise
                self.stats.retries += 1
                if self.retry_backoff:
                    time.sleep(self.retry_backoff * (2 ** attempt))
        raise RuntimeError("Embedding request failed") from last_error

    def list_models(self) -> list[str]:
        payload = self._request_json("GET", f"{self.base_url}/models")
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Models response has no data list")
        return [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]

    def embed(self, texts: list[str], *, batch_size: int = 32) -> tuple[list[list[float]], int]:
        if not texts:
            return [], 0
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        vectors: list[list[float]] = []
        dimension: int | None = None
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset:offset + batch_size]
            payload = self._request_json(
                "POST",
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": batch},
            )
            batch_vectors, batch_dimension = parse_embedding_response(payload, len(batch))
            if dimension is None:
                dimension = batch_dimension
            elif batch_dimension != dimension:
                raise ValueError(f"Embedding dimension changed from {dimension} to {batch_dimension}")
            vectors.extend(batch_vectors)
        if len(vectors) != len(texts):
            raise ValueError(f"Embedding count mismatch: expected {len(texts)}, got {len(vectors)}")
        return vectors, int(dimension or 0)
