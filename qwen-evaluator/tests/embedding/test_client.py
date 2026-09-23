import httpx
import pytest

from qwen_inference_lab.embedding.client import EmbeddingClient, parse_embedding_response


class Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.request = httpx.Request("POST", "http://embedding.test/v1/embeddings")
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("status", request=self.request, response=httpx.Response(self.status_code, request=self.request))
    def json(self):
        return self.payload


class Transport:
    def __init__(self, effects):
        self.effects = iter(effects)
    def request(self, *_args, **_kwargs):
        effect = next(self.effects)
        if isinstance(effect, Exception):
            raise effect
        return effect


def payload(vectors):
    return {"object": "list", "data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]}


def test_embedding_api_response_parsing_and_dimension_detection():
    vectors, dimension = parse_embedding_response(payload([[1, 2, 3], [4, 5, 6]]), 2)
    assert vectors == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert dimension == 3


def test_batch_input_output_count_must_match():
    with pytest.raises(ValueError, match="count mismatch"):
        parse_embedding_response(payload([[1, 2]]), 2)


def test_client_batches_and_keeps_dimension_consistent():
    transport = Transport([Response(payload([[1, 0], [0, 1]])), Response(payload([[1, 1]]))])
    client = EmbeddingClient("http://embedding.test/v1", "model", retries=0, http_client=transport)
    vectors, dimension = client.embed(["a", "b", "c"], batch_size=2)
    assert vectors == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert dimension == 2


def test_api_timeout_is_not_swallowed():
    request = httpx.Request("POST", "http://embedding.test/v1/embeddings")
    client = EmbeddingClient(
        "http://embedding.test/v1", "model", retries=0,
        http_client=Transport([httpx.ReadTimeout("timeout", request=request)]),
    )
    with pytest.raises(httpx.ReadTimeout):
        client.embed(["a"])
    assert client.stats.failures == 1


def test_api_timeout_retries_then_succeeds():
    request = httpx.Request("POST", "http://embedding.test/v1/embeddings")
    client = EmbeddingClient(
        "http://embedding.test/v1", "model", retries=1, retry_backoff=0,
        http_client=Transport([httpx.ReadTimeout("timeout", request=request), Response(payload([[1, 0]]))]),
    )
    vectors, dimension = client.embed(["a"])
    assert vectors == [[1.0, 0.0]]
    assert dimension == 2
    assert client.stats.requests == 2
    assert client.stats.retries == 1
    assert client.stats.failures == 0
