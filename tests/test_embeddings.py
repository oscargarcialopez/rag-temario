"""GoogleEmbedder against a fake client: no network, no key, no cost.

The fake mimics the two behaviours of the real API that matter here: it returns
one vector per ``Content`` it is given, and (like gemini-embedding-2) it folds a
bare list of strings into a single aggregated vector. The raw texts it receives
are recorded, so the tests can assert exactly what would have gone over the wire.
"""

import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from google.genai import errors

from rag_temario.cache import EmbeddingCache
from rag_temario.embeddings import (
    DOCUMENT_TEMPLATE,
    QUERY_TEMPLATE,
    EmbeddingUnavailable,
    GoogleEmbedder,
    describe,
    is_daily_quota,
    is_transient,
    retry_hint,
    takes_task_type,
)

MODEL_2 = "gemini-embedding-2"
MODEL_1 = "gemini-embedding-001"


@dataclass
class Call:
    model: str
    texts: list[str]
    config: Any


class FakeClient:
    """Stands in for ``genai.Client``. ``script`` holds one outcome per call:
    an exception to raise, or None to answer normally; once spent, calls succeed."""

    def __init__(self, script: list[Exception | None] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[Call] = []
        self.models = self

    @staticmethod
    def _text(content: Any) -> str:
        return content if isinstance(content, str) else "".join(p.text for p in content.parts)

    @staticmethod
    def _vector(text: str, dim: int, scale: float) -> list[float]:
        rng = np.random.default_rng(zlib.crc32(text.encode()))
        vector = rng.normal(size=dim)
        return (scale * vector / np.linalg.norm(vector)).tolist()

    def embed_content(self, *, model: str, contents: list[Any], config: Any) -> Any:
        texts = [self._text(c) for c in contents]
        self.calls.append(Call(model, texts, config))
        if self.script:
            outcome = self.script.pop(0)
            if outcome is not None:
                raise outcome
        if model.startswith(MODEL_2) and all(isinstance(c, str) for c in contents):
            texts = ["\n".join(texts)]  # a bare list of strings is one aggregated input
        dim = config.output_dimensionality or 3072
        scale = 1.0 if model.startswith(MODEL_2) else 3.0  # -001 does not normalise
        embeddings = [
            type("Embedding", (), {"values": self._vector(t, dim, scale)})() for t in texts
        ]
        return type("Response", (), {"embeddings": embeddings})()


def api_error(
    code: int = 429,
    status: str = "RESOURCE_EXHAUSTED",
    *,
    retry: str | None = "7s",
    quota_id: str = "EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier",
) -> errors.ClientError:
    details: list[dict[str, Any]] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id}],
        }
    ]
    if retry:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry})
    body = {"error": {"code": code, "message": "quota", "status": status, "details": details}}
    return errors.ClientError(code, body)


def embedder(
    model: str = MODEL_2,
    *,
    dim: int = 768,
    client: FakeClient | None = None,
    cache: EmbeddingCache | None = None,
    sleeps: list[float] | None = None,
    **kwargs: Any,
) -> GoogleEmbedder:
    return GoogleEmbedder(
        model,
        "not-a-real-key",
        cache,
        dim=dim,
        client=client or FakeClient(),
        sleep=(sleeps if sleeps is not None else []).append,
        **kwargs,
    )


# --- dimension ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [MODEL_2, MODEL_1])
@pytest.mark.parametrize("dim", [768, 1536])
def test_output_dimensionality_is_sent_and_honoured(model, dim):
    client = FakeClient()
    vectors = embedder(model, dim=dim, client=client).embed_documents(["uno", "dos"])

    assert vectors.shape == (2, dim)
    assert client.calls and all(call.config.output_dimensionality == dim for call in client.calls)


def test_the_real_client_is_built_with_an_explicit_request_timeout(monkeypatch):
    import google.genai

    seen: dict[str, Any] = {}

    def fake_client(**kwargs: Any) -> FakeClient:
        seen.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(google.genai, "Client", fake_client)
    GoogleEmbedder(MODEL_2, "not-a-real-key", dim=768, request_timeout=30)

    assert seen["api_key"] == "not-a-real-key"
    assert seen["http_options"].timeout == 30_000  # milliseconds, and never "no timeout"


def test_the_embedder_name_carries_model_and_dimension():
    assert embedder(MODEL_2, dim=768).name == "gemini-embedding-2-768"
    assert embedder(MODEL_1, dim=1536).name == "gemini-embedding-001-1536"
    assert embedder(MODEL_2, dim=768).dim == 768


@pytest.mark.parametrize("dim", [64, 4096])
def test_dimensions_the_model_cannot_produce_are_rejected_up_front(dim):
    with pytest.raises(ValueError, match="output dimension"):
        embedder(MODEL_2, dim=dim)


# --- task type vs instruction -------------------------------------------------------------


def test_task_type_is_used_only_by_gemini_embedding_001():
    old, new = FakeClient(), FakeClient()
    embedder(MODEL_1, client=old).embed_documents(["texto"])
    embedder(MODEL_1, client=old).embed_query("texto")
    embedder(MODEL_2, client=new).embed_documents(["texto"])
    embedder(MODEL_2, client=new).embed_query("texto")

    assert [c.config.task_type for c in old.calls] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    assert [c.config.task_type for c in new.calls] == [None, None]
    assert takes_task_type(MODEL_1) and not takes_task_type(MODEL_2)
    assert not takes_task_type("models/gemini-embedding-2")


def test_gemini_embedding_001_receives_the_text_untouched():
    client = FakeClient()
    embedder(MODEL_1, client=client).embed_documents(["texto del documento"])
    embedder(MODEL_1, client=client).embed_query("¿una pregunta?")

    assert [c.texts for c in client.calls] == [["texto del documento"], ["¿una pregunta?"]]


def test_gemini_embedding_2_receives_the_official_instruction_format():
    client = FakeClient()
    embedder(MODEL_2, client=client).embed_documents(["texto del documento"])
    embedder(MODEL_2, client=client).embed_query("¿una pregunta?")

    # exact strings from https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2
    assert client.calls[0].texts == ["title: none | text: texto del documento"]
    assert client.calls[1].texts == ["task: search result | query: ¿una pregunta?"]
    assert DOCUMENT_TEMPLATE.startswith("title: none | text: ")
    assert QUERY_TEMPLATE.startswith("task: search result | query: ")


def test_braces_in_the_text_survive_the_instruction_template():
    client = FakeClient()
    embedder(MODEL_2, client=client).embed_documents(["P(X) = {x : x > 0} y {content}"])

    assert client.calls[0].texts == ["title: none | text: P(X) = {x : x > 0} y {content}"]


def test_query_and_document_vectors_of_the_same_text_differ_and_are_cached_apart(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    model = embedder(MODEL_2, cache=cache)
    as_document = model.embed_documents(["mismo texto"])[0]
    as_query = model.embed_query("mismo texto")

    assert not np.allclose(as_document, as_query)
    assert len(cache) == 2


# --- batching ------------------------------------------------------------------------------


def test_every_text_is_sent_as_its_own_content_so_nothing_is_aggregated():
    client = FakeClient()
    vectors = embedder(MODEL_2, client=client).embed_documents(["a", "b", "c", "d", "e"])

    assert vectors.shape == (5, 768)
    assert len({tuple(np.round(v, 5)) for v in vectors}) == 5  # five distinct vectors
    assert len(client.calls) == 1 and len(client.calls[0].texts) == 5


def test_an_aggregating_response_is_a_loud_failure_not_silent_garbage():
    class Aggregating(FakeClient):
        def embed_content(self, *, model, contents, config):
            response = super().embed_content(model=model, contents=contents, config=config)
            response.embeddings = response.embeddings[:1]
            return response

    with pytest.raises(RuntimeError, match="aggregating"):
        embedder(MODEL_2, client=Aggregating()).embed_documents(["a", "b", "c"])


def test_a_vector_of_the_wrong_size_is_rejected():
    class WrongSize(FakeClient):
        def embed_content(self, *, model, contents, config):
            config.output_dimensionality = 512
            return super().embed_content(model=model, contents=contents, config=config)

    with pytest.raises(RuntimeError, match="expected"):
        embedder(MODEL_2, client=WrongSize()).embed_documents(["a"])


def test_texts_are_split_into_batches_of_the_configured_size():
    client = FakeClient()
    embedder(MODEL_2, client=client, batch_size=2).embed_documents(list("abcde"))

    assert [len(call.texts) for call in client.calls] == [2, 2, 1]


def test_a_text_that_appears_twice_is_embedded_once():
    client = FakeClient()
    vectors = embedder(MODEL_2, client=client).embed_documents(["igual", "otro", "igual"])

    assert sum(len(call.texts) for call in client.calls) == 2
    assert np.allclose(vectors[0], vectors[2])


def test_nothing_to_embed_makes_no_call():
    client = FakeClient()
    assert embedder(MODEL_2, client=client).embed_documents([]).shape == (0, 768)
    assert client.calls == []


def test_output_is_l2_normalised_even_when_the_model_does_not_normalise():
    # gemini-embedding-001 returns truncated vectors with norm != 1 (the fake uses 3.0)
    vectors = embedder(MODEL_1).embed_documents(["uno", "dos"])

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


# --- cache ---------------------------------------------------------------------------------


def test_cache_is_written_after_every_batch_not_only_at_the_end(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    boom = RuntimeError("provider exploded on batch 3")
    client = FakeClient(script=[None, None, boom])
    texts = [f"texto {i}" for i in range(5)]

    with pytest.raises(RuntimeError, match="batch 3"):
        embedder(MODEL_2, client=client, cache=cache, batch_size=2).embed_documents(texts)

    assert len(cache) == 4  # batches 1 and 2 survived the failure of batch 3

    resumed = FakeClient()
    vectors = embedder(MODEL_2, client=resumed, cache=cache, batch_size=2).embed_documents(texts)

    assert vectors.shape == (5, 768)
    assert [len(call.texts) for call in resumed.calls] == [1]  # only the missing text
    assert len(cache) == 5


def test_an_interrupt_loses_nothing_that_was_already_embedded(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    client = FakeClient(script=[None, KeyboardInterrupt()])

    with pytest.raises(KeyboardInterrupt):
        embedder(MODEL_2, client=client, cache=cache, batch_size=2).embed_documents(list("abcde"))

    assert len(cache) == 2  # the first batch was already on disk when Ctrl-C hit the second


def test_an_empty_cache_is_still_written_to(tmp_path):
    """A fresh EmbeddingCache is falsy (it defines __len__): `if cache:` skipped every write."""
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    assert not cache  # the trap

    embedder(MODEL_2, cache=cache).embed_documents(["uno"])

    assert len(cache) == 1


def test_a_second_run_over_the_same_texts_costs_no_api_call(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    first, second = FakeClient(), FakeClient()
    a = embedder(MODEL_2, client=first, cache=cache).embed_documents(["x", "y"])
    b = embedder(MODEL_2, client=second, cache=cache).embed_documents(["x", "y"])

    assert second.calls == []
    assert np.allclose(a, b)


def test_vectors_from_different_models_or_dimensions_never_mix(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    variants = [(MODEL_2, 768), (MODEL_2, 1536), (MODEL_1, 768)]
    clients = [FakeClient() for _ in variants]

    for (model, dim), client in zip(variants, clients, strict=True):
        out = embedder(model, dim=dim, client=client, cache=cache).embed_documents(["mismo"])
        assert out.shape == (1, dim)

    # all three had to go to the API: none was served from another one's entry
    assert [len(c.calls) for c in clients] == [1, 1, 1]
    assert len(cache) == 3


# --- retries -------------------------------------------------------------------------------


def test_a_rate_limit_waits_for_the_delay_the_api_asked_for():
    sleeps: list[float] = []
    client = FakeClient(script=[api_error(retry="7s")])
    model = embedder(MODEL_2, client=client, sleeps=sleeps)

    vectors = model.embed_documents(["uno"])

    assert vectors.shape == (1, 768)
    assert sleeps == [8.0]  # the hint, plus a second to land after the window reopens
    assert (model.api_calls, model.retries) == (2, 1)


def test_without_a_hint_the_wait_doubles_up_to_a_minute():
    sleeps: list[float] = []
    client = FakeClient(script=[api_error(retry=None)] * 8)

    embedder(MODEL_2, client=client, sleeps=sleeps).embed_documents(["uno"])

    assert sleeps == [2, 4, 8, 16, 32, 60, 60, 60]


def test_it_gives_up_with_a_clear_error_once_attempts_run_out():
    sleeps: list[float] = []
    client = FakeClient(script=[api_error(retry=None)] * 10)
    model = embedder(MODEL_2, client=client, sleeps=sleeps, max_retries=4)

    with pytest.raises(EmbeddingUnavailable, match="cached"):
        model.embed_documents(["uno"])

    assert model.api_calls == 4
    assert len(sleeps) == 3


def test_a_daily_quota_stops_at_once_instead_of_sleeping():
    sleeps: list[float] = []
    exhausted = api_error(quota_id="EmbedContentRequestsPerDayPerProjectPerModel-FreeTier")
    model = embedder(MODEL_2, client=FakeClient(script=[exhausted]), sleeps=sleeps)

    with pytest.raises(EmbeddingUnavailable, match="daily"):
        model.embed_documents(["uno"])

    assert sleeps == [] and model.api_calls == 1


def test_a_hint_that_is_not_a_per_minute_wait_is_not_slept_through():
    sleeps: list[float] = []
    model = embedder(MODEL_2, client=FakeClient(script=[api_error(retry="3600s")]), sleeps=sleeps)

    with pytest.raises(EmbeddingUnavailable):
        model.embed_documents(["uno"])

    assert sleeps == []


def test_errors_that_waiting_cannot_fix_are_not_retried():
    sleeps: list[float] = []
    bad_request = api_error(400, "INVALID_ARGUMENT", retry=None)
    model = embedder(MODEL_2, client=FakeClient(script=[bad_request]), sleeps=sleeps)

    with pytest.raises(errors.ClientError):
        model.embed_documents(["uno"])

    assert sleeps == [] and model.api_calls == 1


def test_error_classification_helpers():
    assert is_transient(api_error(429)) and is_transient(api_error(503, "UNAVAILABLE"))
    assert is_transient(TimeoutError()) and is_transient(ConnectionResetError())
    assert not is_transient(api_error(400, "INVALID_ARGUMENT"))
    assert not is_transient(api_error(403, "PERMISSION_DENIED"))
    assert not is_transient(ValueError("boom"))

    assert is_daily_quota(api_error(quota_id="EmbedContentRequestsPerDay-FreeTier"))
    assert not is_daily_quota(api_error())

    assert retry_hint(api_error(retry="34.5s")) == 34.5
    assert retry_hint(api_error(retry=None)) is None
    assert retry_hint(Exception("429. Please retry in 12.5s.")) == 12.5
    assert retry_hint(Exception('{"retryDelay": "9s"}')) == 9.0
    assert retry_hint(Exception("nothing to see")) is None


def test_describe_gives_one_short_line_and_never_the_whole_body():
    assert describe(api_error()).startswith("429 RESOURCE_EXHAUSTED: quota")
    assert describe(ValueError("boom")) == "ValueError: boom"
    assert len(describe(Exception("word " * 500))) < 200
