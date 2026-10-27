"""Embedding providers.

Two implementations share one interface:

* ``GoogleEmbedder`` calls the Gemini embedding API, batches, retries and caches.
* ``HashEmbedder`` is deterministic, offline and free. It exists so the whole
  pipeline — chunking, indexing, retrieval, evaluation — can be exercised and
  unit-tested without a network call or an API key. Its retrieval quality is
  poor by construction, which makes it a useful floor in the results table.

Vectors are L2-normalised on the way out, so cosine similarity is a dot product
everywhere downstream and the local store and pgvector agree on what "score"
means.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from .cache import EmbeddingCache

logger = logging.getLogger(__name__)

# Query and document are embedded differently; the provider maps them into
# different regions of the space and it measurably helps asymmetric retrieval,
# where a one-line question must match an 800-token passage. The two tasks are
# named here once, but *how* a task reaches the model depends on the model:
#
# * gemini-embedding-001 takes it as the ``task_type`` parameter.
# * gemini-embedding-2 has no such parameter ("You cannot use the task_type
#   field for the gemini-embedding-2 model") and wants it written into the text.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

# Instruction formats for gemini-embedding-2, copied from the official docs,
# "Task types with Embeddings 2" > "Retrieval use cases (Asymmetric format)":
#   https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2
#   query    ("Search query")  task: search result | query: {content}
#   document                   title: {title} | text: {content}
#                              ("If there is no title, use title: none.")
# "search result" is the general retrieval query and the closest counterpart of
# the RETRIEVAL_QUERY used for gemini-embedding-001. The docs also list
# "task: question answering | query: {content}"; it was not tried here.
# Documents go in with ``title: none`` on purpose: every retriever in the
# evaluation sees the same chunk text, and a document title ("Tema 2. Mortalidad")
# names the topic, which is exactly what the evaluation scores.
QUERY_TEMPLATE = "task: search result | query: {content}"
DOCUMENT_TEMPLATE = "title: none | text: {content}"

# Native output size. ``output_dimensionality`` can only truncate from here
# (Matryoshka training), and the docs recommend 768, 1536 or 3072.
KNOWN_DIMENSIONS = {"gemini-embedding-2": 3072, "gemini-embedding-001": 3072}
MIN_DIMENSION = 128

# Backoff for transient errors. A provider hint (RetryInfo) wins; without one the
# wait doubles up to a minute, long enough for a per-minute quota to refill.
BACKOFF_START = 2.0
BACKOFF_CAP = 60.0
# A hinted wait longer than this is not a per-minute quota any more: stop
# instead of sleeping for hours, the cache keeps everything embedded so far.
MAX_HINTED_WAIT = 300.0

_TRANSIENT_CODES = frozenset({408, 429, 500, 502, 503, 504})
_TRANSIENT_STATUSES = frozenset(
    {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"}
)
_RETRY_IN_TEXT = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)
_RETRY_DELAY_TEXT = re.compile(r"retryDelay\W+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)


class EmbeddingUnavailable(RuntimeError):
    """The embedding API could not serve the request even after waiting.

    Raised for a quota that will not refill soon and for retries that ran out.
    Everything embedded before the failure is already in the cache, so running
    the same command again resumes where this one stopped.
    """


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def takes_task_type(model: str) -> bool:
    """True for models that read the task from ``task_type`` rather than from the text."""
    return not model.removeprefix("models/").startswith("gemini-embedding-2")


def is_transient(error: BaseException) -> bool:
    """Worth waiting for: rate limits, provider-side 5xx, and network hiccups."""
    code = getattr(error, "code", None)
    status = str(getattr(error, "status", None) or "").upper()
    if code in _TRANSIENT_CODES or status in _TRANSIENT_STATUSES:
        return True
    if isinstance(error, TimeoutError | ConnectionError):
        return True
    import httpx

    return isinstance(error, httpx.TransportError)


def is_daily_quota(error: BaseException) -> bool:
    """A per-day quota names itself in the error ("...PerDay..."); waiting will not help."""
    text = re.sub(r"[\s_-]+", "", str(error).lower())
    return "perday" in text or "daily" in text


def _seconds(duration: object) -> float | None:
    """A protobuf Duration as the API writes it ("34s", "34.5s") -> seconds."""
    if isinstance(duration, str) and duration.endswith("s"):
        try:
            return float(duration[:-1])
        except ValueError:
            return None
    return None


def retry_hint(error: BaseException) -> float | None:
    """Seconds the provider asked us to wait (``RetryInfo.retryDelay``), if it said."""
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        body = details.get("error", details)
        entries = body.get("details") if isinstance(body, dict) else None
        for entry in entries or []:
            seconds = _seconds(entry.get("retryDelay")) if isinstance(entry, dict) else None
            if seconds is not None:
                return seconds
    match = _RETRY_IN_TEXT.search(str(error)) or _RETRY_DELAY_TEXT.search(str(error))
    return float(match.group(1)) if match else None


def describe(error: BaseException) -> str:
    """One log line for an API error: code, status and the provider's own message, trimmed."""
    parts = (getattr(error, "code", None), getattr(error, "status", None))
    head = " ".join(str(part) for part in parts if part)
    message = " ".join(str(getattr(error, "message", None) or error).split())
    return f"{head or type(error).__name__}: {message[:140]}"


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashEmbedder:
    """Hashing trick over word unigrams and bigrams. No API, no key, no cost."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        words = re.findall(r"\w+", text.lower())
        grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        vector = np.zeros(self.dim, dtype=np.float32)
        for gram in grams:
            digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return l2_normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalise(self._vector(text))


class GoogleEmbedder:
    """Gemini embeddings with an on-disk cache, per-batch persistence and bounded retries.

    The model and the output dimension are explicit and travel together: the
    ``name`` (``gemini-embedding-2-768``) namespaces the indexes, and both are
    part of every cache key, so vectors from different models or sizes can
    never be mixed.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        cache: EmbeddingCache | None = None,
        *,
        dim: int = 768,
        batch_size: int = 50,
        max_retries: int = 10,
        request_timeout: float = 120.0,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        native = KNOWN_DIMENSIONS.get(model)
        if dim < MIN_DIMENSION or (native is not None and dim > native):
            ceiling = native or "the model's native size"
            raise ValueError(
                f"{model}: output dimension must be between {MIN_DIMENSION} and {ceiling}, "
                f"got {dim}"
            )
        if client is None:
            from google import genai
            from google.genai import types

            # The SDK sets no timeout by default, so one stalled connection would hang
            # an hour-long sweep forever. A timeout is a transient error: it is retried.
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(request_timeout * 1000)),
            )

        self.model = model
        self.dim = dim
        self.name = f"{model}-{dim}"
        self._client = client
        self._cache = cache
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._sleep = sleep
        self._task_type_supported = takes_task_type(model)
        self.api_calls = 0  # every request sent, retries included
        self.retries = 0

    def _payload(self, text: str, task: str) -> str:
        """The exact string sent to the API: the text itself, or text plus instruction."""
        if self._task_type_supported:
            return text
        template = QUERY_TEMPLATE if task == TASK_QUERY else DOCUMENT_TEMPLATE
        return template.format(content=text)

    def _config(self, task: str):
        from google.genai import types

        if self._task_type_supported:
            return types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim)
        return types.EmbedContentConfig(output_dimensionality=self.dim)

    def _wait_before_retry(self, error: BaseException, attempt: int, hint: float | None) -> float:
        if hint is None:
            return min(BACKOFF_START * 2 ** (attempt - 1), BACKOFF_CAP)
        if hint > MAX_HINTED_WAIT:
            raise EmbeddingUnavailable(
                f"the API asked for a {hint:.0f} s wait, which is not a per-minute quota; "
                "embeddings done so far are cached, run the same command again later"
            ) from error
        return hint + 1.0  # the hint is when the window reopens; land just after it

    def _embed_raw(self, payloads: Sequence[str], task: str) -> list[np.ndarray]:
        from google.genai import types

        # One Content per text. A bare list of strings is read by gemini-embedding-2
        # as several parts of a single input and comes back as ONE aggregated vector:
        #   https://ai.google.dev/gemini-api/docs/embeddings#embedding-aggregation
        contents = [types.Content(parts=[types.Part.from_text(text=p)]) for p in payloads]
        config = self._config(task)

        for attempt in range(1, self._max_retries + 1):
            try:
                self.api_calls += 1
                response = self._client.models.embed_content(
                    model=self.model, contents=contents, config=config
                )
            except Exception as error:  # noqa: BLE001 - provider raises many types
                if not is_transient(error):
                    raise
                if is_daily_quota(error):
                    raise EmbeddingUnavailable(
                        "daily embedding quota exhausted; embeddings done so far are cached, "
                        "run the same command again after the quota resets (or enable billing)"
                    ) from error
                if attempt == self._max_retries:
                    raise EmbeddingUnavailable(
                        f"still failing after {attempt} attempts "
                        f"({type(error).__name__} {getattr(error, 'code', '')}); "
                        "embeddings done so far are cached, run the same command again"
                    ) from error
                hint = retry_hint(error)
                delay = self._wait_before_retry(error, attempt, hint)
                self.retries += 1
                logger.warning(
                    "embedding API %s; waiting %.0f s%s (retry %d/%d)",
                    describe(error),
                    delay,
                    " as the API asked" if hint is not None else "",
                    attempt,
                    self._max_retries - 1,
                )
                self._sleep(delay)
                continue

            vectors = [np.asarray(e.values, dtype=np.float32) for e in response.embeddings or []]
            if len(vectors) != len(payloads):
                raise RuntimeError(
                    f"sent {len(payloads)} texts to {self.model} and got {len(vectors)} "
                    "embeddings back; the model is aggregating inputs instead of "
                    "embedding them one by one"
                )
            wrong = [v.shape for v in vectors if v.shape != (self.dim,)]
            if wrong:
                raise RuntimeError(
                    f"{self.model} returned vectors of shape {wrong[0]}, expected ({self.dim},)"
                )
            return vectors
        raise AssertionError("unreachable")  # pragma: no cover

    def _embed(self, texts: Sequence[str], task: str) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        payloads = [self._payload(text, task) for text in texts]
        keys = [EmbeddingCache.key(self.model, self.dim, task, p) for p in payloads]
        # `is not None`, never plain truthiness: EmbeddingCache defines __len__, so an
        # empty cache is falsy and `if self._cache` would never let it be filled.
        vectors: dict[str, np.ndarray] = {}
        if self._cache is not None:
            vectors = self._cache.get_many(keys)

        # Each distinct text is sent once, however many times it appears.
        pending: dict[str, str] = {}
        for key, payload in zip(keys, payloads, strict=True):
            if key not in vectors and key not in pending:
                pending[key] = payload

        todo = list(pending.items())
        for start in range(0, len(todo), self._batch_size):
            batch = todo[start : start + self._batch_size]
            fresh = self._embed_raw([payload for _, payload in batch], task)
            items = [(key, vector) for (key, _), vector in zip(batch, fresh, strict=True)]
            # Persist after every batch, not at the end of the call: a failure
            # (quota, network, Ctrl-C) 40 batches in must not cost those 40 batches.
            if self._cache is not None:
                self._cache.put_many(items)
            vectors.update(items)
            logger.info(
                "embedded %d/%d new texts (%s, %s)",
                min(start + len(batch), len(todo)), len(todo), task.lower(), self.name,
            )

        return l2_normalise(np.stack([vectors[key] for key in keys]))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, TASK_DOCUMENT)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], TASK_QUERY)[0]

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, TASK_QUERY)


def build_embedder(cfg, *, offline: bool = False) -> Embedder:
    """Pick the provider: the real one when a key is present, the fake otherwise."""
    if offline or not cfg.has_api_key:
        return HashEmbedder()
    return GoogleEmbedder(
        cfg.embed_model, cfg.api_key, EmbeddingCache(cfg.cache_db), dim=cfg.embed_dim
    )
