"""Wiring: corpus -> chunks -> vectors -> index -> retrieval -> metrics.

Kept separate from the CLI so experiments can be driven from a notebook, a test
or (next iteration) an MLflow run without going through argument parsing.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from .cache import corpus_fingerprint, save_pages
from .cache import load_pages as load_cached_pages
from .chunking import Chunk, chunk_pages, get_tokenizer
from .config import Settings
from .corpus import Page, load_corpus
from .embeddings import Embedder
from .evalset import EvalQuestion
from .evaluation import RunConfig, RunResult, score_run
from .retrieval import BM25Index, Retriever
from .stores import build_store
from .stores.base import Hit, VectorStore


def get_pages(cfg: Settings, *, refresh: bool = False) -> list[Page]:
    """Parse the PDFs, or reuse the cached parse when the sources are unchanged."""
    cache_path = cfg.data_dir / "cache" / "pages.jsonl"
    fingerprint = corpus_fingerprint(cfg.corpus_dir)
    if not refresh:
        cached = load_cached_pages(cache_path, fingerprint)
        if cached is not None:
            return cached
    pages = load_corpus(cfg.corpus_dir)
    save_pages(cache_path, fingerprint, pages)
    return pages


def embed_queries(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
    """Batch query embedding when the provider supports it."""
    batch = getattr(embedder, "embed_queries", None)
    if callable(batch):
        return batch(texts)
    return np.stack([embedder.embed_query(text) for text in texts])


def build_index(
    cfg: Settings,
    config: RunConfig,
    embedder: Embedder,
    pages: Sequence[Page],
    *,
    batch_size: int = 256,
) -> tuple[VectorStore | None, list[Chunk]]:
    tokenizer = get_tokenizer()
    chunks = chunk_pages(
        pages,
        tokenizer,
        size=config.chunk_size,
        overlap=config.overlap,
        strategy=config.strategy,
    )
    if config.mode == "bm25":
        # BM25 indexes text, not vectors. Touching the vector store here would
        # not just waste work: with pgvector, reset() re-creates the table for
        # the declared dimension and would drop the real 768-dim index.
        return None, chunks

    store = build_store(cfg, config.index_name)
    store.reset(embedder.dim)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_documents([chunk.text for chunk in batch])
        store.add(batch, vectors)
    return store, chunks


def run_experiment(
    cfg: Settings,
    config: RunConfig,
    embedder: Embedder,
    questions: Sequence[EvalQuestion],
    *,
    pages: Sequence[Page] | None = None,
    k: int = 20,
    reranker=None,
) -> tuple[RunResult, list[tuple[EvalQuestion, list[Hit]]]]:
    pages = pages if pages is not None else get_pages(cfg)
    started = time.perf_counter()
    store, chunks = build_index(cfg, config, embedder, pages)

    bm25 = BM25Index(chunks) if config.mode in ("bm25", "hybrid") else None
    retriever = Retriever(store, embedder, mode=config.mode, bm25=bm25)

    results: list[tuple[EvalQuestion, list[Hit]]] = []
    if config.mode == "bm25":
        for question in questions:
            results.append((question, retriever.retrieve(question.question, k)))
    else:
        vectors = embed_queries(embedder, [q.question for q in questions])
        for question, vector in zip(questions, vectors, strict=True):
            hits = retriever.retrieve_with_vector(question.question, vector, k)
            results.append((question, hits))

    if reranker is not None:
        results = [(q, reranker.rerank(q.question, hits)) for q, hits in results]

    elapsed = time.perf_counter() - started
    return score_run(results, chunks, config, seconds=elapsed), results


def sweep(
    cfg: Settings,
    configs: Sequence[RunConfig],
    embedder: Embedder,
    questions: Sequence[EvalQuestion],
    *,
    on_result=None,
) -> list[RunResult]:
    pages = get_pages(cfg)
    runs = []
    for config in configs:
        result, _ = run_experiment(cfg, config, embedder, questions, pages=pages)
        runs.append(result)
        if on_result:
            on_result(result)
    return runs


def default_sweep(embedder_name: str, store_name: str) -> list[RunConfig]:
    """The grid from the project brief, plus the baselines it did not ask for."""
    grid: list[RunConfig] = []
    for size, overlap in ((400, 50), (800, 100), (1200, 150)):
        grid.append(
            RunConfig(
                chunk_size=size,
                overlap=overlap,
                strategy="window",
                mode="dense",
                embedder=embedder_name,
                store=store_name,
            )
        )
    grid.append(
        RunConfig(
            chunk_size=0, overlap=0, strategy="page", mode="dense",
            embedder=embedder_name, store=store_name,
        )
    )
    best_guess = 800
    for mode in ("bm25", "hybrid"):
        grid.append(
            RunConfig(
                chunk_size=best_guess, overlap=100, strategy="window", mode=mode,
                embedder=embedder_name, store=store_name,
            )
        )
    return grid
