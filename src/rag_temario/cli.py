"""Command line entry point: ``rag <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from .config import settings
from .corpus import corpus_stats
from .embeddings import EmbeddingUnavailable, HashEmbedder, build_embedder
from .evalset import build_eval_set, eval_set_stats, read_eval_set, write_eval_set
from .evaluation import RunConfig
from .pipeline import build_index, default_sweep, get_pages, run_experiment
from .report import render_by_area, render_console, render_markdown
from .rerank import build_reranker
from .retrieval import BM25Index, Retriever
from .stores import build_store


def _configure_logging() -> None:
    """Show the embedding progress and rate-limit waits on stderr; results stay on stdout."""
    log = logging.getLogger("rag_temario")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def report_file(directory: Path, embedder_name: str, *, partial: bool = False) -> Path:
    """Where a sweep's table is written: one file per embedder, never shared.

    A single ``RESULTS.md`` meant that the last command to run won — an offline
    smoke run silently replaced the table of a real-model sweep. A run that only
    covers some configurations or questions gets its own ``-partial`` file for
    the same reason.
    """
    suffix = "-partial" if partial else ""
    return directory / f"RESULTS-{embedder_name}{suffix}.md"


def _run_config(args, embedder_name: str, store_name: str) -> RunConfig:
    return RunConfig(
        chunk_size=args.size,
        overlap=args.overlap,
        strategy=args.strategy,
        mode=args.mode,
        embedder=embedder_name,
        store=store_name,
    )


def cmd_stats(args) -> int:
    cfg = settings()
    pages = get_pages(cfg, refresh=args.refresh)
    print(json.dumps(corpus_stats(pages), indent=2, ensure_ascii=False))
    if cfg.eval_set.exists():
        print(json.dumps(eval_set_stats(read_eval_set(cfg.eval_set)), indent=2, ensure_ascii=False))
    print(f"\nGEMINI_API_KEY: {'set' if cfg.has_api_key else 'MISSING (offline mode)'}")
    print(f"embeddings: {cfg.embed_model} at {cfg.embed_dim} dimensions")
    print(f"store: {cfg.store}")
    return 0


def cmd_build_eval_set(args) -> int:
    cfg = settings()
    questions = build_eval_set(cfg.exams_dir)
    write_eval_set(cfg.eval_set, questions)
    print(json.dumps(eval_set_stats(questions), indent=2, ensure_ascii=False))
    print(f"\nwritten to {cfg.eval_set}")
    return 0


def cmd_index(args) -> int:
    cfg = settings()
    embedder = build_embedder(cfg, offline=args.offline)
    config = _run_config(args, embedder.name, cfg.store)
    pages = get_pages(cfg)
    store, chunks = build_index(cfg, config, embedder, pages)
    print(f"indexed {len(chunks)} chunks as '{config.index_name}' in store '{cfg.store}'")
    calls = getattr(embedder, "api_calls", 0)
    if calls:
        print(f"embedding API calls: {calls}")
    return 0


def cmd_ask(args) -> int:
    cfg = settings()
    # --retrieval-only needs no API key: an index built offline can still be
    # searched offline, which is how you inspect what the retriever is doing.
    if not cfg.has_api_key and not args.retrieval_only:
        print("GEMINI_API_KEY is required to answer questions.", file=sys.stderr)
        return 2
    embedder = build_embedder(cfg)
    config = _run_config(args, embedder.name, cfg.store)
    store = build_store(cfg, config.index_name)
    if not store.count():
        print(f"index '{config.index_name}' is empty — run `rag index` first.", file=sys.stderr)
        return 2

    bm25 = BM25Index(store.all_chunks()) if config.mode in ("bm25", "hybrid") else None
    retriever = Retriever(store, embedder, mode=config.mode, bm25=bm25)
    hits = retriever.retrieve(args.question, args.k)
    reranker = build_reranker(cfg, enabled=not args.no_rerank)
    hits = reranker.rerank(args.question, hits)[: args.top]

    if args.retrieval_only:
        for hit in hits:
            print(f"  {hit.rank:>2}. [{hit.score:6.3f}] {hit.chunk.citation} ({hit.chunk.area})")
            print(f"      {hit.chunk.text[:180].strip()}...")
        return 0

    from .answer import Generator

    result = Generator(cfg.chat_model, cfg.api_key).answer(args.question, hits)
    print(result.render())
    return 0


def cmd_chain(args) -> int:
    """Answer through the LangChain runnable instead of the direct path."""
    cfg = settings()
    if not cfg.has_api_key:
        print("GEMINI_API_KEY is required.", file=sys.stderr)
        return 2
    from .chain import TemarioRetriever, build_chain

    embedder = build_embedder(cfg)
    config = _run_config(args, embedder.name, cfg.store)
    store = build_store(cfg, config.index_name)
    if not store.count():
        print(f"index '{config.index_name}' is empty — run `rag index` first.", file=sys.stderr)
        return 2
    bm25 = BM25Index(store.all_chunks()) if config.mode in ("bm25", "hybrid") else None
    retriever = TemarioRetriever(
        retriever=Retriever(store, embedder, mode=config.mode, bm25=bm25),
        k=args.k,
        top_n=args.top,
        reranker=build_reranker(cfg, enabled=not args.no_rerank),
    )
    print(build_chain(retriever, cfg.chat_model, cfg.api_key).invoke(args.question))
    return 0


def cmd_eval(args) -> int:
    cfg = settings()
    embedder = HashEmbedder() if args.offline else build_embedder(cfg)
    questions = read_eval_set(cfg.eval_set)
    if not questions:
        print("eval set is empty — run `rag build-eval-set` first.", file=sys.stderr)
        return 2
    if args.limit:
        questions = questions[: args.limit]

    configs = default_sweep(embedder.name, cfg.store)
    if args.config:
        configs = [c for c in configs if c.name in set(args.config)]
    pages = get_pages(cfg)

    runs = []
    for config in configs:
        result, _ = run_experiment(cfg, config, embedder, questions, pages=pages, k=args.k)
        runs.append(result)
        print(render_console(result))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    api_calls = getattr(embedder, "api_calls", 0)
    payload = {
        "timestamp": stamp,
        "embedder": embedder.name,
        "store": cfg.store,
        "questions": len(questions),
        "embedding_api_calls": api_calls,
        "runs": [run.to_dict() for run in runs],
    }
    run_path = cfg.runs_dir / f"{stamp}.json"
    run_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report = (
        f"# Retrieval results\n\n"
        f"Embedder `{embedder.name}`, store `{cfg.store}`, {len(questions)} questions "
        f"({sum(q.in_corpus for q in questions)} in corpus, "
        f"{sum(not q.in_corpus for q in questions)} out of corpus). "
        f"Generated {stamp}.\n\n"
        f"{render_markdown(runs)}\n\n"
        f"## recall@5 by area\n\n{render_by_area(runs)}\n"
    )
    report_path = report_file(
        cfg.eval_set.parent, embedder.name, partial=bool(args.limit or args.config)
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"\nwrote {run_path}\nwrote {report_path}")
    if hasattr(embedder, "api_calls"):  # the offline embedder makes none, so says nothing
        retries = getattr(embedder, "retries", 0)
        print(f"embedding API calls: {api_calls} ({retries} were retries after a transient error)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_index_args(sp, *, default_mode="dense"):
        sp.add_argument("--size", type=int, default=800, help="chunk size in tokens")
        sp.add_argument("--overlap", type=int, default=100, help="chunk overlap in tokens")
        sp.add_argument("--strategy", choices=("window", "page"), default="window")
        sp.add_argument("--mode", choices=Retriever.MODES, default=default_mode)

    p = sub.add_parser("stats", help="corpus and eval-set summary")
    p.add_argument("--refresh", action="store_true", help="re-parse the PDFs")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("build-eval-set", help="parse official exams into a labelled eval set")
    p.set_defaults(func=cmd_build_eval_set)

    p = sub.add_parser("index", help="chunk, embed and index the corpus")
    add_index_args(p)
    p.add_argument("--offline", action="store_true", help="use the hash embedder, no API calls")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="answer a question with citations")
    p.add_argument("question")
    add_index_args(p)
    p.add_argument("-k", type=int, default=20, help="candidates to retrieve")
    p.add_argument("--top", type=int, default=5, help="passages kept after re-ranking")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--retrieval-only", action="store_true", help="show passages, skip generation")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chain", help="same question, answered through LangChain")
    p.add_argument("question")
    add_index_args(p)
    p.add_argument("-k", type=int, default=20)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--no-rerank", action="store_true")
    p.set_defaults(func=cmd_chain)

    p = sub.add_parser("eval", help="sweep configurations and write the results table")
    p.add_argument("-k", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="use only the first N questions")
    p.add_argument("--offline", action="store_true", help="hash embedder, no API calls")
    p.add_argument("--config", action="append", help="run only these configuration names")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    _configure_logging()
    try:
        return args.func(args)
    except EmbeddingUnavailable as error:
        print(f"\nstopped: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
