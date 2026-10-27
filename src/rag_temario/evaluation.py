"""Retrieval metrics.

What is measured, and why each one is here:

* **recall@k** — did the top k contain anything from the right syllabus area?
  This is the number that decides whether generation can possibly be right.
* **MRR** — how far down the list the first correct chunk sits. recall@20 can
  look healthy while every correct chunk is at rank 18, which re-ranking then
  has to rescue.
* **precision@k** — how much of the context window is wasted on the wrong area.
* **baselines** — recall@k against a five-class label is meaningless without
  them: the corpus is 46% 'estadistica', so a retriever that always returned
  random chunks would already score well. Both the analytic random baseline and
  the majority-area baseline are reported next to every result.
* **out-of-corpus AUC** — 12 questions have no supporting document at all. If
  the top-1 similarity score separates them from the rest, the system can
  abstain instead of confabulating. This is measured on retrieval alone, with no
  LLM call, so it is free and not confounded by the generator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from .chunking import Chunk
from .evalset import EvalQuestion
from .stores.base import Hit

DEFAULT_KS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines one experiment, and nothing that does not."""

    chunk_size: int
    overlap: int
    strategy: str
    mode: str
    embedder: str
    store: str
    rerank: bool = False

    @property
    def index_name(self) -> str:
        """Namespace of the stored vectors.

        Deliberately excludes ``mode``: dense and hybrid retrieval read the same
        vectors, and putting the mode in the namespace would index the corpus
        twice and bill the embedding API twice for identical text.
        """
        chunking = "page" if self.strategy == "page" else f"{self.chunk_size}-{self.overlap}"
        return f"{self.embedder}_{chunking}"

    @property
    def name(self) -> str:
        suffix = "-rr" if self.rerank else ""
        return f"{self.index_name}_{self.mode}{suffix}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RunResult:
    config: RunConfig
    n_questions: int
    n_chunks: int
    recall_at: dict[int, float] = field(default_factory=dict)
    precision_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    recall_by_area: dict[str, float] = field(default_factory=dict)
    baseline_random_at: dict[int, float] = field(default_factory=dict)
    baseline_majority: float = 0.0
    out_of_corpus_auc: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["config"] = self.config.to_dict()
        data["config_name"] = self.config.name
        for key in ("recall_at", "precision_at", "baseline_random_at"):
            data[key] = {str(k): v for k, v in getattr(self, key).items()}
        return data


def roc_auc(scores: Sequence[float], positives: Sequence[bool]) -> float:
    """Rank-based AUC (Mann-Whitney U), tie-aware. 0.5 means no separation."""
    pairs = sorted(zip(scores, positives, strict=True), key=lambda p: p[0])
    ranks: list[float] = [0.0] * len(pairs)
    index = 0
    while index < len(pairs):
        stop = index
        while stop + 1 < len(pairs) and pairs[stop + 1][0] == pairs[index][0]:
            stop += 1
        average = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[position] = average
        index = stop + 1
    n_pos = sum(1 for _, p in pairs if p)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum = sum(rank for rank, (_, p) in zip(ranks, pairs, strict=True) if p)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def analytic_random_recall(chunks: Sequence[Chunk], questions: Sequence[EvalQuestion], k: int
                           ) -> float:
    """Expected recall@k of drawing k chunks uniformly at random, without replacement.

    P(hit) = 1 - P(all k drawn chunks are outside the target area), averaged over
    the questions so the area mix of the eval set is respected.
    """
    total = len(chunks)
    per_area: dict[str, int] = {}
    for chunk in chunks:
        per_area[chunk.area] = per_area.get(chunk.area, 0) + 1
    scored = [q for q in questions if q.in_corpus]
    if not scored or not total:
        return 0.0
    accumulated = 0.0
    for question in scored:
        in_area = per_area.get(question.area or "", 0)
        miss = 1.0
        for i in range(min(k, total)):
            numerator = total - in_area - i
            if numerator <= 0:
                miss = 0.0
                break
            miss *= numerator / (total - i)
        accumulated += 1 - miss
    return accumulated / len(scored)


def majority_baseline(chunks: Sequence[Chunk], questions: Sequence[EvalQuestion]) -> float:
    """Accuracy of always answering with the area that owns the most chunks."""
    per_area: dict[str, int] = {}
    for chunk in chunks:
        per_area[chunk.area] = per_area.get(chunk.area, 0) + 1
    if not per_area:
        return 0.0
    majority = max(per_area, key=lambda area: per_area[area])
    scored = [q for q in questions if q.in_corpus]
    if not scored:
        return 0.0
    return sum(q.area == majority for q in scored) / len(scored)


def score_run(
    results: Sequence[tuple[EvalQuestion, list[Hit]]],
    chunks: Sequence[Chunk],
    config: RunConfig,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    seconds: float = 0.0,
) -> RunResult:
    in_corpus = [(q, hits) for q, hits in results if q.in_corpus]
    run = RunResult(
        config=config,
        n_questions=len(in_corpus),
        n_chunks=len(chunks),
        seconds=seconds,
    )

    for k in ks:
        hits_at_k = [
            any(hit.chunk.area == question.area for hit in hits[:k])
            for question, hits in in_corpus
        ]
        run.recall_at[k] = sum(hits_at_k) / len(in_corpus) if in_corpus else 0.0
        precisions = [
            sum(hit.chunk.area == question.area for hit in hits[:k]) / max(len(hits[:k]), 1)
            for question, hits in in_corpus
        ]
        run.precision_at[k] = sum(precisions) / len(precisions) if precisions else 0.0
        run.baseline_random_at[k] = analytic_random_recall(chunks, [q for q, _ in in_corpus], k)

    reciprocal = []
    for question, hits in in_corpus:
        rank = next((h.rank for h in hits if h.chunk.area == question.area), None)
        reciprocal.append(1.0 / rank if rank else 0.0)
    run.mrr = sum(reciprocal) / len(reciprocal) if reciprocal else 0.0

    by_area: dict[str, list[bool]] = {}
    for question, hits in in_corpus:
        area = question.area or "?"
        by_area.setdefault(area, []).append(
            any(hit.chunk.area == area for hit in hits[:5])
        )
    run.recall_by_area = {
        area: sum(values) / len(values) for area, values in sorted(by_area.items())
    }

    run.baseline_majority = majority_baseline(chunks, [q for q, _ in results])

    top_scores = [hits[0].score if hits else 0.0 for _, hits in results]
    labels = [q.in_corpus for q, _ in results]
    run.out_of_corpus_auc = roc_auc(top_scores, labels)
    return run
