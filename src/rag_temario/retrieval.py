"""Retrieval strategies: dense, lexical, and their fusion.

A dense retriever is not automatically better than a twenty-year-old lexical
one. On this corpus — dense Spanish technical prose full of exact terminology
('tasa bruta de mortalidad', 'estimador insesgado') — BM25 is a serious
competitor, and refusing to measure it is how teams end up paying for an
embedding API that loses to grep. All three modes share one interface so the
evaluation can run them head to head.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from .chunking import Chunk
from .corpus import fold
from .embeddings import Embedder
from .stores.base import Hit, VectorStore

_TOKEN = re.compile(r"\w+")

# Spanish function words carry no retrieval signal and inflate BM25 length
# normalisation. Small, explicit list — no external dependency.
STOPWORDS = frozenset({
    "a", "al", "algo", "algunos", "ante", "antes", "como", "con", "contra", "cual", "cuando",
    "de", "del", "desde", "donde", "dos", "el", "ella", "ellas", "ellos", "en", "entre", "era",
    "es", "esa", "ese", "eso", "esta", "este", "esto", "ha", "hasta", "hay", "la", "las", "le",
    "les", "lo", "los", "mas", "me", "mi", "mucho", "muy", "no", "nos", "o", "os", "otra",
    "otro", "para", "pero", "por", "porque", "que", "quien", "se", "sea", "segun", "ser", "si",
    "sin", "sobre", "son", "su", "sus", "tambien", "tanto", "te", "tiene", "todo", "todos",
    "tras", "un", "una", "uno", "unos", "y", "ya",
})


def tokenize(text: str) -> list[str]:
    """Accent-folded word tokens with stopwords removed."""
    return [t for t in _TOKEN.findall(fold(text)) if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    """Okapi BM25. ~50 lines, no dependency, and a genuinely hard baseline."""

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._lengths = np.zeros(len(self.chunks), dtype=np.float32)
        for index, chunk in enumerate(self.chunks):
            terms = Counter(tokenize(chunk.text))
            self._lengths[index] = sum(terms.values())
            for term, count in terms.items():
                self._postings[term][index] = count
        self._avg_length = float(self._lengths.mean()) if len(self.chunks) else 0.0
        total = len(self.chunks)
        self._idf = {
            term: math.log(1 + (total - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in self._postings.items()
        }

    def search(self, query: str, k: int) -> list[Hit]:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for index, count in postings.items():
                norm = 1 - self.b + self.b * self._lengths[index] / max(self._avg_length, 1e-9)
                scores[index] += idf * count * (self.k1 + 1) / (count + self.k1 * norm)
        if not len(scores):
            return []
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Hit(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(top, start=1)
            if scores[i] > 0
        ]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], *, k: int = 60) -> list[Hit]:
    """Combine rankings by rank, not by score.

    Cosine similarity and BM25 scores are not on a comparable scale, and
    normalising them is arbitrary. RRF only uses positions, which is why it is
    the standard way to fuse heterogeneous retrievers.
    """
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for hit in ranking:
            scores[hit.chunk.chunk_id] += 1.0 / (k + hit.rank)
            chunks[hit.chunk.chunk_id] = hit.chunk
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    return [
        Hit(chunk=chunks[chunk_id], score=score, rank=rank)
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    ]


class Retriever:
    """Facade over the three modes, so callers pick a strategy by name."""

    MODES = ("dense", "bm25", "hybrid")

    def __init__(
        self,
        store: VectorStore | None,
        embedder: Embedder,
        *,
        mode: str = "dense",
        bm25: BM25Index | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.store = store
        self.embedder = embedder
        self.mode = mode
        self._bm25 = bm25
        if mode != "bm25" and store is None:
            raise ValueError(f"mode {mode!r} needs a vector store")
        if mode in ("bm25", "hybrid") and self._bm25 is None:
            self._bm25 = BM25Index(store.all_chunks())

    def retrieve(self, query: str, k: int = 20) -> list[Hit]:
        if self.mode == "bm25":
            return self._bm25.search(query, k)
        dense = self.store.search(self.embedder.embed_query(query), k)
        if self.mode == "dense":
            return dense
        return reciprocal_rank_fusion([dense, self._bm25.search(query, k)])[:k]

    def retrieve_with_vector(self, query: str, vector: np.ndarray, k: int = 20) -> list[Hit]:
        """Same as ``retrieve`` but reuses an embedding computed elsewhere."""
        if self.mode == "bm25":
            return self._bm25.search(query, k)
        dense = self.store.search(vector, k)
        if self.mode == "dense":
            return dense
        return reciprocal_rank_fusion([dense, self._bm25.search(query, k)])[:k]
