"""Exact cosine search over a dense matrix.

This is the reference implementation: no index, no approximation, no service.
Its results are the ground truth that the pgvector HNSW index is measured
against — an approximate index is only worth its speed if you know what recall
it costs you, and you cannot know that without an exact baseline.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..chunking import Chunk
from .base import Hit


class LocalStore:
    def __init__(self, index_dir: Path, namespace: str) -> None:
        self.namespace = namespace
        self._dir = index_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self._load()

    @property
    def _vectors_path(self) -> Path:
        return self._dir / f"{self.namespace}.npy"

    @property
    def _chunks_path(self) -> Path:
        return self._dir / f"{self.namespace}.jsonl"

    def _load(self) -> None:
        if self._vectors_path.exists() and self._chunks_path.exists():
            self._vectors = np.load(self._vectors_path)
            with self._chunks_path.open(encoding="utf-8") as handle:
                self._chunks = [Chunk(**json.loads(line)) for line in handle]

    def reset(self, dim: int) -> None:
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._chunks = []
        self._vectors_path.unlink(missing_ok=True)
        self._chunks_path.unlink(missing_ok=True)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self._vectors = (
            vectors if self._vectors is None or not len(self._vectors)
            else np.vstack([self._vectors, vectors])
        )
        self._chunks.extend(chunks)
        np.save(self._vectors_path, self._vectors)
        with self._chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        if self._vectors is None or not len(self._vectors):
            return []
        # Vectors are L2-normalised on ingest, so the dot product is the cosine.
        scores = self._vectors @ np.asarray(vector, dtype=np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Hit(chunk=self._chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(top, start=1)
        ]

    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)
