"""The contract every vector store honours."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


class VectorStore(Protocol):
    namespace: str

    def reset(self, dim: int) -> None:
        """Drop anything previously indexed under this namespace."""

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def search(self, vector: np.ndarray, k: int) -> list[Hit]: ...

    def count(self) -> int: ...

    def all_chunks(self) -> list[Chunk]: ...
