"""On-disk caches.

Two things in this pipeline are slow or billed and both are pure functions of
their input, so both are cached: parsing 1,280 PDF pages (~70 s) and embedding
chunks (an API call). Without these, sweeping four chunk configurations means
re-parsing and re-embedding the same text four times.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .corpus import PARSER_VERSION, Page


def corpus_fingerprint(corpus_dir: Path) -> str:
    """Identity of the parse: the source PDFs *and* the parser that read them."""
    parts = [f"parser:{PARSER_VERSION}"]
    for pdf in sorted(corpus_dir.rglob("*.pdf")):
        stat = pdf.stat()
        parts.append(f"{pdf.relative_to(corpus_dir)}:{stat.st_size}:{int(stat.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def load_pages(path: Path, fingerprint: str) -> list[Page] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        if header.get("fingerprint") != fingerprint:
            return None
        return [Page(**json.loads(line)) for line in handle]


def save_pages(path: Path, fingerprint: str, pages: Sequence[Page]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"fingerprint": fingerprint, "pages": len(pages)}) + "\n")
        for page in pages:
            handle.write(json.dumps(page.__dict__, ensure_ascii=False) + "\n")


class EmbeddingCache:
    """Content-addressed store of embedding vectors, keyed by model+dim+task+text."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            " key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        self._conn.commit()

    @staticmethod
    def key(model: str, dim: int, task: str, text: str) -> str:
        """Identity of one vector: everything that changes the number the API returns.

        The model and the output dimension are part of the key, not just the text:
        a 768-dim vector from one model and a 768-dim vector from another are
        both perfectly valid arrays, and nothing downstream would notice if the
        cache handed one back for the other. ``text`` must be the exact string
        that was sent to the provider, instruction prefix included.
        """
        digest = hashlib.sha256(f"{model}\x00{dim}\x00{task}\x00{text}".encode()).hexdigest()
        return digest

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT key, dim, vector FROM vectors WHERE key IN ({placeholders})", batch
            )
            for key, dim, blob in rows:
                found[key] = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        return found

    def put_many(self, items: Sequence[tuple[str, np.ndarray]]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO vectors (key, dim, vector) VALUES (?, ?, ?)",
            [(key, len(vec), np.asarray(vec, dtype=np.float32).tobytes()) for key, vec in items],
        )
        self._conn.commit()

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
