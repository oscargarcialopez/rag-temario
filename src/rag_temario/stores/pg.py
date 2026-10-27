"""pgvector-backed store.

Uses an HNSW index with cosine distance. HNSW is *approximate*: it trades recall
for speed, and how much recall it costs depends on ``ef_search``. That parameter
is exposed here rather than left at its default precisely so the evaluation can
put a number on the trade-off against the exact local store.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..chunking import Chunk
from .base import Hit

TABLE = "rag_chunks"

COLUMNS = (
    "namespace",
    "chunk_id",
    "doc_id",
    "area",
    "title",
    "page_start",
    "page_end",
    "n_tokens",
    "text",
    "embedding",
)


class PgVectorStore:
    def __init__(self, dsn: str, namespace: str, *, ef_search: int = 40) -> None:
        self.namespace = namespace
        self.ef_search = ef_search
        self._dsn = dsn
        self._conn = None

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            import psycopg
            from pgvector.psycopg import register_vector

            self._conn = psycopg.connect(self._dsn, autocommit=True)
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self._conn)
        return self._conn

    def _current_dim(self) -> int | None:
        row = self.conn.execute(
            "SELECT a.atttypmod FROM pg_attribute a"
            " JOIN pg_class c ON c.oid = a.attrelid"
            " WHERE c.relname = %s AND a.attname = 'embedding'",
            (TABLE,),
        ).fetchone()
        return int(row[0]) if row and row[0] and row[0] > 0 else None

    def ensure_schema(self, dim: int) -> None:
        current = self._current_dim()
        if current is not None and current != dim:
            # Changing embedding model changes the column type; there is nothing
            # to migrate, the vectors are not comparable across models anyway.
            self.conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
            current = None
        if current is None:
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS {TABLE} ("
                " id bigserial PRIMARY KEY,"
                " namespace text NOT NULL,"
                " chunk_id text NOT NULL,"
                " doc_id text NOT NULL,"
                " area text NOT NULL,"
                " title text NOT NULL,"
                " page_start int NOT NULL,"
                " page_end int NOT NULL,"
                " n_tokens int NOT NULL,"
                " text text NOT NULL,"
                f" embedding vector({dim}) NOT NULL,"
                " UNIQUE (namespace, chunk_id))"
            )
            self.conn.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_hnsw"
                f" ON {TABLE} USING hnsw (embedding vector_cosine_ops)"
            )
            self.conn.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_namespace ON {TABLE} (namespace)"
            )

    def reset(self, dim: int) -> None:
        self.ensure_schema(dim)
        self.conn.execute(f"DELETE FROM {TABLE} WHERE namespace = %s", (self.namespace,))

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.ensure_schema(int(np.asarray(vectors).shape[1]))
        placeholders = ", ".join(["%s"] * len(COLUMNS))
        rows = [
            (
                self.namespace,
                chunk.chunk_id,
                chunk.doc_id,
                chunk.area,
                chunk.title,
                chunk.page_start,
                chunk.page_end,
                chunk.n_tokens,
                chunk.text,
                np.asarray(vector, dtype=np.float32),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) VALUES ({placeholders})"
                " ON CONFLICT (namespace, chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                rows,
            )

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        with self.conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = %s", (self.ef_search,))
            cur.execute(
                "SELECT chunk_id, doc_id, area, title, page_start, page_end, n_tokens, text,"
                f" 1 - (embedding <=> %s) AS score FROM {TABLE}"
                " WHERE namespace = %s ORDER BY embedding <=> %s LIMIT %s",
                (np.asarray(vector, dtype=np.float32), self.namespace,
                 np.asarray(vector, dtype=np.float32), k),
            )
            rows = cur.fetchall()
        return [
            Hit(
                chunk=Chunk(
                    chunk_id=row[0],
                    doc_id=row[1],
                    area=row[2],
                    title=row[3],
                    page_start=row[4],
                    page_end=row[5],
                    n_tokens=row[6],
                    text=row[7],
                ),
                score=float(row[8]),
                rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def count(self) -> int:
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM {TABLE} WHERE namespace = %s", (self.namespace,)
        ).fetchone()
        return int(row[0]) if row else 0

    def all_chunks(self) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT chunk_id, doc_id, area, title, page_start, page_end, n_tokens, text"
            f" FROM {TABLE} WHERE namespace = %s ORDER BY id",
            (self.namespace,),
        ).fetchall()
        return [
            Chunk(
                chunk_id=r[0], doc_id=r[1], area=r[2], title=r[3],
                page_start=r[4], page_end=r[5], n_tokens=r[6], text=r[7],
            )
            for r in rows
        ]
