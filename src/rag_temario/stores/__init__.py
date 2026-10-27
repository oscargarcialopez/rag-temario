"""Vector stores. Two backends, one interface."""

from .base import Hit, VectorStore
from .local import LocalStore
from .pg import PgVectorStore

__all__ = ["Hit", "VectorStore", "LocalStore", "PgVectorStore", "build_store"]


def build_store(cfg, namespace: str) -> VectorStore:
    if cfg.store == "pgvector":
        if not cfg.pg_dsn:
            raise RuntimeError("RAG_STORE=pgvector but RAG_PG_DSN is empty")
        return PgVectorStore(cfg.pg_dsn, namespace)
    return LocalStore(cfg.index_dir, namespace)
