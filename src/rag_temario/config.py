"""Runtime configuration, resolved once from the environment.

Every path in ``.env`` is interpreted relative to the repository root, so the
same file works no matter which directory the CLI is invoked from.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Where the PDFs are expected when nothing is configured: ``corpus/<area>/*.pdf``
# for the syllabus and ``exams/**/*.pdf`` for the official exams. Both are
# git-ignored; point RAG_CORPUS_DIR / RAG_EXAMS_DIR anywhere else to override.
DEFAULT_CORPUS_DIR = "corpus"
DEFAULT_EXAMS_DIR = "exams"

DEFAULT_EMBED_MODEL = "gemini-embedding-2"
# 768 is one of the sizes Google recommends for this family (768 / 1536 / 3072)
# and keeps a pgvector HNSW index under that extension's 2,000-dimension limit.
DEFAULT_EMBED_DIM = 768


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    exams_dir: Path
    embed_model: str
    embed_dim: int
    chat_model: str
    store: str
    pg_dsn: str
    api_key: str

    @property
    def data_dir(self) -> Path:
        return ROOT / "data"

    @property
    def cache_db(self) -> Path:
        return self.data_dir / "cache" / "embeddings.sqlite"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def eval_set(self) -> Path:
        return ROOT / "eval" / "eval_set.jsonl"

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def settings_from_env(env: Mapping[str, str]) -> Settings:
    """Build settings from a mapping, so tests can pass one instead of mutating os.environ."""
    raw_dim = env.get("RAG_EMBED_DIM", "").strip()
    try:
        embed_dim = int(raw_dim) if raw_dim else DEFAULT_EMBED_DIM
    except ValueError:
        raise ValueError(f"RAG_EMBED_DIM must be an integer, got {raw_dim!r}") from None
    return Settings(
        corpus_dir=_path(env.get("RAG_CORPUS_DIR", DEFAULT_CORPUS_DIR)),
        exams_dir=_path(env.get("RAG_EXAMS_DIR", DEFAULT_EXAMS_DIR)),
        embed_model=env.get("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        embed_dim=embed_dim,
        chat_model=env.get("RAG_CHAT_MODEL", "gemini-2.5-flash"),
        store=env.get("RAG_STORE", "local"),
        pg_dsn=env.get("RAG_PG_DSN", ""),
        api_key=env.get("GEMINI_API_KEY", ""),
    )


@lru_cache(maxsize=1)
def settings() -> Settings:
    load_dotenv(ROOT / ".env")
    cfg = settings_from_env(os.environ)
    for directory in (cfg.cache_db.parent, cfg.index_dir, cfg.runs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return cfg
