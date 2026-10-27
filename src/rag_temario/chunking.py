"""Split page-anchored text into overlapping chunks.

Chunk size is the first real knob in a RAG system and the one most often chosen
by copying a tutorial. It is treated here as an experimental parameter: the
chunker is deterministic and configurable, and `eval/run_eval.py` sweeps it.

Tokenisation uses ``cl100k_base`` as a stable, offline, reproducible proxy. It
is not the tokenizer the embedding model uses, but it is deterministic across
machines and free to run, which matters more for comparing configurations than
matching the provider's count exactly.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from .corpus import Page


class Tokenizer(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class _RegexTokenizer:
    """Fallback used when tiktoken is unavailable. Round-trips exactly."""

    name = "regex-word"
    _SPLIT = re.compile(r"\s+|\S+")

    def __init__(self) -> None:
        self._vocab: list[str] = []
        self._index: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in self._SPLIT.findall(text):
            if piece not in self._index:
                self._index[piece] = len(self._vocab)
                self._vocab.append(piece)
            ids.append(self._index[piece])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self._vocab[i] for i in ids)


def get_tokenizer() -> Tokenizer:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 - any failure means "use the fallback"
        return _RegexTokenizer()

    class _Tiktoken:
        name = "cl100k_base"

        def encode(self, text: str) -> list[int]:
            return encoding.encode(text, disallowed_special=())

        def decode(self, ids: Sequence[int]) -> str:
            return encoding.decode(list(ids))

    return _Tiktoken()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    area: str
    title: str
    page_start: int
    page_end: int
    n_tokens: int
    text: str

    @property
    def citation(self) -> str:
        if self.page_start == self.page_end:
            return f"{self.title}, p. {self.page_start}"
        return f"{self.title}, pp. {self.page_start}-{self.page_end}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MIN_CHUNK_TOKENS = 24


def chunk_pages(
    pages: Sequence[Page],
    tokenizer: Tokenizer,
    *,
    size: int = 800,
    overlap: int = 100,
    strategy: str = "window",
) -> list[Chunk]:
    """Chunk every document in ``pages``, preserving page provenance.

    ``strategy="page"`` emits one chunk per page and ignores size/overlap; it is
    the honest baseline to beat, since the source documents are already split
    into semantically coherent pages by their authors.
    """
    if strategy not in {"window", "page"}:
        raise ValueError(f"unknown strategy: {strategy}")
    if strategy == "window" and not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size); got size={size} overlap={overlap}")

    by_doc: dict[str, list[Page]] = {}
    for page in pages:
        by_doc.setdefault(page.doc_id, []).append(page)

    chunks: list[Chunk] = []
    for doc_id, doc_pages in by_doc.items():
        doc_pages = sorted(doc_pages, key=lambda p: p.page)
        first = doc_pages[0]
        if strategy == "page":
            for page in doc_pages:
                ids = tokenizer.encode(page.text)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}#p{page.page}",
                        doc_id=doc_id,
                        area=page.area,
                        title=page.title,
                        page_start=page.page,
                        page_end=page.page,
                        n_tokens=len(ids),
                        text=page.text,
                    )
                )
            continue

        # Flatten the document into one token stream, remembering which page
        # each token came from, so a chunk that straddles a page break can still
        # cite the exact range.
        ids: list[int] = []
        page_of_token: list[int] = []
        for page in doc_pages:
            token_ids = tokenizer.encode(page.text + "\n\n")
            ids.extend(token_ids)
            page_of_token.extend([page.page] * len(token_ids))

        step = size - overlap
        for index, start in enumerate(range(0, max(len(ids), 1), step)):
            window = ids[start : start + size]
            if len(window) < MIN_CHUNK_TOKENS and chunks and chunks[-1].doc_id == doc_id:
                break  # trailing scrap already covered by the overlap
            if not window:
                break
            text = tokenizer.decode(window).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{index}",
                    doc_id=doc_id,
                    area=first.area,
                    title=first.title,
                    page_start=page_of_token[start],
                    page_end=page_of_token[min(start + size, len(ids)) - 1],
                    n_tokens=len(window),
                    text=text,
                )
            )
            if start + size >= len(ids):
                break
    return chunks
