"""Repair words that PDF extraction split in two.

pypdf occasionally emits an intra-word space where the source font used tight
kerning: 'envejecida' comes out as 'env ejecida', 'desarrollo' as 'desa rrollo'.
Dense retrieval tolerates this; lexical retrieval does not, because 'desa' and
'rrollo' simply are not the query term.

The repair is deliberately narrow. Measured on this corpus, 0.30% of alphabetic
tokens look like a split, but the overwhelming majority are function words
('e'+'n' -> 'en') that no retriever depends on. Only merges producing a content
word of at least ``min_length`` characters are applied, and only when the merged
form is attested in the corpus itself more often than either fragment — so the
corpus is its own dictionary and no external word list is needed.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")


def build_vocabulary(texts: Iterable[str]) -> Counter[str]:
    vocab: Counter[str] = Counter()
    for text in texts:
        vocab.update(word.lower() for word in WORD.findall(text))
    return vocab


def repair_text(
    text: str,
    vocab: Counter[str],
    *,
    min_length: int = 6,
    min_frequency: int = 5,
) -> tuple[str, int]:
    """Return the repaired text and the number of merges applied."""
    merges = 0
    out: list[str] = []
    tokens = re.split(r"(\s+)", text)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        # A candidate is 'word', ' ', 'word' with exactly one space between.
        if (
            index + 2 < len(tokens)
            and tokens[index + 1] == " "
            and WORD.fullmatch(token)
            and WORD.fullmatch(tokens[index + 2])
        ):
            nxt = tokens[index + 2]
            merged = (token + nxt).lower()
            if (
                len(merged) >= min_length
                and vocab[merged] >= min_frequency
                and vocab[merged] > vocab[token.lower()]
                and vocab[merged] > vocab[nxt.lower()]
            ):
                out.append(token + nxt)
                merges += 1
                index += 3
                continue
        out.append(token)
        index += 1
    return "".join(out), merges


_DIGITS = re.compile(r"\d+")


def strip_boilerplate(page_texts: list[str], *, threshold: float = 0.6, edge: int = 2) -> list[str]:
    """Remove running headers and footers.

    Rather than hard-coding one publisher's header, a line is boilerplate when it
    (a) sits within ``edge`` lines of the top or bottom of its page and (b) with
    its digits masked ('Página 3 de 8' -> 'Página # de #') appears on at least
    ``threshold`` of the pages.

    The positional condition is not decoration. Without it, digit masking makes
    'Pregunta 1.', 'Pregunta 2.' ... collapse into one repeated line and the
    function deletes the body of the document it was meant to clean — which is
    exactly what it did on first run against the exam PDFs.
    """
    if len(page_texts) < 4:
        return page_texts

    def edge_indices(lines: list[str]) -> set[int]:
        filled = [i for i, line in enumerate(lines) if line.strip()]
        # A header cannot be the whole page: on a short page the band shrinks so
        # that at least one line in the middle is never a removal candidate.
        band = min(edge, max(1, (len(filled) - 1) // 2))
        return set(filled[:band]) | set(filled[-band:])

    counts: Counter[str] = Counter()
    for text in page_texts:
        lines = text.splitlines()
        counts.update({_DIGITS.sub("#", lines[i].strip()) for i in edge_indices(lines)})

    minimum = max(3, int(threshold * len(page_texts)))
    boilerplate = {line for line, count in counts.items() if count >= minimum and len(line) > 3}
    if not boilerplate:
        return page_texts

    cleaned = []
    for text in page_texts:
        lines = text.splitlines()
        edges = edge_indices(lines)
        kept = [
            line
            for i, line in enumerate(lines)
            if not (i in edges and _DIGITS.sub("#", line.strip()) in boilerplate)
        ]
        cleaned.append("\n".join(kept).strip())
    return cleaned
