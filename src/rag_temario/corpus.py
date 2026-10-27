"""Turn the source PDFs into clean, page-anchored text.

Page anchoring is not an implementation detail: the whole point of the system is
to answer with a citation the reader can check, so every unit of text keeps the
document and the page it came from all the way to the final answer.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .normalize import strip_boilerplate

# pypdf logs a wall of font-encoding warnings on these PDFs; they are noise.
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Areas of the syllabus. The folder names are the canonical identifiers; the
# official exams label their blocks with accented Spanish, hence the fold.
KNOWN_AREAS = ("demografia", "economia", "estadistica", "informatica", "muestreo")

# Bumped whenever the parsing or cleaning rules change, so the page cache is
# invalidated by a change in the code and not only by a change in the PDFs.
PARSER_VERSION = "2"


def fold(text: str) -> str:
    """Lowercase and strip accents, so 'Estadística' and 'estadistica' match."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower().strip()


def area_from_block(block_name: str) -> str | None:
    """Map an exam block heading to a corpus area, or None if out of corpus."""
    folded = fold(block_name)
    for area in KNOWN_AREAS:
        if area in folded:
            return area
    return None


@dataclass(frozen=True)
class Page:
    """One page of one document."""

    doc_id: str
    area: str
    title: str
    page: int
    text: str


_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$")
_WHITESPACE = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_page_text(raw: str) -> str:
    """Drop standalone page numbers and collapse the whitespace pypdf leaves."""
    lines = []
    for line in raw.splitlines():
        if _PAGE_NUMBER_LINE.match(line):
            continue
        lines.append(_WHITESPACE.sub(" ", line).strip())
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def read_pdf(path: Path, area: str) -> Iterator[Page]:
    title = path.stem
    doc_id = f"{area}/{title}"
    reader = PdfReader(str(path))
    raw = [clean_page_text(page.extract_text() or "") for page in reader.pages]
    for number, text in enumerate(strip_boilerplate(raw), start=1):
        if len(text) < 40:  # covers, separators, and pages that are a single figure
            continue
        yield Page(doc_id=doc_id, area=area, title=title, page=number, text=text)


def load_corpus(corpus_dir: Path) -> list[Page]:
    """Read every PDF under ``corpus_dir``; the sub-folder name is the area."""
    pages: list[Page] = []
    for pdf in sorted(corpus_dir.rglob("*.pdf")):
        area = fold(pdf.parent.name)
        if area not in KNOWN_AREAS:
            continue
        pages.extend(read_pdf(pdf, area))
    return pages


def corpus_stats(pages: list[Page]) -> dict[str, object]:
    by_area: dict[str, int] = {}
    docs: set[str] = set()
    for page in pages:
        by_area[page.area] = by_area.get(page.area, 0) + 1
        docs.add(page.doc_id)
    return {
        "documents": len(docs),
        "pages": len(pages),
        "characters": sum(len(p.text) for p in pages),
        "pages_by_area": dict(sorted(by_area.items())),
    }
