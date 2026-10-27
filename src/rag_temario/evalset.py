"""Build the labelled evaluation set from the official exams.

The usual advice is to hand-write ~50 questions. That produces an eval set
biased towards the phrasing of whoever wrote the system, which is the one bias
an eval set must not have. This corpus comes with something better: the real
exams, published by the examining body, whose questions are grouped under headings that
name the syllabus area they belong to. The heading is the label — free, real,
and written by the examiner rather than by the engineer being evaluated.

One extra gift: the exams contain a block ('Derecho Público y Organización del
Estado') whose syllabus is *not* in the corpus. Those questions are kept and
labelled ``in_corpus=False``. A retriever always returns its top-k, so it never
says 'I don't know'; these questions measure whether the system as a whole can.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

from .corpus import area_from_block, clean_page_text, fold
from .normalize import strip_boilerplate

BLOCK = re.compile(r"Bloque\s+([IVX]+)\s*[-–.]\s*([^\n]{3,60})")
QUESTION = re.compile(r"Pregunta\s+(\d+)\s*[.\-]?\s*")
YEAR = re.compile(r"(20\d{2})")

MIN_QUESTION_CHARS = 40
MAX_QUESTION_CHARS = 1200


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    area: str | None
    in_corpus: bool
    block: str
    source: str
    year: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _exam_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [clean_page_text(page.extract_text() or "") for page in reader.pages]
    return "\n".join(strip_boilerplate(pages))


def _clean_question(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # Drop trailing administrative noise that follows the last question.
    text = re.split(r"(?:NO ESCRIBA|FIN DEL (?:CUESTIONARIO|EJERCICIO))", text)[0]
    return text.strip(" .;:-")


def parse_exam(path: Path) -> Iterator[EvalQuestion]:
    """Yield every question in one exam, labelled with its block's area."""
    text = _exam_text(path)
    year_match = YEAR.search(path.stem)
    year = int(year_match.group(1)) if year_match else None
    slug = re.sub(r"[^a-z0-9]+", "-", fold(path.stem)).strip("-")

    blocks = list(BLOCK.finditer(text))
    if not blocks:
        return
    for index, match in enumerate(blocks):
        block_name = match.group(2).strip()
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(text)
        body = text[match.end() : end]
        area = area_from_block(block_name)

        questions = list(QUESTION.finditer(body))
        for position, question_match in enumerate(questions):
            stop = (
                questions[position + 1].start()
                if position + 1 < len(questions)
                else len(body)
            )
            content = _clean_question(body[question_match.end() : stop])
            if not MIN_QUESTION_CHARS <= len(content) <= MAX_QUESTION_CHARS:
                continue
            yield EvalQuestion(
                id=f"{slug}-b{index + 1}-q{question_match.group(1)}",
                question=content,
                area=area,
                in_corpus=area is not None,
                block=block_name,
                source=path.name,
                year=year,
            )


def build_eval_set(exams_dir: Path) -> list[EvalQuestion]:
    """Parse every exam under ``exams_dir`` and de-duplicate repeated questions."""
    seen: set[str] = set()
    questions: list[EvalQuestion] = []
    for pdf in sorted(exams_dir.rglob("*.pdf")):
        for question in parse_exam(pdf):
            fingerprint = fold(re.sub(r"\W+", "", question.question))[:200]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            questions.append(question)
    return questions


def write_eval_set(path: Path, questions: list[EvalQuestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question.to_dict(), ensure_ascii=False) + "\n")


def read_eval_set(path: Path) -> list[EvalQuestion]:
    with path.open(encoding="utf-8") as handle:
        return [EvalQuestion(**json.loads(line)) for line in handle if line.strip()]


def eval_set_stats(questions: list[EvalQuestion]) -> dict[str, object]:
    by_area: dict[str, int] = {}
    for question in questions:
        key = question.area or f"(out of corpus) {question.block}"
        by_area[key] = by_area.get(key, 0) + 1
    return {
        "questions": len(questions),
        "in_corpus": sum(q.in_corpus for q in questions),
        "out_of_corpus": sum(not q.in_corpus for q in questions),
        "sources": len({q.source for q in questions}),
        "by_area": dict(sorted(by_area.items(), key=lambda kv: -kv[1])),
    }
