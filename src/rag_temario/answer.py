"""Grounded generation with citations, and the option to refuse.

The output schema forces two things the model would otherwise be happy to skip:
every claim carries a document and a page, and ``answered`` can be false. A RAG
system that cannot say 'this is not in the corpus' will invent an answer for
every question it is asked, which is the failure mode that makes people stop
trusting these systems.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from .stores.base import Hit

SYSTEM = """Responde EXCLUSIVAMENTE con la información de los fragmentos aportados, \
que proceden del temario oficial de la oposición al Cuerpo de Diplomados en Estadística \
del Estado.

Reglas:
- Si los fragmentos no contienen la respuesta, pon answered=false y explica qué falta.
  No uses conocimiento propio para rellenar huecos.
- Cada afirmación relevante debe apoyarse en una cita con documento y página.
- Responde en español, con precisión técnica y sin florituras."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answered": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document": {"type": "string"},
                    "page": {"type": "integer"},
                },
                "required": ["document", "page"],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["answered", "answer", "citations", "reason"],
}


@dataclass
class Answer:
    answered: bool
    answer: str
    citations: list[dict] = field(default_factory=list)
    reason: str = ""
    hits: list[Hit] = field(default_factory=list)

    def render(self) -> str:
        if not self.answered:
            return f"Sin respuesta en el corpus. {self.reason}"
        lines = [self.answer, ""]
        for citation in self.citations:
            lines.append(f"  — {citation.get('document')}, p. {citation.get('page')}")
        return "\n".join(lines)


def format_context(hits: Sequence[Hit]) -> str:
    return "\n\n".join(
        f"[{hit.rank}] documento: {hit.chunk.title} | área: {hit.chunk.area}"
        f" | página: {hit.chunk.page_start}\n{hit.chunk.text}"
        for hit in hits
    )


class Generator:
    def __init__(self, model: str, api_key: str) -> None:
        from google import genai

        self.model = model
        self._client = genai.Client(api_key=api_key)
        self.api_calls = 0

    def answer(self, question: str, hits: Sequence[Hit]) -> Answer:
        from google.genai import types

        if not hits:
            return Answer(False, "", reason="La búsqueda no devolvió ningún fragmento.")
        prompt = (
            f"{SYSTEM}\n\nFragmentos:\n{format_context(hits)}\n\nPregunta: {question}"
        )
        self.api_calls += 1
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SCHEMA,
                temperature=0.1,
            ),
        )
        payload = json.loads(response.text)
        return Answer(
            answered=bool(payload["answered"]),
            answer=payload.get("answer", ""),
            citations=payload.get("citations", []),
            reason=payload.get("reason", ""),
            hits=list(hits),
        )
