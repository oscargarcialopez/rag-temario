"""Re-ranking: ask a model to reorder what similarity retrieved.

Retrieval optimises for 'looks like the query in embedding space'. Relevance is
a different question, and a cheap model reading 20 passages answers it better
than cosine does. The whole point of retrieving k=20 and keeping 5 is to let
this step throw away the 15 that only looked right.

One call scores all candidates together, because the marginal cost of an extra
passage in a prompt is far lower than the fixed cost of another round trip, and
because a model comparing candidates side by side is more consistent than one
scoring them in isolation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .stores.base import Hit

PROMPT = """Eres un evaluador de relevancia para un sistema de búsqueda sobre el temario \
de la oposición al Cuerpo de Diplomados en Estadística del Estado.

Pregunta:
{query}

Fragmentos candidatos:
{candidates}

Puntúa cada fragmento de 0 a 10 según su utilidad REAL para responder la pregunta:
10 = contiene la respuesta; 5 = trata el tema pero no responde; 0 = irrelevante.
Responde solo con el JSON pedido, sin texto adicional."""

SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["id", "score"],
            },
        }
    },
    "required": ["scores"],
}


class NullReranker:
    """Keeps the retrieval order. The control condition, not a placeholder."""

    name = "none"

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        return list(hits)


class LLMReranker:
    def __init__(self, model: str, api_key: str, *, snippet_chars: int = 700) -> None:
        from google import genai

        self.name = f"llm:{model}"
        self.model = model
        self._client = genai.Client(api_key=api_key)
        self._snippet_chars = snippet_chars
        self.api_calls = 0

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if len(hits) < 2:
            return list(hits)
        from google.genai import types

        candidates = "\n\n".join(
            f"[{index}] ({hit.chunk.citation}) {hit.chunk.text[: self._snippet_chars]}"
            for index, hit in enumerate(hits)
        )
        try:
            self.api_calls += 1
            response = self._client.models.generate_content(
                model=self.model,
                contents=PROMPT.format(query=query, candidates=candidates),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SCHEMA,
                    temperature=0.0,
                ),
            )
            scores = {
                int(item["id"]): float(item["score"])
                for item in json.loads(response.text)["scores"]
            }
        except Exception:  # noqa: BLE001 - a failed re-rank must not fail the query
            return list(hits)

        order = sorted(
            range(len(hits)),
            key=lambda index: (-scores.get(index, -1.0), hits[index].rank),
        )
        return [
            Hit(chunk=hits[index].chunk, score=scores.get(index, 0.0), rank=rank)
            for rank, index in enumerate(order, start=1)
        ]


def build_reranker(cfg, *, enabled: bool = True):
    if not enabled or not cfg.has_api_key:
        return NullReranker()
    return LLMReranker(cfg.chat_model, cfg.api_key)
