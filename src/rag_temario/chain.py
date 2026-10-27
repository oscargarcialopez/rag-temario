"""LangChain surface over the same retrieval stack.

Two entry points exist on purpose:

* ``pipeline.run_experiment`` calls the retriever and the generator directly.
  The evaluation harness uses that path, because measuring a system means
  controlling every step of it, and a framework in the middle only adds places
  for a default to change the numbers.
* ``build_chain`` exposes the same stack as a LangChain runnable, so the
  retriever composes with anything else in that ecosystem — memory, agents,
  ``LangServe`` — without rewriting the retrieval logic.

The retriever below is a real ``BaseRetriever``: metadata carries the citation
fields through, so a chain consuming it can still say which page it used.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from .answer import SYSTEM
from .retrieval import Retriever
from .stores.base import Hit

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "Fragmentos:\n{context}\n\nPregunta: {question}")]
)


def hit_to_document(hit: Hit) -> Document:
    chunk = hit.chunk
    return Document(
        id=chunk.chunk_id,
        page_content=chunk.text,
        metadata={
            "document": chunk.title,
            "doc_id": chunk.doc_id,
            "area": chunk.area,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "citation": chunk.citation,
            "score": hit.score,
            "rank": hit.rank,
        },
    )


class TemarioRetriever(BaseRetriever):
    """Adapts the project's retriever (plus optional re-ranking) to LangChain."""

    retriever: Retriever
    k: int = 20
    top_n: int = 5
    reranker: Any = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        hits = self.retriever.retrieve(query, self.k)
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits)
        return [hit_to_document(hit) for hit in hits[: self.top_n]]


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('rank')}] documento: {doc.metadata.get('document')}"
        f" | área: {doc.metadata.get('area')} | página: {doc.metadata.get('page_start')}"
        f"\n{doc.page_content}"
        for doc in documents
    )


def build_chain(retriever: BaseRetriever, model: str, api_key: str, *, temperature: float = 0.1):
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)
    return (
        RunnableParallel(
            context=retriever | RunnableLambda(format_documents),
            question=RunnablePassthrough(),
        )
        | PROMPT
        | llm
        | StrOutputParser()
    )
