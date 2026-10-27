import pytest

from rag_temario.chunking import Chunk, chunk_pages, get_tokenizer
from rag_temario.corpus import Page


def make_pages(n_pages=4, words_per_page=400):
    return [
        Page(
            doc_id="estadistica/Tema 1",
            area="estadistica",
            title="Tema 1",
            page=i,
            text=" ".join(f"palabra{i}x{j}" for j in range(words_per_page)),
        )
        for i in range(1, n_pages + 1)
    ]


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer()


def test_chunks_respect_size_and_cover_the_document(tokenizer):
    chunks = chunk_pages(make_pages(), tokenizer, size=200, overlap=20)
    assert chunks, "expected at least one chunk"
    assert all(c.n_tokens <= 200 for c in chunks)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert {c.doc_id for c in chunks} == {"estadistica/Tema 1"}


def test_consecutive_chunks_overlap(tokenizer):
    chunks = chunk_pages(make_pages(), tokenizer, size=200, overlap=50)
    first_tail = tokenizer.encode(chunks[0].text)[-30:]
    second = tokenizer.encode(chunks[1].text)
    assert any(first_tail[0] == token for token in second[:60])


def test_page_provenance_is_monotonic_and_within_range(tokenizer):
    chunks = chunk_pages(make_pages(6), tokenizer, size=300, overlap=30)
    for chunk in chunks:
        assert 1 <= chunk.page_start <= chunk.page_end <= 6
    assert [c.page_start for c in chunks] == sorted(c.page_start for c in chunks)


def test_page_strategy_emits_one_chunk_per_page(tokenizer):
    chunks = chunk_pages(make_pages(5), tokenizer, strategy="page")
    assert len(chunks) == 5
    assert [c.page_start for c in chunks] == [1, 2, 3, 4, 5]


def test_citation_formats_single_and_multi_page(tokenizer):
    single = Chunk("a", "d", "area", "Tema 1", 3, 3, 10, "x")
    spread = Chunk("b", "d", "area", "Tema 1", 3, 5, 10, "x")
    assert single.citation == "Tema 1, p. 3"
    assert spread.citation == "Tema 1, pp. 3-5"


@pytest.mark.parametrize("size,overlap", [(100, 100), (100, 150), (100, -1)])
def test_invalid_overlap_is_rejected(tokenizer, size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(make_pages(1), tokenizer, size=size, overlap=overlap)
