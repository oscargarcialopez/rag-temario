import numpy as np

from rag_temario.chunking import Chunk
from rag_temario.embeddings import HashEmbedder, l2_normalise
from rag_temario.retrieval import BM25Index, Retriever, reciprocal_rank_fusion, tokenize
from rag_temario.stores.base import Hit
from rag_temario.stores.local import LocalStore


def chunk(idx, text, area="estadistica"):
    return Chunk(f"c{idx}", f"{area}/T", area, "T", 1, 1, len(text.split()), text)


CHUNKS = [
    chunk(0, "La tasa bruta de mortalidad se calcula dividiendo las defunciones", "demografia"),
    chunk(1, "El estimador insesgado tiene esperanza igual al parametro", "estadistica"),
    chunk(2, "El muestreo estratificado divide la poblacion en estratos", "muestreo"),
    chunk(3, "La balanza de pagos registra las transacciones con el exterior", "economia"),
]


def test_tokenize_folds_accents_and_drops_stopwords():
    assert tokenize("La Estadística de los hogares") == ["estadistica", "hogares"]


def test_bm25_ranks_the_lexically_matching_chunk_first():
    index = BM25Index(CHUNKS)
    hits = index.search("tasa bruta de mortalidad", k=3)
    assert hits[0].chunk.chunk_id == "c0"
    assert hits[0].rank == 1


def test_bm25_returns_nothing_for_unknown_terms():
    assert BM25Index(CHUNKS).search("zzzz qqqq", k=3) == []


def test_reciprocal_rank_fusion_prefers_consensus():
    a = [Hit(CHUNKS[0], 1.0, 1), Hit(CHUNKS[1], 0.9, 2)]
    b = [Hit(CHUNKS[1], 5.0, 1), Hit(CHUNKS[2], 4.0, 2)]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0].chunk.chunk_id == "c1"  # ranked in both lists
    assert [h.rank for h in fused] == [1, 2, 3]


def test_local_store_roundtrip_and_ordering(tmp_path):
    store = LocalStore(tmp_path, "unit")
    vectors = l2_normalise(np.eye(4, dtype=np.float32))
    store.reset(4)
    store.add(CHUNKS, vectors)
    assert store.count() == 4

    hits = store.search(vectors[2], k=2)
    assert hits[0].chunk.chunk_id == "c2"
    assert hits[0].score > hits[1].score
    assert [h.rank for h in hits] == [1, 2]

    reopened = LocalStore(tmp_path, "unit")
    assert reopened.count() == 4


def test_local_store_rejects_mismatched_lengths(tmp_path):
    store = LocalStore(tmp_path, "unit2")
    store.reset(4)
    try:
        store.add(CHUNKS, np.zeros((3, 4), dtype=np.float32))
    except ValueError as error:
        assert "chunks" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_retriever_modes_agree_on_an_easy_query(tmp_path):
    embedder = HashEmbedder(64)
    store = LocalStore(tmp_path, "modes")
    store.reset(embedder.dim)
    store.add(CHUNKS, embedder.embed_documents([c.text for c in CHUNKS]))
    query = "muestreo estratificado de la poblacion"
    for mode in ("dense", "bm25", "hybrid"):
        hits = Retriever(store, embedder, mode=mode).retrieve(query, k=4)
        assert hits[0].chunk.area == "muestreo", mode
