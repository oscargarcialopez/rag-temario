import numpy as np

from rag_temario.cache import EmbeddingCache, load_pages, save_pages
from rag_temario.corpus import Page


def test_embedding_cache_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    key = EmbeddingCache.key("model", 3, "TASK", "hola")
    cache.put_many([(key, np.array([1.0, 2.0, 3.0], dtype=np.float32))])
    found = cache.get_many([key])
    assert np.allclose(found[key], [1.0, 2.0, 3.0])
    assert len(cache) == 1


def test_embedding_cache_key_depends_on_model_dim_task_and_text():
    base = EmbeddingCache.key("m", 768, "t", "x")
    assert base != EmbeddingCache.key("m2", 768, "t", "x")
    assert base != EmbeddingCache.key("m", 1536, "t", "x")
    assert base != EmbeddingCache.key("m", 768, "t2", "x")
    assert base != EmbeddingCache.key("m", 768, "t", "y")


def test_embedding_cache_key_fields_cannot_run_into_each_other():
    # dim 76 + task "8t" must not collide with dim 768 + task "t"
    assert EmbeddingCache.key("m", 76, "8t", "x") != EmbeddingCache.key("m", 768, "t", "x")


def test_page_cache_is_invalidated_by_fingerprint(tmp_path):
    path = tmp_path / "pages.jsonl"
    pages = [Page("a/T", "a", "T", 1, "hola")]
    save_pages(path, "fp1", pages)
    assert load_pages(path, "fp1") == pages
    assert load_pages(path, "fp2") is None
