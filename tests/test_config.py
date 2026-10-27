import pytest

from rag_temario.cli import report_file
from rag_temario.config import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_EMBED_DIM,
    DEFAULT_EMBED_MODEL,
    DEFAULT_EXAMS_DIR,
    ROOT,
    settings_from_env,
)
from rag_temario.embeddings import KNOWN_DIMENSIONS


def test_defaults_are_the_current_model_and_generic_paths():
    cfg = settings_from_env({})

    assert cfg.embed_model == "gemini-embedding-2"
    assert cfg.embed_dim == 768
    assert cfg.corpus_dir == ROOT / "corpus"
    assert cfg.exams_dir == ROOT / "exams"
    assert not cfg.has_api_key
    assert cfg.store == "local"


def test_environment_overrides_every_default():
    cfg = settings_from_env(
        {
            "RAG_EMBED_MODEL": "gemini-embedding-001",
            "RAG_EMBED_DIM": "1536",
            "RAG_CORPUS_DIR": "/somewhere/else/pdfs",
            "RAG_EXAMS_DIR": "mine/exams",
            "GEMINI_API_KEY": "not-a-real-key",
        }
    )

    assert (cfg.embed_model, cfg.embed_dim) == ("gemini-embedding-001", 1536)
    assert str(cfg.corpus_dir) == "/somewhere/else/pdfs"
    assert cfg.exams_dir == ROOT / "mine" / "exams"  # relative paths hang off the repo root
    assert cfg.has_api_key


def test_a_blank_dimension_falls_back_to_the_default():
    assert settings_from_env({"RAG_EMBED_DIM": " "}).embed_dim == DEFAULT_EMBED_DIM


def test_a_dimension_that_is_not_a_number_is_reported_by_name():
    with pytest.raises(ValueError, match="RAG_EMBED_DIM"):
        settings_from_env({"RAG_EMBED_DIM": "lots"})


def test_env_example_agrees_with_the_code_defaults():
    """The model name was once stale in four places at once; keep the example honest."""
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    example = dict(
        line.split("=", 1) for line in lines if "=" in line and not line.startswith("#")
    )

    assert example["RAG_EMBED_MODEL"] == DEFAULT_EMBED_MODEL
    assert int(example["RAG_EMBED_DIM"]) == DEFAULT_EMBED_DIM
    assert example["RAG_CORPUS_DIR"] == DEFAULT_CORPUS_DIR
    assert example["RAG_EXAMS_DIR"] == DEFAULT_EXAMS_DIR
    assert DEFAULT_EMBED_MODEL in KNOWN_DIMENSIONS
    assert KNOWN_DIMENSIONS[DEFAULT_EMBED_MODEL] >= DEFAULT_EMBED_DIM


def test_each_embedder_gets_its_own_results_file(tmp_path):
    offline = report_file(tmp_path, "hash-256")
    real = report_file(tmp_path, "gemini-embedding-2-768")

    assert offline.name == "RESULTS-hash-256.md"
    assert real.name == "RESULTS-gemini-embedding-2-768.md"
    assert offline != real


def test_a_partial_run_never_overwrites_the_full_table(tmp_path):
    full = report_file(tmp_path, "gemini-embedding-2-768")
    partial = report_file(tmp_path, "gemini-embedding-2-768", partial=True)

    assert partial.name == "RESULTS-gemini-embedding-2-768-partial.md"
    assert partial != full
