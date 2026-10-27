"""The eval-set parser is the one piece where a silent failure is invisible:
a regex that stops matching yields an empty eval set, and an empty eval set
reports perfect scores on nothing. These tests pin the shape of the real exams."""

from rag_temario.corpus import area_from_block, fold
from rag_temario.evalset import EvalQuestion, build_eval_set, eval_set_stats


def test_fold_strips_accents_and_case():
    assert fold("Estadística") == "estadistica"
    assert fold(" DEMOGRAFÍA ") == "demografia"


def test_area_from_block_maps_exam_headings_to_corpus_folders():
    assert area_from_block("Estadística") == "estadistica"
    assert area_from_block("Muestreo") == "muestreo"
    assert area_from_block("Derecho Público y Organización del Estado") is None


def test_eval_set_stats_counts_in_and_out_of_corpus():
    questions = [
        EvalQuestion("a", "q", "estadistica", True, "Estadística", "e.pdf", 2026),
        EvalQuestion("b", "q", None, False, "Derecho", "e.pdf", 2026),
    ]
    stats = eval_set_stats(questions)
    assert stats["questions"] == 2
    assert stats["in_corpus"] == 1
    assert stats["out_of_corpus"] == 1


def test_build_eval_set_on_the_real_exams(tmp_path):
    """Skipped when the source PDFs are not present (they are not in the repo)."""
    import pytest

    from rag_temario.config import settings

    cfg = settings()
    if not cfg.exams_dir.exists():
        pytest.skip("exam PDFs not available")
    questions = build_eval_set(cfg.exams_dir)
    assert len(questions) >= 50, "the exams should yield a usable eval set"
    assert any(not q.in_corpus for q in questions), "expected out-of-corpus questions"
    assert all(q.question.strip() for q in questions)
    assert len({q.id for q in questions}) == len(questions), "ids must be unique"
