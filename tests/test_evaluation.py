import math

from rag_temario.chunking import Chunk
from rag_temario.evalset import EvalQuestion
from rag_temario.evaluation import (
    RunConfig,
    analytic_random_recall,
    majority_baseline,
    roc_auc,
    score_run,
)
from rag_temario.stores.base import Hit


def chunk(idx, area):
    return Chunk(f"c{idx}", f"{area}/T", area, "T", 1, 1, 10, "text")


def question(idx, area, in_corpus=True):
    return EvalQuestion(f"q{idx}", "text", area, in_corpus, "B", "exam.pdf", 2026)


def test_roc_auc_perfect_and_random_separation():
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5


def test_roc_auc_is_nan_without_both_classes():
    assert math.isnan(roc_auc([0.1, 0.2], [True, True]))


def test_analytic_random_recall_matches_hand_computation():
    chunks = [chunk(i, "a") for i in range(2)] + [chunk(i + 2, "b") for i in range(8)]
    questions = [question(0, "a")]
    # P(at least one of 2 draws from {2 of 10} in area a) = 1 - (8/10)(7/9)
    assert analytic_random_recall(chunks, questions, 2) == 1 - (8 / 10) * (7 / 9)


def test_analytic_random_recall_reaches_one_when_k_covers_everything():
    chunks = [chunk(i, "a") for i in range(3)]
    assert analytic_random_recall(chunks, [question(0, "a")], 3) == 1.0


def test_majority_baseline_uses_the_largest_area():
    chunks = [chunk(i, "big") for i in range(9)] + [chunk(9, "small")]
    questions = [question(0, "big"), question(1, "small")]
    assert majority_baseline(chunks, questions) == 0.5


def test_score_run_computes_recall_precision_and_mrr():
    chunks = [chunk(i, "a") for i in range(5)] + [chunk(i + 5, "b") for i in range(5)]
    hits = [Hit(chunk(0, "b"), 0.9, 1), Hit(chunk(1, "a"), 0.8, 2), Hit(chunk(2, "a"), 0.7, 3)]
    config = RunConfig(800, 100, "window", "dense", "hash", "local")
    result = score_run([(question(0, "a"), hits)], chunks, config, ks=(1, 3))

    assert result.recall_at[1] == 0.0          # rank 1 is the wrong area
    assert result.recall_at[3] == 1.0
    assert result.precision_at[3] == 2 / 3
    assert result.mrr == 0.5                   # first correct chunk at rank 2
    assert result.recall_by_area == {"a": 1.0}


def test_score_run_excludes_out_of_corpus_questions_from_recall():
    chunks = [chunk(0, "a")]
    hits = [Hit(chunk(0, "a"), 0.9, 1)]
    config = RunConfig(800, 100, "window", "dense", "hash", "local")
    result = score_run(
        [(question(0, "a"), hits), (question(1, None, in_corpus=False), hits)],
        chunks, config, ks=(1,),
    )
    assert result.n_questions == 1
    assert result.recall_at[1] == 1.0
