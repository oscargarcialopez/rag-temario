from rag_temario.normalize import build_vocabulary, repair_text, strip_boilerplate


def test_strip_boilerplate_removes_running_header_and_footer():
    pages = [
        f"ACME REPORT 2026\nBody of page {i}\nunique line {i}\nPage {i} of 8"
        for i in range(1, 9)
    ]
    cleaned = strip_boilerplate(pages)
    assert all("ACME REPORT" not in page for page in cleaned)
    assert all("of 8" not in page for page in cleaned)
    assert "Body of page 3" in cleaned[2]


def test_strip_boilerplate_keeps_numbered_body_lines():
    """Regression: digit masking made 'Pregunta 1.', 'Pregunta 2.' ... look like
    one repeated line, so the stripper deleted the questions it was cleaning.
    Boilerplate must also be positional — at the edge of the page."""
    pages = [
        f"Exam header\nBloque I - Estadistica\nPregunta {i}.\nExplain something here.\nPage {i}"
        for i in range(1, 9)
    ]
    cleaned = strip_boilerplate(pages)
    assert all(f"Pregunta {i}." in cleaned[i - 1] for i in range(1, 9))
    assert all("Exam header" not in page for page in cleaned)


def test_strip_boilerplate_leaves_short_documents_alone():
    pages = ["a", "b", "c"]
    assert strip_boilerplate(pages) == pages


def test_repair_text_merges_only_attested_content_words():
    vocab = build_vocabulary(["desarrollo " * 10, "el la de " * 10])
    repaired, merges = repair_text("el desa rrollo de la region", vocab)
    assert repaired == "el desarrollo de la region"
    assert merges == 1


def test_repair_text_does_not_merge_short_function_words():
    vocab = build_vocabulary(["en " * 50, "e n"])
    repaired, merges = repair_text("e n el texto", vocab)
    assert merges == 0
    assert repaired == "e n el texto"
