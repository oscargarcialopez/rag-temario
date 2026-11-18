# AGENTS.md

## What this is

Retrieval-augmented question answering over an official syllabus of Spanish public statistics,
with answers that cite the document and page they come from. The core of the repo is the
evaluation harness: retrieval is scored against real questions from the official exams, labelled
by the examiners' own headings. Pipeline, results and limitations are in [README.md](README.md);
read them there instead of restating them here.

## Layout

- `src/rag_temario/`: the package. `cli.py` is the `rag` command; `pipeline.py` wires corpus,
  chunks, embeddings, index, retrieval and metrics; `stores/` has the exact NumPy store and pgvector.
- `tests/`: the pytest suite. It runs offline, with fake API clients: no key, no PDFs.
- `eval/eval_set.jsonl`: the labelled eval set, parsed from the official exams.
- `eval/RESULTS-<embedder>.md`: results tables written by `rag eval`; the README quotes them.
- `data/`: generated, git-ignored: `cache/` (`embeddings.sqlite`, parsed pages), `index/`, `runs/`.
- `corpus/`, `exams/` (git-ignored): default folders for the source PDFs, which are not in the
  repo; `RAG_CORPUS_DIR` and `RAG_EXAMS_DIR` point elsewhere.
- `.env.example`: the settings template. Copy it to `.env`, which is git-ignored.
- `Makefile`: the interface. Each target is one line over the `rag` CLI.

## Commands

```bash
make install        # uv sync: .venv with runtime and dev dependencies
make lint           # ruff check src tests
make test           # pytest -q; offline, no key needed
make check          # lint + test: exactly what CI runs
make stats          # corpus and eval-set summary; needs the corpus PDFs
make eval-set       # rebuild eval/eval_set.jsonl from the exam PDFs
make eval-offline   # free sweep, hash embedder; needs the PDFs, rewrites eval/RESULTS-hash-256.md
make eval           # PAID: full sweep with the real embedding model
make index          # PAID: chunk, embed and index the corpus (800/100, dense)
make ask Q="..."    # PAID: answer one question with citations (embed, re-rank, generate)
make clean          # delete data/index, data/runs and data/cache, embedding cache included
uv run pytest tests/<file>.py::<test> -q   # a single test
```

## Rules

- **Paid API calls need the owner's explicit OK.** `make eval`, `make index`, `make ask` and
  `rag chain` call the Gemini API whenever `GEMINI_API_KEY` is set, in the environment or in `.env`.
  Embeddings are cached in `data/cache/embeddings.sqlite` by model, dimension, task and text, so a
  repeat sweep makes no requests, but any change to parsing, chunking or the model becomes billed
  cache misses. The free paths are `make check` and `make eval-offline`. `make clean` needs the
  same OK: it deletes the cache, so the next sweep re-embeds the whole corpus.
- **Numbers come from generated output, never from the prose.** A figure in the README, or in any
  post about this repo, must come from `eval/RESULTS-<embedder>.md` (written by `rag eval`), the
  counts printed by `make stats`, or the run records in `data/runs/*.json`. There is no automated
  numbers check, so compare by hand. To change a number, regenerate its source and copy from it;
  never change a figure in the README alone, and never hand-edit a `RESULTS-*.md` table.
- **The eval set belongs to the examiners.** `eval/eval_set.jsonl` is parsed from the official exams
  and its labels come from their block headings. Never edit it, or the heading-to-area mapping in
  `corpus.py`, to improve a score. If the exam parser has to change, rebuild with `make eval-set`
  and say why in the commit.
- **No secrets, no stray files.** Never commit API keys, `.env` (only `.env.example`, with empty
  values), `*:Zone.Identifier` files (`.gitignore` does not cover them), the PDFs or `data/`.
  Stage files by name and read `git status` before committing.
- **English only.** Code, comments, docs and commit messages are in English. The corpus and the
  exam questions are Spanish data: leave them as they are.

## Definition of done

From a clean checkout, with no API key in the environment, both of these pass:

```bash
make install
make check    # ruff, then pytest: the same steps as CI
```

Also: `git status` shows only the files you meant to change, `uv.lock` is committed with any
dependency change, and every README figure you touched matches the file it comes from.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/): `type: summary`, with a type such as
`feat`, `fix`, `docs`, `test`, `refactor` or `chore`. Imperative mood; header at most 60 characters.
