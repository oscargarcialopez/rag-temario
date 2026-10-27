PY := .venv/bin/python
RAG := .venv/bin/rag

.PHONY: install stats eval-set eval eval-offline index ask test lint check clean

install:            ## create the venv and install everything
	uv sync

stats:              ## corpus and eval-set summary
	$(RAG) stats

eval-set:           ## rebuild the labelled eval set from the official exams
	$(RAG) build-eval-set

index:              ## build the default index (800/100, dense)
	$(RAG) index

eval:               ## full sweep with the real embedding model
	$(RAG) eval

eval-offline:       ## full sweep with the hash embedder: no API key, no cost
	$(RAG) eval --offline

ask:                ## make ask Q="¿qué es un estimador insesgado?"
	$(RAG) ask "$(Q)"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

check: lint test    ## what CI would run

clean:              ## drop derived artefacts, keep the eval set
	rm -rf data/index data/runs data/cache
