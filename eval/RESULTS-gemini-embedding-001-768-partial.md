# Retrieval results

Embedder `gemini-embedding-001-768`, store `local`, 98 questions (86 in corpus, 12 out of corpus). Generated 20261008T113431Z.

| configuration | chunks | R@1 | random R@1 | R@5 | random R@5 | lift@5 | R@20 | MRR | out-of-corpus AUC | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `gemini-embedding-001-768_800-100_dense` | 1138 | 0.965 | 0.282 | 1.000 | 0.748 | +1.000 | 1.000 | 0.983 | 0.986 | 58.7 |

## recall@5 by area

| configuration | demografia | economia | estadistica | informatica | muestreo |
|---|---|---|---|---|---|
| `gemini-embedding-001-768_800-100_dense` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
