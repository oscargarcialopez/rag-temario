# Retrieval results

Embedder `gemini-embedding-2-768`, store `local`, 98 questions (86 in corpus, 12 out of corpus). Generated 20261008T113533Z.

| configuration | chunks | R@1 | random R@1 | R@5 | random R@5 | lift@5 | R@20 | MRR | out-of-corpus AUC | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `gemini-embedding-2-768_400-50_dense` | 2266 | 0.942 | 0.282 | 1.000 | 0.747 | +1.000 | 1.000 | 0.969 | 0.974 | 1.2 |
| `gemini-embedding-2-768_800-100_dense` | 1138 | 0.953 | 0.282 | 1.000 | 0.748 | +1.000 | 1.000 | 0.977 | 0.979 | 1.0 |
| `gemini-embedding-2-768_1200-150_dense` | 764 | 0.965 | 0.281 | 1.000 | 0.748 | +1.000 | 1.000 | 0.981 | 0.958 | 0.7 |
| `gemini-embedding-2-768_page_dense` | 1279 | 0.977 | 0.299 | 1.000 | 0.742 | +1.000 | 1.000 | 0.988 | 0.988 | 0.9 |
| `gemini-embedding-2-768_800-100_bm25` | 1138 | 0.930 | 0.282 | 0.988 | 0.748 | +0.954 | 1.000 | 0.956 | 0.696 | 0.8 |
| `gemini-embedding-2-768_800-100_hybrid` | 1138 | 0.988 | 0.282 | 1.000 | 0.748 | +1.000 | 1.000 | 0.994 | 0.785 | 1.4 |

## recall@5 by area

| configuration | demografia | economia | estadistica | informatica | muestreo |
|---|---|---|---|---|---|
| `gemini-embedding-2-768_400-50_dense` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `gemini-embedding-2-768_800-100_dense` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `gemini-embedding-2-768_1200-150_dense` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `gemini-embedding-2-768_page_dense` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `gemini-embedding-2-768_800-100_bm25` | 1.00 | 1.00 | 1.00 | 1.00 | 0.86 |
| `gemini-embedding-2-768_800-100_hybrid` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
