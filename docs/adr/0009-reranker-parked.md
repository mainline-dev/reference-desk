# Cross-encoder reranking: parked

Status: Parked (2026-11-17)

## Context and Problem Statement

The fused search (ADR 0008) scores the question and each passage
separately, by embeddings and by BM25. A cross-encoder reads the
question and a passage together and scores the pair, which is
slower but usually ranks better. Nogueira and Cho (2019) "describe a
simple re-implementation of BERT for query-based passage
re-ranking". BEIR found that re-ranking and "late-interaction-based
models on average achieve the best zero-shot performances, however,
at high computational costs" (Thakur et al. 2021).

Should Reference Desk rerank the fused list before the top 8 go into
the prompt?

## Considered Options

- No reranker: the fused list, at most 2 passages per page
- BAAI/bge-reranker-base, run locally, rescoring the fused top 50

## Decision Outcome

Parked: no reranker. On 2026-11-17, eval.py scored the rerank
variant in experiments.py on the 25 dev questions that have pages to
find, against the default fused search with at most 2 passages per
page. It does not meet the win rule (at least 3 questions fixed and
at most 1 broken, or none broken and fused recall@8 up).

| | Fused | Reranked |
|---|---|---|
| Fused recall@8 | 0.564 | 0.535 |
| Fused recall@20 | 0.683 | 0.736 |
| Fused MRR | 0.622 | 0.633 |
| Questions fixed / broken | | 2 (q11, q31) / 1 (q09) |
| Contested, recall@8 | 0.833 | 0.458 |
| Vocabulary mismatch, recall@8 | 0.300 | 0.524 |
| Added time per query | | median 1,567 ms |

Two things park it.

- Latency. Rescoring 50 passages adds a median of 1,567 ms to every
  query on this Mac's GPU (Apple MPS), and the model takes about
  1.1 GB on disk.
- Contested questions. These are the questions where the sources
  disagree and the answer has to show both sides. Reranking took
  contested recall@8 from 0.833 to 0.458. On q09 (IaC in a separate
  repository) it pushed both I. Codebase and Everything as Code out
  of the top 8. On q18 (config shipped with the deployment) it pushed
  out Deployment automation. Only 4 dev questions are contested, so
  this is a warning, not a measurement.

It did help where the question's words differ from the pages'
(vocabulary mismatch recall@8 0.300 to 0.524), which is the gap
index-time expansion is now meant to close.

### Consequences

Search stays as fast as two list lookups and a fusion, and contested
questions keep both sides in the top 8. The rerank variant stays in
experiments.py. Try it again if the eval gains more contested
questions, if it can run on a faster GPU, or if a reranker is found
that keeps opposing passages, and then accept or replace this ADR.

## References

Nogueira, R. and Cho, K. (2019). Passage Re-ranking with BERT.
arXiv preprint arXiv:1901.04085. https://doi.org/10.48550/arXiv.1901.04085

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. and Gurevych, I.
(2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
Information Retrieval Models. In Proceedings of the Neural
Information Processing Systems Track on Datasets and Benchmarks 1
(NeurIPS Datasets and Benchmarks 2021).
https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html
(also arXiv:2104.08663)
