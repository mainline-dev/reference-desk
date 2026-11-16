# Hybrid search: BM25 and embeddings, fused by reciprocal rank

Status: Accepted

## Context and Problem Statement

Embeddings find chunks that say the same thing as the question in
other words, but miss exact terms: for "BuildKit", the two chunks
that name it ranked 39th and 135th by embeddings. BM25 finds exact
terms but not paraphrases. BEIR found that "BM25 remains a strong
baseline for zero-shot text retrieval", and that dense retrievers
perform strongly on some datasets "while on many other datasets
perform significantly worse than BM25" (Thakur et al. 2021).

## Considered Options

- Embeddings alone
- BM25 alone
- Both, merged by a weighted sum of normalised scores (a convex
  combination)
- Both, merged by reciprocal rank fusion

## Decision Outcome

Both, merged by reciprocal rank fusion with k = 60, keeping the top
20. BM25 (Robertson and Zaragoza 2009) runs over the same embed_text
the vectors were made from. Each chunk scores the sum of
1 / (k + rank) over the lists it is in. Cormack et al. (2009) found
that RRF "consistently yields better results than any individual
system", and that k = 60 "was near-optimal, but that the choice was
not critical". RRF needs no training data and no score normalisation,
and this corpus has no labelled questions yet.

### Consequences

Exact terms and paraphrases both reach the top 20: for "BuildKit",
fusion puts the two chunks that name it first and second. Bruch et
al. (2023) found RRF "sensitive to its parameters", and that a convex
combination "outperforms RRF in in-domain and out-of-domain
settings", needing only a small set of examples to tune. Once the
eval has labelled questions, a convex combination is the alternative
to test.

## References

Bruch, S., Gai, S. and Ingber, A. (2023). An Analysis of Fusion
Functions for Hybrid Retrieval. ACM Transactions on Information
Systems, 42(1), 1-35. https://doi.org/10.1145/3596512 (also
arXiv:2210.11934)

Cormack, G. V., Clarke, C. L. A. and Büttcher, S. (2009). Reciprocal
Rank Fusion outperforms Condorcet and individual Rank Learning
Methods. In Proceedings of the 32nd International ACM SIGIR
Conference on Research and Development in Information Retrieval, pp.
758-759. ACM. https://doi.org/10.1145/1571941.1572114

Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance
Framework: BM25 and Beyond. Foundations and Trends in Information
Retrieval, 3(4), 333-389. https://doi.org/10.1561/1500000019

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. and Gurevych, I.
(2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
Information Retrieval Models. In Proceedings of the Neural
Information Processing Systems Track on Datasets and Benchmarks 1
(NeurIPS Datasets and Benchmarks 2021).
https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html
