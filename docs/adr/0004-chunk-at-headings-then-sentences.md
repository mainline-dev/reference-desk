# Chunk at headings, then at sentence ends

Status: Accepted

## Context and Problem Statement

Retrieval returns chunks, not pages, and the unit matters: Chen et
al. (2024) "discover that the retrieval unit choice significantly
impacts the performance of both retrieval and downstream tasks".
Pages in the corpus run from under 100 words to over 6,000, with
headings where the topic changes. Answers quote whole sentences, so
a chunk that ends mid-sentence cannot be quoted from.

## Considered Options

- Fixed-size chunks of tokens, ignoring the page's structure
- Semantic chunking, splitting where neighbouring sentences'
  embeddings diverge
- Structure first: split at every heading, then pack whole sentences
  up to a size limit

## Decision Outcome

Structure first. chunk.py splits each page at every heading outside
code, then packs whole sentences into chunks of up to 300 words. A
code block or a table stays one block. A chunk under 40 words joins
its neighbour within the same section. A chunk whose embed_text
would be over the model's limit is cut again at sentence ends, then
line breaks (ADR 0007).

- Jimeno Yepes et al. (2024) found that chunking by a document's
  structural elements, rather than by paragraph alone, improved RAG
  results on financial reports.
- Wang et al. (2024) note that "Token-level Chunking is
  straightforward but may split sentences, affecting retrieval
  quality", and chose sentence-level chunking.
- Qu et al. (2025) found that "the computational costs associated
  with semantic chunking are not justified by consistent performance
  gains" over fixed-size chunking.

On size, Smith and Troynikov (2024) tested RecursiveCharacterTextSplitter
and TokenTextSplitter with text-embedding-3-large: recall was highest
at 400 tokens with no overlap, precision and IoU highest at 200, and
800-token chunks with 400 of overlap scored lowest on all four.
Karpukhin et al. (2020) split documents into "disjoint text blocks of
100 words". A target of 300 words sits between these; the eval
decides whether it moves.

### Consequences

Chunks follow the authors' own sections, so a heading path describes
each one (ADR 0006). Sections vary, so chunk sizes do too, from 8 to
about 335 words. No chunk cuts a sentence, which the verifier relies
on (ADR 0002).

## References

Chen, T., Wang, H., Chen, S., Yu, W., Ma, K., Zhao, X., Zhang, H. and
Yu, D. (2024). Dense X Retrieval: What Retrieval Granularity Should
We Use? In Proceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing, pp. 15159-15177. Association for
Computational Linguistics. https://doi.org/10.18653/v1/2024.emnlp-main.845

Jimeno Yepes, A., You, Y., Milczek, J., Laverde, S. and Li, R.
(2024). Financial Report Chunking for Effective Retrieval Augmented
Generation. arXiv:2402.05131 (preprint).
https://doi.org/10.48550/arXiv.2402.05131

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S.,
Chen, D. and Yih, W. (2020). Dense Passage Retrieval for Open-Domain
Question Answering. In Proceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing (EMNLP), pp.
6769-6781. Association for Computational Linguistics.
https://doi.org/10.18653/v1/2020.emnlp-main.550

Qu, R., Tu, R. and Bao, F. S. (2025). Is Semantic Chunking Worth the
Computational Cost? In Findings of the Association for Computational
Linguistics: NAACL 2025, pp. 2155-2177. Association for Computational
Linguistics. https://doi.org/10.18653/v1/2025.findings-naacl.114

Smith, B. and Troynikov, A. (2024). Evaluating Chunking Strategies
for Retrieval. Chroma Technical Report.
https://trychroma.com/research/evaluating-chunking

Wang, X., Wang, Z., Gao, X., Zhang, F., Wu, Y., Xu, Z., Shi, T.,
Wang, Z., Li, S., Qian, Q., Yin, R., Lv, C., Zheng, X. and Huang, X.
(2024). Searching for Best Practices in Retrieval-Augmented
Generation. In Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing, pp. 17716-17736. Association
for Computational Linguistics. https://aclanthology.org/2024.emnlp-main.981
