# No chunk overlap until the eval decides

Status: Proposed

## Context and Problem Statement

Overlapping chunks repeat the end of one chunk at the start of the
next, so a fact near a boundary is whole in at least one of them.
They also index the same sentences twice, and when neighbouring
chunks are both retrieved they fill an answer's eight passages with
repeated text. chunk.py has OVERLAP_SENTENCES set to 0; the choice is
between 0 and 2 sentences.

## Considered Options

- No overlap
- Two sentences of overlap
- Overlap in tokens, as a fraction of the chunk size

## Decision Outcome

Proposed: no overlap, until the retrieval eval compares 0 with 2
sentences on this corpus. The evidence is mixed.

- With text-embedding-3-large and 400-token chunks, removing a
  200-token overlap raised recall (88.1 to 89.5 with
  RecursiveCharacterTextSplitter) and IoU (3.3 to 3.6). Smith and
  Troynikov (2024) note that "reducing chunk overlap improves IoU
  scores, as this metric penalizes redundant information".
- With the smaller all-MiniLM-L6-v2, TokenTextSplitter recall fell
  from 0.824 to 0.771 without overlap, "suggesting that for smaller
  context, overlapping chunks are necessary for high recall" (Smith
  and Troynikov 2024). bge-small-en-v1.5 is also a small model, which
  is why the eval tries 2.
- Karpukhin et al. (2020) split documents into disjoint passages and
  report of overlapping ones: "which we do not find advantageous
  compared to the non-overlapping version".

Overlap is counted in whole sentences, not tokens, so the verifier
can still quote whole sentences (ADR 0002).

### Consequences

With no overlap each sentence is in exactly one chunk, and the eight
passages hold no repeats. A fact split across a boundary may be found
in neither half. The eval decides between 0 and 2, and this ADR is
then accepted or replaced.

## References

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S.,
Chen, D. and Yih, W. (2020). Dense Passage Retrieval for Open-Domain
Question Answering. In Proceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing (EMNLP), pp.
6769-6781. Association for Computational Linguistics.
https://doi.org/10.18653/v1/2020.emnlp-main.550

Smith, B. and Troynikov, A. (2024). Evaluating Chunking Strategies
for Retrieval. Chroma Technical Report.
https://trychroma.com/research/evaluating-chunking
