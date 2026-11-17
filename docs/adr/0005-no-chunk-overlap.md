# No chunk overlap

Status: Accepted (2026-11-17)

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

No overlap. The published evidence is mixed, so the retrieval eval
compared 0 with 2 sentences on this corpus.

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

Overlap was counted in whole sentences, not tokens, so the verifier
could still quote whole sentences (ADR 0002).

On 2026-11-17, eval.py scored both on the 25 dev questions that have
pages to find (the overlap2 variant in experiments.py), against the
default fused search with at most 2 passages per page. Overlap only
changes sections longer than MAX_WORDS, so it added 60 chunks: 3,707
against 3,647.

| | No overlap | 2 sentences |
|---|---|---|
| Fused recall@8 | 0.564 | 0.572 |
| Fused recall@20 | 0.683 | 0.678 |
| Fused MRR | 0.622 | 0.618 |
| Fused nDCG@10 | 0.589 | 0.582 |
| Questions fixed / broken | | 0 / 0 |
| Questions better / worse by MRR | | 1 / 2 |

The whole recall@8 gain is one question (q39), where one more answer
page reached the top 8 (0.60 to 0.80). Every other measure fell
slightly, and q35 and q36 ranked their first answer page lower.
Against the earlier default without the per-page cap the picture was
the same: recall@8 0.533 to 0.537, nothing fixed or broken. A gain of
one page on one question does not pay for indexing sentences twice
and repeating them across the eight passages.

### Consequences

Each sentence is in exactly one chunk, and the eight passages hold no
repeats. A fact split across a boundary may be found in neither half.
OVERLAP_SENTENCES
stays 0 in chunk.py, and the overlap2 variant stays in experiments.py
to test again if the chunk sizes or the embedding model change.

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
