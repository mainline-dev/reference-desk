# Put the heading path in front of what is embedded

Status: Proposed

## Context and Problem Statement

A chunk taken out of its page loses what its headings said. A chunk
under "Change Advisory Board Gates > Why this is a problem > It
increases rework" may never name the board. The embedding and BM25
see only what they are given.

## Considered Options

- Embed the chunk's text alone
- Put the page title and heading path in front of the text that is
  embedded and indexed
- Have a model write a note of each chunk's context and embed that
  with it

## Decision Outcome

Proposed: each chunk's embed_text is its heading path ("Title >
Section > Subsection"), a blank line, then its text. The chunk's
text field, which answers quote from, stays an exact slice of the
page. embed_text also reduces links to their words, leaves out
"Referenced in:" lists and tidies tables, so the tokens the model
reads are content.

Karpukhin et al. (2020) prepended each passage "with the title of the
Wikipedia article where the passage is from, along with an [SEP]
token". Wang et al. (2024) note that "Enhancing chunk blocks with
metadata like titles, keywords, and hypothetical questions can
improve retrieval".

### Consequences

Every chunk carries its context to both BM25 and the embedding, at
the cost of the path's tokens within the 480-token limit (ADR 0007).
Whether it helps on this corpus is for the eval to show, against
embedding the text alone; this ADR is then accepted or replaced.

## References

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S.,
Chen, D. and Yih, W. (2020). Dense Passage Retrieval for Open-Domain
Question Answering. In Proceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing (EMNLP), pp.
6769-6781. Association for Computational Linguistics.
https://doi.org/10.18653/v1/2020.emnlp-main.550

Wang, X., Wang, Z., Gao, X., Zhang, F., Wu, Y., Xu, Z., Shi, T.,
Wang, Z., Li, S., Qian, Q., Yin, R., Lv, C., Zheng, X. and Huang, X.
(2024). Searching for Best Practices in Retrieval-Augmented
Generation. In Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing, pp. 17716-17736. Association
for Computational Linguistics. https://aclanthology.org/2024.emnlp-main.981
