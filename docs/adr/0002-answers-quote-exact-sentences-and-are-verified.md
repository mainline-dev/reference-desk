# Answers quote exact sentences, and every quote is verified

Status: Accepted

## Context and Problem Statement

A model answering from retrieved passages can say things the passages
do not, and cite passages that do not support it. In generative search
engines "a mere 51.5% of generated sentences are fully supported by
citations and only 74.5% of citations support their associated
sentence" (Liu et al. 2023). On ELI5, "even the best models lack
complete citation support 50% of the time" (Gao et al. 2023). A reader
of the reference desk needs to know which words come from a source.

## Considered Options

- Free answers citing passages, unchecked
- Free answers citing passages, checked by a second model
- Answers built from exact quotes, each with its passage number,
  checked by a program against the passage text

## Decision Outcome

Exact quotes, checked by a program. The model sees the top 8 passages
as [c1] to [c8], with links reduced to their words, and must quote
exact sentences with their passage numbers. verify.py finds each quote
in its passage's text, ignoring white space, link markup and the case
of the quote's first letter, and nothing else. It records the exact
slice of the chunk text each checked quote matched.

Every part of an answer is checked (an exact quote), unchecked (the
model's own words beside a checked quote, labelled as its wording) or
flagged (a sentence with no quote, or a quote not in its passage).
Quote coverage, the share of the answer's words inside checked quotes,
is saved with each answer, with the prompt version that produced it.
With the structured prompt (v3) the model returns JSON quotes, an
optional summary and an optional note of disagreement, and the answer
is rendered from what checked.

This applies the AIS framework of Rashkin et al. (2023), under which
output about the world "is to be verified against an independent,
provided source", with a check a program can run.

When the passages do not answer the question, the reply is "Not in the
sources." (or, in v3, an empty list of quotes). As with unanswerable
questions in SQuAD, a system must "determine when no answer is
supported by the paragraph and abstain from answering" (Rajpurkar et
al. 2018).

### Consequences

A checked quote is in its source, word for word, and can be traced to
the exact characters it came from. A quote changed by one word is
flagged, so strictness costs some answers a reader would accept. The
check shows that a quote is in a passage, not that it answers the
question; the model's own words around it stay unchecked and are shown
as such.

## References

Gao, T., Yen, H., Yu, J. and Chen, D. (2023). Enabling Large Language
Models to Generate Text with Citations. In Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing, pp.
6465-6488. Association for Computational Linguistics.
https://doi.org/10.18653/v1/2023.emnlp-main.398

Liu, N., Zhang, T. and Liang, P. (2023). Evaluating Verifiability in
Generative Search Engines. In Findings of the Association for
Computational Linguistics: EMNLP 2023, pp. 7001-7025. Association for
Computational Linguistics.
https://doi.org/10.18653/v1/2023.findings-emnlp.467

Rajpurkar, P., Jia, R. and Liang, P. (2018). Know What You Don't Know:
Unanswerable Questions for SQuAD. In Proceedings of the 56th Annual
Meeting of the Association for Computational Linguistics (Volume 2:
Short Papers), pp. 784-789. Association for Computational Linguistics.
https://doi.org/10.18653/v1/P18-2124

Rashkin, H., Nikolaev, V., Lamm, M., Aroyo, L., Collins, M., Das, D.,
Petrov, S., Tomar, G. S., Turc, I. and Reitter, D. (2023). Measuring
Attribution in Natural Language Generation Models. Computational
Linguistics, 49(4), 777-840. https://doi.org/10.1162/coli_a_00486
