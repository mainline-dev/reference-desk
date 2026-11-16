# Embed with bge-small-en-v1.5, within 480 tokens

Status: Accepted

## Context and Problem Statement

Chunks need an embedding model that runs on a laptop, needs no API
and can be pinned to an exact version. Whatever a model reads past
its input limit is silently ignored, so a chunk longer than the limit
is only partly searchable. MTEB found that "no particular text
embedding method dominates across all tasks" (Muennighoff et al.
2023), so any choice is a starting point for the eval.

## Considered Options

- bge-small-en-v1.5, 384 dimensions, run locally with
  sentence-transformers
- A larger local model, such as bge-base or bge-large
- A hosted embedding API

## Decision Outcome

BAAI/bge-small-en-v1.5, at the revision recorded in
corpus/manifest.json, with normalised vectors so that a dot product
is the cosine.

Queries get the model's instruction and passages do not. The BGE
paper attaches a task instruction "to the query side" in training
(Xiao et al. 2024). The model card gives the query instruction
"Represent this sentence for searching relevant passages: " and says
"no instruction needs to be added to passages".

Every embed_text fits in 480 tokens by the model's own tokeniser,
special tokens included. The model reads at most 512, its
max_seq_length and the sequence length BERT was pre-trained on
(Devlin et al. 2019); 480 leaves headroom. chunk.py cuts any longer
chunk at sentence ends or line breaks, and embed.py stops if an
embed_text is over the limit.

### Consequences

The whole corpus embeds in about 25 seconds on a Mac GPU, and search
loads the model in under a second with no network. Changing the model
means embedding everything again; search.py refuses to run if the
manifest's model or revision differs from the one it loads. The eval
may show a larger model is worth its cost.

## References

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). BERT:
Pre-training of Deep Bidirectional Transformers for Language
Understanding. In Proceedings of the 2019 Conference of the North
American Chapter of the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long and Short Papers), pp.
4171-4186. Association for Computational Linguistics.
https://doi.org/10.18653/v1/N19-1423

Muennighoff, N., Tazi, N., Magne, L. and Reimers, N. (2023). MTEB:
Massive Text Embedding Benchmark. In Proceedings of the 17th
Conference of the European Chapter of the Association for
Computational Linguistics, pp. 2014-2037. Association for
Computational Linguistics. https://doi.org/10.18653/v1/2023.eacl-main.148

Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D. and Nie, J.-Y.
(2024). C-Pack: Packed Resources For General Chinese Embeddings. In
Proceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval, pp. 641-649. ACM.
https://doi.org/10.1145/3626772.3657878 (also arXiv:2309.07597)

Documentation: BAAI, bge-small-en-v1.5 model card, revision
5c38ec7c405ec4b44b94cc5a9bb96e735b38267a.
https://huggingface.co/BAAI/bge-small-en-v1.5
