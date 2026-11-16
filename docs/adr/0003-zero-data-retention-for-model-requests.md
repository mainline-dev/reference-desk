# Model requests go only to zero data retention providers

Status: Accepted

## Context and Problem Statement

answer.py sends each question and its retrieved passages to a hosted
model through OpenRouter. The passages are published material; the
questions are not. A provider that keeps prompts may store them or
train on them, and text a model was trained on can be recovered
from it: Carlini et al. (2021) "are able to extract hundreds of
verbatim text sequences from the model's training data", including
names, phone numbers and email addresses.

## Considered Options

- Any provider OpenRouter routes to
- A local model only
- Hosted models, restricted on every request to zero data retention
  providers

## Decision Outcome

Every request carries provider: {"zdr": true}, set by REFDESK_ZDR,
which is true by default. If no zero data retention provider will
serve the model, answer.py stops. It never sends the request again
without the restriction. The default model, DeepSeek V3 0324, is a
chat model, not a reasoning one, and SiliconFlow serves it with zero
data retention. Each answer records whether zdr was on, the provider
that served it and its reasoning tokens.

### Consequences

Only models with a zero data retention provider can be used, and a
model with one such provider is unavailable whenever that provider
is. OpenRouter's zdr flag also allows Safety Retention endpoints
unless the account's privacy setting is "ZDR only", so strict zero
retention needs that setting too. A local model behind the same
OpenAI-compatible interface remains an option.

## References

Carlini, N., Tramèr, F., Wallace, E., Jagielski, M., Herbert-Voss,
A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, Ú.,
Oprea, A. and Raffel, C. (2021). Extracting Training Data from Large
Language Models. In 30th USENIX Security Symposium (USENIX Security
21), pp. 2633-2650. USENIX Association.
https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting

Documentation: OpenRouter, Zero Data Retention.
https://openrouter.ai/docs/guides/features/zdr
