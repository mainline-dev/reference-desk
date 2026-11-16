# Reference Desk

Reference Desk answers questions about continuous delivery practice from
published sources: DORA, MinimumCD, Beyond MinimumCD and the Twelve-Factor App.
Every answer quotes its sources word for word, and every quote is checked
against the passage it came from.

## Scope

Reference Desk holds published practice material only, no customer or team data.

## Pipeline

Each script runs from the repo root, in this order:

1. `scripts/normalise.py` cleans each source in `corpus/raw/` into Markdown with one front matter schema, and writes `corpus/units.jsonl` and `SOURCES.md`.
2. `scripts/chunk.py` cuts each page into chunks at headings and sentence ends, each within 480 tokens, and writes `corpus/chunks.jsonl`.
3. `scripts/embed.py` embeds the chunks with bge-small-en-v1.5 and writes `corpus/chunks.npy` and `corpus/manifest.json`.
4. `scripts/search.py "question"` searches with BM25 and embeddings, merged by reciprocal rank fusion.
5. `scripts/answer.py "question"` answers from the top 8 passages in exact quotes, using the model set in `.env`.
6. `scripts/verify.py` checks every quote in a saved answer against its passage.
7. `scripts/eval.py` scores retrieval against the questions in `eval.jsonl`.

The decisions behind each step, with the research they rest on, are in `docs/adr/`.

## Setup

```sh
uv sync
git clone https://github.com/dora-team/dora.dev corpus/raw/dora.dev
git clone https://github.com/Minimum-CD/cd-manifesto corpus/raw/cd-manifesto
git clone https://github.com/bdfinst/cd-migration corpus/raw/cd-migration
git clone https://github.com/twelve-factor/twelve-factor corpus/raw/twelve-factor
cp .env.example .env
./scripts/normalise.py && ./scripts/chunk.py && ./scripts/embed.py
```

Add your OpenRouter key to `.env` before running `answer.py`; `.env.example`
explains each setting. The first `chunk.py` and `embed.py` runs download the
tokeniser and model from Hugging Face, and later runs use the cached copies.
`SOURCES.md` lists the commit of each source that the current corpus was built
from.

## Evaluation

```sh
./scripts/eval.py
```

This scores BM25, embeddings and the fused search on the dev questions, overall
and by question type, and saves the run to `corpus/evals/`. The holdout
questions are scored too, but listed one by one only with `--show-holdout`:
tune on dev and look at the holdout last.

The tests need neither the corpus nor the network:

```sh
uv run python -m unittest discover -s tests
```

## Licence

The code is MIT licensed: see `LICENSE.md`. The corpus is not included, and
each source keeps its own licence: see `SOURCES.md`.
