#!/usr/bin/env -S uv run --quiet python
"""Embed every chunk in corpus/chunks.jsonl into corpus/chunks.npy, and describe
the vectors in corpus/manifest.json.

Row i of chunks.npy is the embedding of line i of chunks.jsonl. The manifest
names the model and its exact revision, so a query is embedded the same way.

Usage (from the repo root): ./scripts/embed.py
"""
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import try_to_load_from_cache
from sentence_transformers import SentenceTransformer

import chunk  # scripts/chunk.py, for the limit the chunks were cut to and its splitting rules

CHUNKS = Path("corpus/chunks.jsonl")
VECTORS = Path("corpus/chunks.npy")
MANIFEST = Path("corpus/manifest.json")

MODEL = "BAAI/bge-small-en-v1.5"  # ADR 0007: the embedding model
DIMENSIONS = 384                  # ADR 0007: bge-small-en-v1.5's embedding size
MAX_TOKENS = chunk.MAX_TOKENS     # ADR 0007: the limit every embed_text was cut to fit
# ADR 0007: bge v1.5 puts this in front of a query, never in front of a passage.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 64                   # our own rule
SANITY_QUERIES = 10               # our own rule: chunks to find again from their first sentence
SANITY_SEED = 0                   # our own rule: the same chunks on every run


def load_model():
    """The model from the Hugging Face cache, and the Hub commit it came from.

    Downloads it once if it is not cached yet. Uses the Mac GPU if there is one
    (our own rule).
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model = SentenceTransformer(MODEL, device=device, local_files_only=True)
    except OSError:  # not cached yet: fetch the files it needs, once
        model = SentenceTransformer(MODEL, device=device)
    # The cache keeps each file under snapshots/<the Hub commit it came from>/.
    revision = Path(try_to_load_from_cache(MODEL, "config.json")).parent.name
    return model, revision, device


def first_sentence(text):
    """The first sentence of a chunk's first paragraph of prose, links as their words."""
    for start, end in chunk.blocks(text, 0, len(text)):
        block = text[start:end]
        if chunk.FENCE.match(block) or block.lstrip().startswith(("|", "<")):
            continue
        s, e = chunk.sentences(text, start, end)[0]
        sentence = chunk.LINK.sub(r"\1", text[s:e]).replace("**", "")
        return " ".join(sentence.lstrip("-*+0123456789. ").split())
    return ""


def main():
    started = time.perf_counter()
    if chunk.TOKENISER != MODEL:
        sys.exit(f"chunk.py cut the chunks with {chunk.TOKENISER}'s tokeniser, not {MODEL}'s")
    raw = CHUNKS.read_bytes()
    chunks = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    texts = [c["embed_text"] for c in chunks]

    model, revision, device = load_model()
    loaded = time.perf_counter()

    # ADR 0007: nothing may be cut off. Every embed_text must fit in what the model reads.
    longest = max(len(ids) for ids in model.tokenizer(texts)["input_ids"])
    if longest > min(MAX_TOKENS, model.max_seq_length):
        sys.exit(f"an embed_text has {longest} tokens; the chunks were cut to {MAX_TOKENS}")

    # ADR 0007: passages are embedded as they are, with no instruction in front.
    # Unit length (normalised), so a dot product is the cosine similarity (our own rule).
    embed_start = time.perf_counter()
    vectors = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype(np.float32, copy=False)
    embedded = time.perf_counter()

    norms = np.linalg.norm(vectors, axis=1)
    worst = float(np.abs(norms - 1).max())
    if vectors.shape != (len(chunks), DIMENSIONS):
        sys.exit(f"vectors are {vectors.shape}, expected ({len(chunks)}, {DIMENSIONS})")
    if worst > 1e-3:
        sys.exit(f"a vector's length is off 1 by {worst:.2e}")

    np.save(VECTORS, vectors)
    MANIFEST.write_text(json.dumps({
        "model": MODEL,
        "model_revision": revision,
        "dimensions": DIMENSIONS,
        "normalised": True,
        "max_tokens": MAX_TOKENS,
        "query_instruction": QUERY_INSTRUCTION,
        "chunks_sha256": hashlib.sha256(raw).hexdigest(),
        "chunk_count": len(chunks),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{MODEL} at {revision[:12]} on {device}")
    print(f"shape {vectors.shape} matches {len(chunks)} chunks; every length is 1 within {worst:.1e}")
    print(f"longest embed_text {longest} tokens; the model reads {model.max_seq_length}, so nothing is cut off")
    print(f"wrote {VECTORS} and {MANIFEST}")

    # Sanity search: a chunk's own first sentence, asked as a query, should find it.
    rng, picks = random.Random(SANITY_SEED), []
    for i in rng.sample(range(len(chunks)), len(chunks)):
        sentence = first_sentence(chunks[i]["text"])
        if len(sentence.split()) >= 5:  # a label is not a query
            picks.append((i, sentence))
        if len(picks) == SANITY_QUERIES:
            break
    queries = model.encode([QUERY_INSTRUCTION + s for _, s in picks], batch_size=BATCH_SIZE,
                           normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    ranks = []
    print(f"\nsanity search ({SANITY_QUERIES} chunks, seed {SANITY_SEED}): rank of each chunk for its own first sentence")
    for (i, sentence), query in zip(picks, queries):
        scores = vectors @ query
        rank = int((scores > scores[i]).sum()) + 1
        ranks.append(rank)
        print(f"  {rank:>4}  {chunks[i]['id']}\n        {sentence[:100]}")
    searched = time.perf_counter()

    print(f"\nranks 1 to 5: {sum(r <= 5 for r in ranks)} of {len(ranks)}")
    print(f"timings: load {loaded - started:.1f}s, embed {embedded - embed_start:.1f}s "
          f"({len(chunks) / (embedded - embed_start):.0f} chunks/s), sanity {searched - embedded:.1f}s, "
          f"total {searched - started:.1f}s")


if __name__ == "__main__":
    main()
