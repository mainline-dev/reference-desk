#!/usr/bin/env -S uv run --quiet python
"""Search the chunks for a question by embeddings and by BM25, merged by
reciprocal rank fusion.

Usage (from the repo root):
  ./scripts/search.py "question"                    both lists, fused
  ./scripts/search.py "question" --embeddings-only  ordered by the embeddings alone
  ./scripts/search.py "question" --bm25-only        ordered by BM25 alone

Both lists are computed in every mode, so each result shows its rank in both.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

import embed  # scripts/embed.py: the model, how it is loaded, and where the vectors are

UNITS = Path("corpus/units.jsonl")
RRF_K = 60   # ADR 0008: reciprocal rank fusion constant
TOP = 20     # ADR 0008: results kept after fusion
WORD = re.compile(r"\w+")


def words(text):
    """BM25's tokens: lower-cased runs of letters, digits and underscores."""
    return WORD.findall(text.lower())


def ranking(scores, ranked):
    """{chunk index: rank from 1} for the chunks `ranked` allows, best score first."""
    order = [int(i) for i in np.argsort(-scores, kind="stable") if ranked[i]]
    return {i: rank for rank, i in enumerate(order, 1)}


def load():
    """Chunks, vectors, manifest and model, refusing anything out of step (ADR 0007)."""
    manifest = json.loads(embed.MANIFEST.read_text(encoding="utf-8"))
    raw = embed.CHUNKS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["chunks_sha256"]:
        sys.exit(f"{embed.CHUNKS} has changed since it was embedded: run ./scripts/embed.py")
    if manifest["model"] != embed.MODEL:
        sys.exit(f"the vectors were made with {manifest['model']}, but search uses {embed.MODEL}")
    model, revision, _ = embed.load_model()
    if revision != manifest["model_revision"]:
        sys.exit(f"the vectors were made with {embed.MODEL} at {manifest['model_revision'][:12]}, "
                 f"but the cached model is at {revision[:12]}: run ./scripts/embed.py")
    chunks = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    vectors = np.load(embed.VECTORS)
    if vectors.shape != (len(chunks), manifest["dimensions"]):
        sys.exit(f"{embed.VECTORS} is {vectors.shape}, but there are {len(chunks)} chunks")
    return chunks, vectors, manifest, model


def load_units():
    """{unit id: unit} for titles, URLs and licences, or {} if units.jsonl is missing."""
    if not UNITS.exists():
        return {}
    return {u["id"]: u for u in map(json.loads, UNITS.read_text(encoding="utf-8").splitlines())}


def rankings(question, chunks, vectors, manifest, model):
    """Both lists for a question and their fusion.

    Returns (cosine, bm25, by_embeddings, by_bm25, fused): the scores for every
    chunk, each list as {chunk index: rank from 1}, and {chunk index: fused score}.
    """
    # ADR 0007: the query, not the passages, gets bge's instruction. Both are unit
    # length, so the dot product is the cosine.
    query = model.encode([manifest["query_instruction"] + question], normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)[0]
    cosine = vectors @ query
    # BM25 over the same embed_text the vectors were made from, built at start-up.
    bm25 = BM25Okapi([words(c["embed_text"]) for c in chunks]).get_scores(words(question))

    by_embeddings = ranking(cosine, np.ones(len(chunks), bool))
    by_bm25 = ranking(bm25, bm25 > 0)  # a chunk without any of the words is not in the list
    fused = {i: sum(1 / (RRF_K + r[i]) for r in (by_embeddings, by_bm25) if i in r)
             for i in set(by_embeddings) | set(by_bm25)}
    return cosine, bm25, by_embeddings, by_bm25, fused


def fused_order(fused):
    """Chunk indices, best fused score first, the top TOP of them."""
    return sorted(fused, key=lambda i: (-fused[i], i))[:TOP]


def main():
    parser = argparse.ArgumentParser(description="Search the reference desk.")
    parser.add_argument("question")
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--embeddings-only", action="store_true", help="order by the embeddings alone")
    only.add_argument("--bm25-only", action="store_true", help="order by BM25 alone")
    args = parser.parse_args()

    chunks, vectors, manifest, model = load()
    units = load_units()
    cosine, bm25, by_embeddings, by_bm25, fused = rankings(args.question, chunks, vectors, manifest, model)

    if args.embeddings_only:
        mode, order, score = "embeddings only (cosine)", sorted(by_embeddings, key=by_embeddings.get), cosine
    elif args.bm25_only:
        mode, order, score = "BM25 only (BM25 score)", sorted(by_bm25, key=by_bm25.get), bm25
    else:
        mode, order, score = f"fused (reciprocal rank fusion, k={RRF_K})", fused_order(fused), fused
    print(f'"{args.question}": {mode}, top {TOP} of {len(chunks)} chunks, {len(by_bm25)} with a BM25 match\n')

    for rank, i in enumerate(order[:TOP], 1):
        c = chunks[i]
        unit = units.get(c["unit_id"], {})
        in_bm25 = by_bm25.get(i, "-")
        print(f"{rank:>2}  {score[i]:.4f}  embeddings {by_embeddings[i]}, BM25 {in_bm25}  "
              f"{unit.get('title', c['heading_path'][0])}")
        print(f"    {' > '.join(c['heading_path'])}")
        print(f"    {unit.get('url') or unit.get('path', c['unit_id'])}")
        print(f"    {' '.join(c['text'].split())[:200]}\n")


if __name__ == "__main__":
    main()
