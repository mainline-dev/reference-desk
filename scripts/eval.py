#!/usr/bin/env -S uv run --quiet python
"""Score retrieval against eval.jsonl: BM25, embeddings and the fused search.

Each question lists the pages (units) that answer it (grade 2) or relate to it
(grade 1). Search returns chunks; a page's rank is the rank of its best chunk.
For each question and each way of searching:

  recall@8    share of grade-2 pages among the top 8 chunks, the passages
              answer.py puts in the prompt
  recall@20   the same over the top 20 chunks that search.py keeps
  MRR         1 / the rank of the first grade-2 page
  nDCG@10     graded: grade 2 counts 3, grade 1 counts 1, over the top 10 pages

Questions with no pages to find (unanswerable, and anything still pending, such
as a steelman or injection page not yet written) are not scored here; they test
answers, not retrieval.

The dev and holdout splits are reported separately, overall and by question
type. Dev questions are listed one by one, with those where no grade-2 page
reaches the fused top 8. Holdout questions are listed only with --show-holdout:
tune on dev, and look at the holdout last. Every run is saved to
corpus/evals/<timestamp>.json with the git commit, so runs can be compared.

Usage (from the repo root): ./scripts/eval.py [--show-holdout]
"""
import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import search

EVAL = Path("eval.jsonl")
RUNS = Path("corpus/evals")
GAIN = {2: 3, 1: 1}  # 2^grade - 1
MODES = ["bm25", "embeddings", "fused"]
METRICS = ["recall@8", "recall@20", "MRR", "nDCG@10"]
PROMPT_PASSAGES = 8


def page_order(chunk_order, chunks):
    """Pages in the order their best chunk ranks: {unit id: rank from 1}."""
    order = {}
    for i in chunk_order:
        order.setdefault(chunks[i]["unit_id"], len(order) + 1)
    return order


def scores(chunk_order, chunks, grades):
    pages = page_order(chunk_order, chunks)
    answers = {u for u, g in grades.items() if g == 2}
    found = lambda k: {chunks[i]["unit_id"] for i in chunk_order[:k]}
    ranked = sorted((pages[u] for u in answers if u in pages))
    dcg = sum(GAIN[grades[u]] / math.log2(r + 1) for u, r in pages.items() if r <= 10 and u in grades)
    ideal = sum(g / math.log2(r + 1) for r, g in enumerate(sorted((GAIN[g] for g in grades.values()), reverse=True)[:10], 1))
    return {
        "recall@8": len(answers & found(8)) / len(answers),
        "recall@20": len(answers & found(20)) / len(answers),
        "MRR": 1 / ranked[0] if ranked else 0.0,
        "nDCG@10": dcg / ideal,
    }


def best_chunk_rank(chunk_order, chunks, grades):
    """The rank of the first chunk from a grade-2 page, or None if none is ranked."""
    return next((rank for rank, i in enumerate(chunk_order, 1) if grades.get(chunks[i]["unit_id"]) == 2), None)


def orders_for(question, chunks, vectors, manifest, model):
    """Chunk indices in rank order for each way of searching."""
    _, _, by_embeddings, by_bm25, fused = search.rankings(question, chunks, vectors, manifest, model)
    return {
        "bm25": sorted(by_bm25, key=by_bm25.get),
        "embeddings": sorted(by_embeddings, key=by_embeddings.get),
        "fused": sorted(fused, key=lambda i: (-fused[i], i)),
    }


def average(rows, mode):
    return {m: sum(r["scores"][mode][m] for r in rows) / len(rows) for m in METRICS}


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def table(title, groups):
    """groups: [(label, rows)]. One line per label and way of searching."""
    print(f"\n{title}")
    print(f"  {'':22}{'mode':>11}" + "".join(f"{m:>11}" for m in METRICS))
    for label, rows in groups:
        for n, mode in enumerate(MODES):
            name = f"{label} ({len(rows)})" if n == 0 else ""
            print(f"  {name:22}{mode:>11}" + "".join(f"{v:>11.3f}" for v in average(rows, mode).values()))


def main():
    parser = argparse.ArgumentParser(description="Score retrieval against eval.jsonl.")
    parser.add_argument("--show-holdout", action="store_true", help="also list holdout questions one by one")
    args = parser.parse_args()

    raw = EVAL.read_bytes()
    questions = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    chunks, vectors, manifest, model = search.load()

    rows = []
    for q in questions:
        grades = {e["unit_id"]: e["grade"] for e in q["expected"] if "unit_id" in e}
        if 2 not in grades.values():
            continue  # nothing to retrieve yet: scored by the answer eval
        orders = orders_for(q["question"], chunks, vectors, manifest, model)
        rows.append({
            "id": q["id"], "type": q["type"], "split": q["split"], "question": q["question"],
            "scores": {mode: scores(orders[mode], chunks, grades) for mode in MODES},
            "best_rank": {mode: best_chunk_rank(orders[mode], chunks, grades) for mode in MODES},
        })

    summary = {}
    for split in ("dev", "holdout"):
        in_split = [r for r in rows if r["split"] == split]
        types = sorted({r["type"] for r in in_split})
        by_type = [(t, [r for r in in_split if r["type"] == t]) for t in types]
        table(f"{split}: all scored questions", [("all", in_split)])
        table(f"{split}: by question type", by_type)
        summary[split] = {"all": {m: average(in_split, m) for m in MODES},
                          "by_type": {t: {m: average(rs, m) for m in MODES} for t, rs in by_type}}
        if split == "dev" or args.show_holdout:
            print(f"\n{split}: question by question (MRR, and fused recall@8)")
            print(f"  {'id':5}{'type':21}" + "".join(f"{m:>11}" for m in MODES) + f"{'fused r@8':>11}")
            for r in in_split:
                print(f"  {r['id']:5}{r['type']:21}" + "".join(f"{r['scores'][m]['MRR']:>11.2f}" for m in MODES)
                      + f"{r['scores']['fused']['recall@8']:>11.2f}")

    misses = [r for r in rows if r["split"] == "dev" and r["scores"]["fused"]["recall@8"] == 0]
    print(f"\ndev: questions where no grade-2 page reaches the fused top {PROMPT_PASSAGES} "
          f"(rank of the first chunk from a grade-2 page; - if never ranked)")
    for r in misses:
        ranks = "  ".join(f"{m} {r['best_rank'][m] or '-'}" for m in MODES)
        print(f"  {r['id']}  {ranks}  | {r['question']}")
    if not misses:
        print("  none")

    stamp = datetime.now(timezone.utc)
    run = {
        "created": stamp.isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "eval_sha256": hashlib.sha256(raw).hexdigest(),
        "chunks_sha256": manifest["chunks_sha256"],
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "summary": summary,
        "questions": rows,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{stamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved {path} (git {run['git_commit'][:12]}{', with uncommitted changes' if run['git_dirty'] else ''})")


if __name__ == "__main__":
    main()
