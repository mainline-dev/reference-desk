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
type. Dev questions are listed one by one. Then comes a failures report: each
dev question with no grade-2 page in the fused top 8, where its grade-2 pages
rank in each list, and which pages came back instead. Holdout questions are
listed only with --show-holdout: tune on dev, and look at the holdout last.
Every run is saved to corpus/evals/<timestamp>.json with the git commit.

--variant scores a retrieval variant from experiments.py, on the dev split only.
--compare compares a run with a saved one question by question. A question is
fixed when no grade-2 page was in the fused top 8 before and one is now, and
broken the other way round; better or worse by the rank of its first grade-2
page. A change wins if it fixes at least 3 questions and breaks at most 1.

Usage (from the repo root):
  ./scripts/eval.py [--show-holdout]
  ./scripts/eval.py --variant NAME [--compare corpus/evals/<run>.json]
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
WIN_FIXED, WIN_BROKEN = 3, 1  # a change wins if it fixes at least 3 questions and breaks at most 1


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


def compare(rows, base_rows):
    """Question-by-question change in the fused search from a saved run's rows to these."""
    base = {r["id"]: r for r in base_rows}
    result = {"fixed": [], "broken": [], "better": [], "worse": [], "same": 0}
    for r in rows:
        if r["id"] not in base:
            continue
        before, now = base[r["id"]]["scores"]["fused"], r["scores"]["fused"]
        if before["recall@8"] == 0 < now["recall@8"]:
            result["fixed"].append(r["id"])
        elif now["recall@8"] == 0 < before["recall@8"]:
            result["broken"].append(r["id"])
        change = (r["id"], before["MRR"], now["MRR"])
        if now["MRR"] > before["MRR"] + 1e-9:
            result["better"].append(change)
        elif now["MRR"] < before["MRR"] - 1e-9:
            result["worse"].append(change)
        else:
            result["same"] += 1
    result["wins"] = len(result["fixed"]) >= WIN_FIXED and len(result["broken"]) <= WIN_BROKEN
    return result


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
    parser.add_argument("--variant", help="score a retrieval variant from experiments.py, on dev only")
    parser.add_argument("--compare", metavar="RUN", help="compare with a saved run, question by question")
    args = parser.parse_args()

    raw = EVAL.read_bytes()
    questions = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    chunks, vectors, manifest, model = search.load()
    titles = {u: unit.get("title", u) for u, unit in search.load_units().items()}
    variant, splits = None, ("dev", "holdout")
    if args.variant:
        import experiments  # loaded only when needed: it brings the reranker and the model client
        if args.variant not in experiments.NAMES:
            parser.error(f"--variant must be one of {', '.join(experiments.NAMES)}")
        variant = experiments.Variant(args.variant, chunks, vectors, manifest, model)
        splits = ("dev",)  # a variant never touches the holdout

    rows = []
    for q in questions:
        grades = {e["unit_id"]: e["grade"] for e in q["expected"] if "unit_id" in e}
        if q["split"] not in splits or 2 not in grades.values():
            continue  # another split, or nothing to retrieve yet (scored by the answer eval)
        if variant:
            orders, pool = variant.orders(q["question"]), variant.chunks
        else:
            orders, pool = orders_for(q["question"], chunks, vectors, manifest, model), chunks
        answers = [u for u, g in grades.items() if g == 2]
        pages = {mode: page_order(orders[mode], pool) for mode in MODES}
        rows.append({
            "id": q["id"], "type": q["type"], "split": q["split"], "question": q["question"],
            "scores": {mode: scores(orders[mode], pool, grades) for mode in MODES},
            "best_rank": {mode: best_chunk_rank(orders[mode], pool, grades) for mode in MODES},
            "answer_page_ranks": {u: {mode: pages[mode].get(u) for mode in MODES} for u in answers},
            "fused_top8_pages": [pool[i]["unit_id"] for i in orders["fused"][:PROMPT_PASSAGES]],
        })

    summary = {}
    for split in splits:
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
    print(f"\ndev failures: no grade-2 page in the fused top {PROMPT_PASSAGES} ({len(misses)})")
    for r in misses:
        print(f"\n  {r['id']} ({r['type']}): {r['question']}")
        print("    grade-2 pages, page rank in bm25 / embeddings / fused:")
        for unit, ranked in r["answer_page_ranks"].items():
            where = " / ".join(str(ranked[m] or "-") for m in MODES)
            print(f"      {where:>16}  {titles.get(unit, unit)}")
        print("    what the fused top 8 held instead:")
        counts = {}
        for unit in r["fused_top8_pages"]:
            counts[unit] = counts.get(unit, 0) + 1
        for unit, n in counts.items():
            print(f"      {n} x {titles.get(unit, unit)}")

    stamp = datetime.now(timezone.utc)
    run = {
        "created": stamp.isoformat(timespec="seconds"),
        "variant": args.variant or "default",
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "eval_sha256": hashlib.sha256(raw).hexdigest(),
        "chunks_sha256": manifest["chunks_sha256"],
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "summary": summary,
        "questions": rows,
    }
    if variant and variant.timings:
        each = variant.timings
        median = sorted(each)[len(each) // 2]
        run["rerank_seconds"] = {"each": each, "median": median, "mean": sum(each) / len(each)}
        run["reranker_revision"] = variant.reranker_revision
        print(f"\nreranking the fused top {experiments.RERANK_DEPTH} on {model.device}: median {median * 1000:.0f} ms "
              f"per query, mean {run['rerank_seconds']['mean'] * 1000:.0f} ms, first query {each[0] * 1000:.0f} ms "
              f"({experiments.RERANKER} at {variant.reranker_revision[:12]})")
    if variant and getattr(variant, "generated", None):
        run["generated"] = variant.generated
        run["llm_prompt"] = experiments.REWRITE if args.variant == "rewrite" else experiments.HYDE

    if args.compare:
        base = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        result = compare([r for r in rows if r["split"] == "dev"],
                         [r for r in base["questions"] if r["split"] == "dev"])
        run["compared_with"] = {"run": args.compare, **result}
        print(f"\ncompared with {args.compare} ({base.get('variant', 'default')}), dev, fused search:")
        print(f"  fixed  ({len(result['fixed'])}): {', '.join(result['fixed']) or 'none'}")
        print(f"  broken ({len(result['broken'])}): {', '.join(result['broken']) or 'none'}")
        for name in ("better", "worse"):
            changes = ", ".join(f"{i} {a:.2f}->{b:.2f}" for i, a, b in result[name])
            print(f"  {name:6} ({len(result[name])}, by MRR): {changes or 'none'}")
        print(f"  unchanged: {result['same']}")
        base_types = base["summary"]["dev"]["by_type"]
        print("  by type, fused recall@8 and MRR, before -> after:")
        for t, rs in summary["dev"]["by_type"].items():
            if t in base_types:
                b, a = base_types[t]["fused"], rs["fused"]
                print(f"    {t:22} recall@8 {b['recall@8']:.3f} -> {a['recall@8']:.3f}   MRR {b['MRR']:.3f} -> {a['MRR']:.3f}")
        print(f"  verdict: {'wins' if result['wins'] else 'does not win'} (fixes {len(result['fixed'])}, breaks "
              f"{len(result['broken'])}; needs at least {WIN_FIXED} fixed and at most {WIN_BROKEN} broken)")

    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{stamp.strftime('%Y%m%dT%H%M%SZ')}-{run['variant']}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved {path} (git {run['git_commit'][:12]}{', with uncommitted changes' if run['git_dirty'] else ''})")


if __name__ == "__main__":
    main()
