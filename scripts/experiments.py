"""Retrieval variants for eval.py --variant. None of them changes the defaults in
search.py, chunk.py or embed.py: each builds what it needs beside them, and each
ends as the default does, with search.fused_order (at most PER_PAGE passages from
one page). Runs saved before the cap became the default, on 2026-11-17, were
compared against the uncapped list.

  baseline           the default fused search, as search.py runs it
  weighted           reciprocal rank fusion with embeddings weighted 2:1 against BM25
  rerank             the fused top 50 rescored by BAAI/bge-reranker-base (parked: ADR 0009)
  rewrite            the model rewrites the question into a search query first
  hyde               the model writes a passage that would answer the question; its
                     embedding replaces the question's, and BM25 searches the question
                     with the passage (HyDE: Gao, Ma, Lin and Callan 2023, "Precise
                     Zero-Shot Dense Retrieval without Relevance Labels", ACL)
  hyde-plus          the question's own two lists and the HyDE passage's embeddings,
                     all three fused
  idf-weight         BM25's weight in the fusion set by the query's rarest term (bm25_weight)
  section-feedback   the default's top passage from each of its first 3 pages, each used as a
                     query of its own (BM25 over its embed_text, embeddings by its stored
                     vector), and all eight lists fused with the question's two
  rocchio            the question's vector moved towards the mean of those 3 passages'
                     vectors (0.6 and 0.4, then normalised), searched again and fused with
                     the question's BM25 list as usual (Rocchio's relevance feedback,
                     with the top passages taken as relevant)
  overlap2           chunks with 2 sentences of overlap (ADR 0005), embedded again
  no-heading-path    embed_text without the heading path (ADR 0006), for both lists
  path-bm25-only     the heading path kept for BM25 and dropped for the embeddings (ADR 0006)
  context            A: expand.py's context sentences in embed_text, for both lists
  questions-bm25     B: expand.py's questions added to BM25's text only
  questions-vectors  C: expand.py's questions as extra vectors; a chunk scores its best match
  context-questions  D: A and C together

rewrite, hyde and hyde-plus call the answering model from .env (llm.py), standing
in for a client's own model over MCP. Its replies are cached in
corpus/evals/llm-cache.json, so a run can be repeated without calling it again.

The last four read corpus/expansions.jsonl (scripts/expand.py). Generated text
goes only into what is indexed, never into a chunk's text, so it never reaches a
prompt or the verifier.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import chunk as chunker
import search

NAMES = ["baseline", "weighted", "rerank", "rewrite", "hyde", "hyde-plus", "idf-weight", "section-feedback",
         "rocchio", "overlap2", "no-heading-path", "path-bm25-only", "context", "questions-bm25",
         "questions-vectors", "context-questions"]
FEEDBACK_PAGES = 3        # section-feedback and rocchio: passages from this many different pages
ROCCHIO = (0.6, 0.4)      # rocchio: the question's weight and the feedback passages' mean's
EXPANDED = {"context", "questions-bm25", "questions-vectors", "context-questions"}
WEIGHTS = (2, 1)          # embeddings, BM25
RERANKER = "BAAI/bge-reranker-base"
RERANK_DEPTH = 50
CACHE = Path("corpus/evals/llm-cache.json")
UNITS = Path("corpus/units.jsonl")
EXPANSIONS = Path("corpus/expansions.jsonl")

# idf-weight: BM25's weight runs from LOW, when even the query's rarest term is in COMMON
# or more of the chunks, up to 1 when its rarest term is in RARE or fewer. Set by hand,
# not tuned on the eval.
COMMON, RARE, LOW = 0.10, 0.01, 0.25

# The first version named the four sources, and the model added their names to every query
# (the first rewrite run saved in corpus/evals/, 2026-11-17). This one names none.
REWRITE = ("Rewrite the question as a search query for a library of guidance on software delivery practice. "
           "Use the terms practitioners use for the practices involved. Do not name any source, organisation "
           "or book. Reply with the query only, on one line.")
HYDE = ("You write the passage that a reference guide on software delivery practice (DORA, MinimumCD, "
        "Beyond MinimumCD, the Twelve-Factor App) would contain to answer a question. Write three to five "
        "sentences in the guide's own terms. Reply with the passage only.")


def fuse(lists, weights=None):
    """Reciprocal rank fusion of {chunk: rank} lists, each with its weight: {chunk: fused score}."""
    fused = {}
    for weight, ranks in zip(weights or [1] * len(lists), lists):
        for i, r in ranks.items():
            fused[i] = fused.get(i, 0) + weight / (search.RRF_K + r)
    return fused


def as_scores(order):
    """An order, best first, as {chunk: score} that search.fused_order sorts back into it."""
    return {i: -n for n, i in enumerate(order)}


def top_pages(order, chunks, n=FEEDBACK_PAGES):
    """The first chunk from each of the first n different pages in an order."""
    seen, picked = set(), []
    for i in order:
        if chunks[i]["unit_id"] not in seen:
            seen.add(chunks[i]["unit_id"])
            picked.append(i)
            if len(picked) == n:
                break
    return picked


def idf(df, n):
    """BM25's inverse document frequency for a term in df of n chunks, as rank-bm25's
    BM25Okapi computes it (Robertson and Zaragoza 2009)."""
    return math.log((n - df + 0.5) / (df + 0.5))


def bm25_weight(terms, df, n):
    """(BM25's weight in the fusion, the query's rarest term). The weight rises linearly in
    the rarest term's IDF, from LOW at the IDF of a term in COMMON of the chunks to 1 at the
    IDF of a term in RARE of them, and stays within those bounds. Terms in no chunk are left
    out: BM25 cannot match them."""
    known = sorted({t for t in terms if df.get(t)})
    if not known:
        return LOW, None
    rarest = max(known, key=lambda t: idf(df[t], n))
    common, rare = idf(COMMON * n, n), idf(RARE * n, n)
    share = min(max((idf(df[rarest], n) - common) / (rare - common), 0.0), 1.0)
    return LOW + (1 - LOW) * share, rarest


def with_context(embed_text, context):
    """embed_text with the context sentences between the heading path and the text."""
    path, _, body = embed_text.partition("\n\n")
    return f"{path}\n\n{context}\n\n{body}"


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_expansions(chunks):
    """{chunk id: expansion}, refusing to run unless every chunk has one made from its current text."""
    if not EXPANSIONS.exists():
        sys.exit(f"{EXPANSIONS} is missing: run ./scripts/expand.py first")
    found = {}
    for line in EXPANSIONS.read_text(encoding="utf-8").splitlines():
        if line:
            e = json.loads(line)
            found[e["chunk_id"]] = e
    stale = [c["id"] for c in chunks
             if c["id"] not in found or found[c["id"]]["text_sha256"] != text_sha256(c["text"])]
    if stale:
        sys.exit(f"{len(stale)} of {len(chunks)} chunks have no expansion made from their current text "
                 f"(the first is {stale[0]}): run ./scripts/expand.py")
    versions = sorted({(e["model"], e["prompt_version"]) for e in found.values()})
    if len(versions) > 1:
        sys.exit(f"{EXPANSIONS} mixes models or prompt versions: {versions}")
    return found


class Index:
    """Chunk vectors (with any extra vectors per chunk) and a BM25 index over the given texts."""

    def __init__(self, chunks, vectors, bm25_texts, manifest, model, extra=None):
        from rank_bm25 import BM25Okapi  # here, so the pure functions above import without it

        self.chunks, self.vectors, self.manifest, self.model = chunks, vectors, manifest, model
        self.bm25 = BM25Okapi([search.words(t) for t in bm25_texts])
        self.extra = extra  # (vectors, the chunk each belongs to), or None
        self.df = None

    def query_vector(self, text):
        return self.model.encode([self.manifest["query_instruction"] + text], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)[0]

    def passage_vector(self, text):
        return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]

    def by_embeddings(self, vector):
        """{chunk: rank} by cosine; with extra vectors, a chunk scores its best match among all of its vectors."""
        import numpy as np

        cosine = self.vectors @ vector
        if self.extra is not None:
            extra, owner = self.extra
            cosine = cosine.copy()
            np.maximum.at(cosine, owner, extra @ vector)
        return search.ranking(cosine, np.ones(len(cosine), bool))

    def by_bm25(self, text):
        scores = self.bm25.get_scores(search.words(text))
        return search.ranking(scores, scores > 0)

    def doc_freqs(self):
        """{term: number of chunks it is in}, from the BM25 index."""
        if self.df is None:
            from collections import Counter
            self.df = Counter(term for doc in self.bm25.doc_freqs for term in doc)
        return self.df


def embed_all(model, texts):
    return model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False)


def without_path(chunks, units):
    """Each chunk's embed_text as the default builds it (links reduced, tables tidied, a
    repeated table header where a piece starts inside a table), only without the heading path."""
    tables, texts = {}, []
    for c in chunks:
        if c["unit_id"] not in tables:
            tables[c["unit_id"]] = chunker.tables(units[c["unit_id"]]["text"])
        header = chunker.table_header(tables[c["unit_id"]], c["start"])
        texts.append(chunker.embed_text([], c["text"], header).strip())
    return texts


class Variant:
    """orders(question) gives {mode: chunk indices, best first} over self.chunks."""

    def __init__(self, name, chunks, vectors, manifest, model):
        self.name, self.timings, self.notes = name, [], {}
        units = {u["id"]: u for u in map(json.loads, UNITS.read_text(encoding="utf-8").splitlines())}
        bm25_texts, extra = [c["embed_text"] for c in chunks], None
        if name == "overlap2":
            chunker.OVERLAP_SENTENCES = 2
            count = chunker.token_counter()
            chunks = [c for u in units.values() for c in chunker.chunk_unit(u, count)[0]]
            bm25_texts = [c["embed_text"] for c in chunks]
            vectors = embed_all(model, bm25_texts)
        elif name in ("no-heading-path", "path-bm25-only"):
            plain = without_path(chunks, units)
            vectors = embed_all(model, plain)
            if name == "no-heading-path":
                bm25_texts = plain
        elif name in EXPANDED:
            expansions = load_expansions(chunks)
            first = next(iter(expansions.values()))
            self.notes["expansions"] = {"model": first["model"], "prompt_version": first["prompt_version"]}
            if name in ("context", "context-questions"):
                texts = [with_context(c["embed_text"], expansions[c["id"]]["context"]) for c in chunks]
                count = chunker.token_counter()
                tokens = [count(t) for t in texts]
                self.notes["context_tokens"] = {"over_max_tokens": sum(t > chunker.MAX_TOKENS for t in tokens),
                                                "over_model_limit": sum(t > model.max_seq_length for t in tokens),
                                                "model_limit": model.max_seq_length}
                vectors, bm25_texts = embed_all(model, texts), texts
            if name == "questions-bm25":
                bm25_texts = [t + "\n\n" + "\n".join(expansions[c["id"]]["questions"])
                              for t, c in zip(bm25_texts, chunks)]
            if name in ("questions-vectors", "context-questions"):
                import numpy as np
                # Generated questions are embedded as questions are, with the query instruction.
                owner = [n for n, c in enumerate(chunks) for _ in expansions[c["id"]]["questions"]]
                asked = [manifest["query_instruction"] + q for c in chunks for q in expansions[c["id"]]["questions"]]
                extra = (embed_all(model, asked), np.array(owner))
        self.chunks = chunks
        self.index = Index(chunks, vectors, bm25_texts, manifest, model, extra)
        if name == "rerank":
            from huggingface_hub import snapshot_download
            from sentence_transformers import CrossEncoder
            path = snapshot_download(RERANKER, allow_patterns=["*.json", "*.txt", "*.model", "model.safetensors"])
            self.reranker = CrossEncoder(path, max_length=512, device=model.device)
            self.reranker_revision = Path(path).name
        if name in ("rewrite", "hyde", "hyde-plus"):
            import llm
            self.config = llm.settings()
            self.cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
            self.generated = {}

    def ask(self, system, question):
        """The model's reply, from the cache if this exact request was made before."""
        import llm
        key = hashlib.sha256(json.dumps([self.config["model"], system, question]).encode()).hexdigest()
        if key not in self.cache:
            reply, provider, _ = llm.chat(self.config, system, question)
            self.cache[key] = {"model": self.config["model"], "provider": provider, "reply": reply.strip()}
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return self.cache[key]["reply"]

    def orders(self, question):
        index = self.index
        if self.name == "rewrite":
            query = self.generated[question] = self.ask(REWRITE, question)
            lists = [index.by_embeddings(index.query_vector(query)), index.by_bm25(query)]
            return self.result(lists, fuse(lists))
        if self.name == "hyde":
            passage = self.generated[question] = self.ask(HYDE, question)
            lists = [index.by_embeddings(index.passage_vector(passage)), index.by_bm25(question + " " + passage)]
            return self.result(lists, fuse(lists))
        vector = index.query_vector(question)
        lists = [index.by_embeddings(vector), index.by_bm25(question)]
        if self.name in ("section-feedback", "rocchio"):
            feedback = top_pages(search.fused_order(fuse(lists), self.chunks, top=None), self.chunks)
            self.notes.setdefault("feedback", {})[question] = [self.chunks[i]["unit_id"] for i in feedback]
            if self.name == "section-feedback":
                more = [ranks for i in feedback
                        for ranks in (index.by_embeddings(index.vectors[i]), index.by_bm25(self.chunks[i]["embed_text"]))]
                return self.result(lists, fuse(lists + more))
            import numpy as np
            moved = ROCCHIO[0] * vector + ROCCHIO[1] * index.vectors[feedback].mean(axis=0)
            lists = [index.by_embeddings(moved / np.linalg.norm(moved)), lists[1]]
            return self.result(lists, fuse(lists))
        if self.name == "hyde-plus":
            passage = self.generated[question] = self.ask(HYDE, question)
            return self.result(lists, fuse(lists + [index.by_embeddings(index.passage_vector(passage))]))
        if self.name == "weighted":
            return self.result(lists, fuse(lists, WEIGHTS))
        if self.name == "idf-weight":
            df = index.doc_freqs()
            weight, rarest = bm25_weight(search.words(question), df, len(self.chunks))
            self.notes.setdefault("bm25_weights", {})[question] = {
                "weight": round(weight, 3), "rarest": rarest, "chunks_with_it": df.get(rarest, 0)}
            return self.result(lists, fuse(lists, (1, weight)))
        fused = fuse(lists)
        if self.name == "rerank":
            order = search.fused_order(fused, self.chunks, top=None)  # capped, as the default is
            top = order[:RERANK_DEPTH]
            started = time.perf_counter()
            scores = self.reranker.predict([(question, self.chunks[i]["embed_text"]) for i in top],
                                           show_progress_bar=False)
            self.timings.append(time.perf_counter() - started)
            fused = as_scores([i for _, i in sorted(zip(scores, top), key=lambda p: -p[0])] + order[RERANK_DEPTH:])
        return self.result(lists, fused)

    def result(self, lists, fused):
        by_embeddings, by_bm25 = lists
        return {"bm25": sorted(by_bm25, key=by_bm25.get),
                "embeddings": sorted(by_embeddings, key=by_embeddings.get),
                "fused": search.fused_order(fused, self.chunks, top=None)}
