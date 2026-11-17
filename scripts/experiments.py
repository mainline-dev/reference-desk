"""Retrieval variants for eval.py --variant. None of them changes the defaults in
search.py, chunk.py or embed.py: each builds what it needs beside them.

  baseline         the default fused search, as search.py runs it
  weighted         reciprocal rank fusion with embeddings weighted 2:1 against BM25
  cap2             the default fused list with at most 2 passages per page
  rerank           the fused top 50 rescored by BAAI/bge-reranker-base, run locally
  rewrite          the model rewrites the question into a search query first
  hyde             the model writes a passage that would answer the question; its
                   embedding stands in for the question's, and BM25 searches the
                   question with the passage (HyDE: Gao, Ma, Lin and Callan 2023,
                   "Precise Zero-Shot Dense Retrieval without Relevance Labels", ACL)
  overlap2         chunks with 2 sentences of overlap (ADR 0005), embedded again
  no-heading-path  embed_text without the heading path in front (ADR 0006),
                   embedded and indexed again

rewrite and hyde call the answering model from .env (llm.py), standing in for a
client's own model over MCP. Its replies are cached in corpus/evals/llm-cache.json,
so a run can be repeated without calling it again.
"""
import hashlib
import json
import time
from pathlib import Path

import chunk as chunker
import search

NAMES = ["baseline", "weighted", "cap2", "rerank", "rewrite", "hyde", "overlap2", "no-heading-path"]
WEIGHTS = (2, 1)          # embeddings, BM25
PER_PAGE = 2
RERANKER = "BAAI/bge-reranker-base"
RERANK_DEPTH = 50
CACHE = Path("corpus/evals/llm-cache.json")

# The first version named the four sources, and the model added their names to every query
# (run 20261117T*-rewrite.json before this change). This one names none.
REWRITE = ("Rewrite the question as a search query for a library of guidance on software delivery practice. "
           "Use the terms practitioners use for the practices involved. Do not name any source, organisation "
           "or book. Reply with the query only, on one line.")
HYDE = ("You write the passage that a reference guide on software delivery practice (DORA, MinimumCD, "
        "Beyond MinimumCD, the Twelve-Factor App) would contain to answer a question. Write three to five "
        "sentences in the guide's own terms. Reply with the passage only.")


def fuse(by_embeddings, by_bm25, weights=(1, 1)):
    """Reciprocal rank fusion of two {chunk: rank} lists, each with its weight: chunk indices, best first."""
    fused = {}
    for weight, ranks in zip(weights, (by_embeddings, by_bm25)):
        for i, r in ranks.items():
            fused[i] = fused.get(i, 0) + weight / (search.RRF_K + r)
    return sorted(fused, key=lambda i: (-fused[i], i))


def cap_per_page(order, chunks, limit=PER_PAGE):
    """The order with at most `limit` passages from any one page."""
    seen, kept = {}, []
    for i in order:
        unit = chunks[i]["unit_id"]
        if seen.get(unit, 0) < limit:
            kept.append(i)
            seen[unit] = seen.get(unit, 0) + 1
    return kept


def ranks(scores, allowed):
    return search.ranking(scores, allowed)


class Index:
    """Chunks with their vectors and a BM25 index over the given texts."""

    def __init__(self, chunks, vectors, texts, manifest, model):
        from rank_bm25 import BM25Okapi  # here, so the pure functions above import without it

        self.chunks, self.vectors, self.manifest, self.model = chunks, vectors, manifest, model
        self.bm25 = BM25Okapi([search.words(t) for t in texts])

    def query_vector(self, text):
        return self.model.encode([self.manifest["query_instruction"] + text], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)[0]

    def passage_vector(self, text):
        return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]

    def lists(self, bm25_text, vector):
        """({chunk: rank} by embeddings, {chunk: rank} by BM25)."""
        cosine = self.vectors @ vector
        bm25 = self.bm25.get_scores(search.words(bm25_text))
        return ranks(cosine, cosine == cosine), ranks(bm25, bm25 > 0)

    def orders(self, by_embeddings, by_bm25, fused):
        return {"bm25": sorted(by_bm25, key=by_bm25.get),
                "embeddings": sorted(by_embeddings, key=by_embeddings.get),
                "fused": fused}


def embed_all(model, texts):
    return model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False)


class Variant:
    """orders(question) gives {mode: chunk indices, best first} over self.chunks."""

    def __init__(self, name, chunks, vectors, manifest, model):
        self.name, self.timings = name, []
        texts = [c["embed_text"] for c in chunks]
        if name == "overlap2":
            chunker.OVERLAP_SENTENCES = 2
            count = chunker.token_counter()
            units = [json.loads(line) for line in Path("corpus/units.jsonl").read_text(encoding="utf-8").splitlines()]
            chunks = [c for u in units for c in chunker.chunk_unit(u, count)[0]]
            texts = [c["embed_text"] for c in chunks]
            vectors = embed_all(model, texts)
        elif name == "no-heading-path":
            # The same embed_text as the default (links reduced, tables tidied, a repeated table
            # header where a piece starts inside a table), only without the heading path.
            units = {u["id"]: u for u in map(json.loads, Path("corpus/units.jsonl").read_text(encoding="utf-8").splitlines())}
            tables = {}
            texts = []
            for c in chunks:
                if c["unit_id"] not in tables:
                    tables[c["unit_id"]] = chunker.tables(units[c["unit_id"]]["text"])
                header = chunker.table_header(tables[c["unit_id"]], c["start"])
                texts.append(chunker.embed_text([], c["text"], header).strip())
            vectors = embed_all(model, texts)
        self.index = Index(chunks, vectors, texts, manifest, model)
        self.chunks = chunks
        if name == "rerank":
            from huggingface_hub import snapshot_download
            from sentence_transformers import CrossEncoder
            path = snapshot_download(RERANKER, allow_patterns=["*.json", "*.txt", "*.model", "model.safetensors"])
            self.reranker = CrossEncoder(path, max_length=512, device=model.device)
            self.reranker_revision = Path(path).name
        if name in ("rewrite", "hyde"):
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
            query = self.ask(REWRITE, question)
            self.generated[question] = query
            by_e, by_b = index.lists(query, index.query_vector(query))
            return index.orders(by_e, by_b, fuse(by_e, by_b))
        if self.name == "hyde":
            passage = self.ask(HYDE, question)
            self.generated[question] = passage
            by_e, by_b = index.lists(question + " " + passage, index.passage_vector(passage))
            return index.orders(by_e, by_b, fuse(by_e, by_b))
        by_e, by_b = index.lists(question, index.query_vector(question))
        if self.name == "weighted":
            return index.orders(by_e, by_b, fuse(by_e, by_b, WEIGHTS))
        fused = fuse(by_e, by_b)
        if self.name == "cap2":
            fused = cap_per_page(fused, self.chunks)
        elif self.name == "rerank":
            top = fused[:RERANK_DEPTH]
            started = time.perf_counter()
            scores = self.reranker.predict([(question, self.chunks[i]["embed_text"]) for i in top],
                                           show_progress_bar=False)
            self.timings.append(time.perf_counter() - started)
            fused = [i for _, i in sorted(zip(scores, top), key=lambda p: -p[0])] + fused[RERANK_DEPTH:]
        return index.orders(by_e, by_b, fused)
