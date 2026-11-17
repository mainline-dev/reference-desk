#!/usr/bin/env -S uv run --quiet python
"""Index-time expansion, written by local models under Ollama.

Chunks: for each chunk, questions that a non-expert might ask and the passage
answers, in everyday words, and a sentence or two on what the passage is about
within its page. Saved to corpus/expansions.jsonl, one line per chunk.

Images: for each image whose alt text is empty or just "image", a vision model
writes a one-paragraph description. Saved to corpus/image-descriptions.jsonl,
one line per image, with its URL. The file is read from the source's checkout
in corpus/raw/, never fetched. Images inside code fences are code, so Mermaid
diagrams are left as they are: their source is already text.

Everything generated is for matching only. It goes into what is indexed (the
context, questions-bm25, questions-vectors and context-questions variants in
experiments.py; no variant uses image descriptions yet), never into a chunk's
text, so it is never shown, quoted or cited, and the verifier never sees it.

Settings, from the environment or .env (the environment wins):
  REFDESK_EXPAND_URL           Ollama's address (default http://localhost:11434)
  REFDESK_EXPAND_MODEL         the model for chunks (default qwen3.6:35b)
  REFDESK_EXPAND_VISION_MODEL  the model for images (default: the chunks' model)

Each page's chunks go to one worker, in order, after the same system prompt and
page, so the server can reuse what it has just read. Anything already done from
the same text or image, by the same model and prompt version, is skipped, so a
stopped run can be started again.

Usage (from the repo root):
  ./scripts/expand.py --limit 5      the first 5 chunks still to do, and the images,
                                     printed for review
  ./scripts/expand.py [--workers N]  every chunk and image still to do
"""
import argparse
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import chunk   # fence_after: CommonMark code fences, so nothing in code counts as an image
import verify  # reduce_links: page and passage as answer.py shows passages, links reduced to their words

CHUNKS = Path("corpus/chunks.jsonl")
UNITS = Path("corpus/units.jsonl")
RAW = Path("corpus/raw")
EXPANSIONS = Path("corpus/expansions.jsonl")
IMAGE_DESCRIPTIONS = Path("corpus/image-descriptions.jsonl")
DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.6:35b"
PROMPT_VERSION = "e1"
IMAGE_PROMPT_VERSION = "i1"
QUESTIONS = 5
MAX_QUESTION_WORDS = 40
MAX_CONTEXT_WORDS = 80
MAX_DESCRIPTION_WORDS = 200
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}  # what a vision model reads; SVG would need rasterising
ATTEMPTS = 3
# The sampling the Qwen3.6-35B-A3B model card recommends for non-thinking use, with a seed.
# A little randomness helps five questions differ from each other; the file, not a rerun,
# is the record of what was generated.
OPTIONS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
           "num_ctx": 16384}  # the longest page is about 6,700 words: well inside, with the passage

SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}, "minItems": QUESTIONS, "maxItems": QUESTIONS},
        "context": {"type": "string"},
    },
    "required": ["questions", "context"],
}

SYSTEM = ("You help a search index match people's questions to passages from guidance on software delivery "
          "practice. You are shown a page and one passage from it. Reply with JSON only.")
IMAGE_SYSTEM = ("You describe images from guidance on software delivery practice, so that a search index can "
                "match people's questions to them. Reply with one paragraph of plain text.")

MD_IMAGE = re.compile(r'!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)')
HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.I)


def user_prompt(unit, chunk_):
    """The page first and the passage after it, so every chunk of a page shares the same start."""
    page = verify.reduce_links(unit["text"])[0].strip()
    passage = verify.reduce_links(chunk_["text"])[0].strip()
    return (f"Page: {unit['title']}\n\n{page}\n\n"
            f"---\n\nPassage, from the section \"{' > '.join(chunk_['heading_path'])}\":\n\n{passage}\n\n---\n\n"
            f"About the passage, not the page as a whole, write:\n"
            f"- \"questions\": {QUESTIONS} different questions that someone new to the subject might ask and that "
            f"this passage answers. Use everyday words, the way they would ask, not the passage's own terms.\n"
            f"- \"context\": one or two sentences on what this passage is about within its page.\n"
            f'Reply with JSON only: {{"questions": ["...", "...", "...", "...", "..."], "context": "..."}}')


def parse(content):
    """(questions, context) from the model's reply, or ValueError saying what is wrong with it."""
    try:
        reply = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"not JSON: {e}") from None
    if not isinstance(reply, dict):
        raise ValueError("not a JSON object")
    questions, context = reply.get("questions"), reply.get("context")
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise ValueError("questions is not a list of strings")
    questions = [" ".join(q.split()) for q in questions]
    if len(questions) != QUESTIONS or not all(questions):
        raise ValueError(f"{len([q for q in questions if q])} questions, not {QUESTIONS}")
    if len({q.lower() for q in questions}) != QUESTIONS:
        raise ValueError("questions repeat")
    if any(len(q.split()) > MAX_QUESTION_WORDS for q in questions):
        raise ValueError(f"a question is over {MAX_QUESTION_WORDS} words")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("no context")
    context = " ".join(context.split())
    if len(context.split()) > MAX_CONTEXT_WORDS:
        raise ValueError(f"context is over {MAX_CONTEXT_WORDS} words")
    return questions, context


def attribute(tag, name):
    match = re.search(rf"""\b{name}\s*=\s*(?:"([^"]*)"|'([^']*)')""", tag, re.I)
    return None if match is None else (match.group(1) if match.group(1) is not None else match.group(2))


def find_images(text):
    """[(src, alt)] for every Markdown image and <img> tag outside code fences, in order."""
    found, fence = [], None
    for line in text.splitlines():
        was, fence = fence, chunk.fence_after(line, fence)
        if was is not None or fence is not None:
            continue  # a fence line or code, Mermaid diagrams included: left as it is
        found += [(src, alt.strip()) for alt, src in MD_IMAGE.findall(line)]
        for tag in HTML_IMAGE.findall(line):
            src = attribute(tag, "src")
            if src:
                found.append((src, (attribute(tag, "alt") or "").strip()))
    return found


def needs_description(alt):
    return not alt or alt.lower() == "image"


def image_file(unit, src):
    """The image's file in the source's checkout, or None if it is served from somewhere else."""
    if urlparse(src).scheme or src.startswith("/"):
        return None
    root = (RAW / unit["source"]).resolve()
    path = (root / Path(unit["path"]).parent / unquote(src)).resolve()
    return path if path.is_relative_to(root) else None


def image_prompt(unit, src, heading_path):
    """Where the image is, and the text just before and after it, links reduced to their words."""
    text = unit["text"]
    at = max(text.find(src), 0)
    around = verify.reduce_links(text[max(at - 1500, 0):at + 500])[0].strip()
    return (f"This image is on the page \"{unit['title']}\", in the section \"{' > '.join(heading_path)}\". "
            f"The text around it:\n\n{around}\n\n---\n\n"
            "Describe what the image shows in one paragraph of plain text: what kind of image it is (a chart, "
            "diagram, table, screenshot or photo), what it shows, and any labels, axes, steps or numbers in it. "
            "Describe only what is in the image.")


def parse_description(content):
    """One paragraph from the model's reply, or ValueError saying what is wrong with it."""
    description = " ".join(content.split())
    if not description:
        raise ValueError("no description")
    if len(description.split()) > MAX_DESCRIPTION_WORDS:
        raise ValueError(f"description is over {MAX_DESCRIPTION_WORDS} words")
    return description


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def text_sha256(text):
    return sha256(text.encode("utf-8"))


def current(record, chunk_, model):
    return (record is not None and record["text_sha256"] == text_sha256(chunk_["text"])
            and record["model"] == model and record["prompt_version"] == PROMPT_VERSION)


def current_image(record, image_sha256, model):
    return (record is not None and record["image_sha256"] == image_sha256
            and record["model"] == model and record["prompt_version"] == IMAGE_PROMPT_VERSION)


def settings():
    from dotenv import load_dotenv
    load_dotenv()  # .env fills in only what the environment does not set
    model = os.environ.get("REFDESK_EXPAND_MODEL", DEFAULT_MODEL)
    return (os.environ.get("REFDESK_EXPAND_URL", DEFAULT_URL).rstrip("/"), model,
            os.environ.get("REFDESK_EXPAND_VISION_MODEL", model))


def model_digest(client, url, model):
    """The digest of the model as the server has it, or exit saying how to get it."""
    import httpx
    try:
        tags = client.get(f"{url}/api/tags", timeout=10).json()["models"]
    except httpx.HTTPError as e:
        sys.exit(f"Cannot reach Ollama at {url} ({e}). Start it there, or set REFDESK_EXPAND_URL. "
                 f"For another machine, start Ollama on it with OLLAMA_HOST=0.0.0.0.")
    for m in tags:
        if model in (m.get("name"), m.get("model")):
            return m["digest"]
    sys.exit(f"{model} is not on the server at {url}: run `ollama pull {model}` there.")


def chat(client, url, model, system, user, read, images=None, schema=None):
    """(read(reply), seed): another seed is tried when a reply does not read."""
    import httpx
    problem = None
    for seed in range(ATTEMPTS):
        message = {"role": "user", "content": user, **({"images": images} if images else {})}
        body = {"model": model, "stream": False, "think": False, "keep_alive": "30m",
                "options": {**OPTIONS, "seed": seed},
                "messages": [{"role": "system", "content": system}, message], **({"format": schema} if schema else {})}
        try:
            response = client.post(f"{url}/api/chat", json=body, timeout=600)
            response.raise_for_status()
            reply = response.json()["message"]
        except httpx.HTTPError as e:
            problem = f"request failed: {e}"
            time.sleep(2 ** seed)
            continue
        if reply.get("thinking"):
            raise SystemExit(f"{model} thought before answering although thinking was turned off; "
                             "pick a model that can answer without thinking")
        try:
            return read(reply["content"]), seed
        except ValueError as e:
            problem = str(e)
    raise RuntimeError(problem)


def read_jsonl(path, key):
    """{key: record}, a later line for the same key winning, or {} if there is no file."""
    found = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                record = json.loads(line)
                found[record[key]] = record
    return found


def expand_chunks(client, url, model, chunks, units, limit, workers):
    done = read_jsonl(EXPANSIONS, "chunk_id")
    todo = [c for c in chunks if not current(done.get(c["id"]), c, model)]
    skipped = len(chunks) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    pages = {}
    for c in todo:
        pages.setdefault(c["unit_id"], []).append(c)
    digest = model_digest(client, url, model)
    print(f"Chunks: {model} ({digest[:12]}) at {url}, prompt {PROMPT_VERSION}: {len(todo)} chunks from "
          f"{len(pages)} pages to expand, {skipped} already done, {workers} workers")

    lock, made, failed = threading.Lock(), [], []
    EXPANSIONS.parent.mkdir(parents=True, exist_ok=True)
    out = EXPANSIONS.open("a", encoding="utf-8")
    started = time.perf_counter()

    def expand_page(page_chunks):
        unit = units[page_chunks[0]["unit_id"]]
        for c in page_chunks:
            began = time.perf_counter()
            try:
                (questions, context), seed = chat(client, url, model, SYSTEM, user_prompt(unit, c), parse,
                                                  schema=SCHEMA)
            except RuntimeError as e:
                with lock:
                    failed.append((c["id"], str(e)))
                continue
            record = {"chunk_id": c["id"], "unit_id": c["unit_id"], "text_sha256": text_sha256(c["text"]),
                      "model": model, "model_digest": digest, "prompt_version": PROMPT_VERSION,
                      "options": {**OPTIONS, "seed": seed}, "questions": questions, "context": context,
                      "seconds": round(time.perf_counter() - began, 2),
                      "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with lock:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                made.append(record)
                if len(made) % 50 == 0:
                    rate = (time.perf_counter() - started) / len(made)
                    print(f"  {len(made)} of {len(todo)}, {rate:.1f} s a chunk, about "
                          f"{rate * (len(todo) - len(made)) / 60:.0f} min to go", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(expand_page, pages.values()))
    out.close()
    elapsed = time.perf_counter() - started

    if limit is None and not failed:
        latest = read_jsonl(EXPANSIONS, "chunk_id")  # one current line per chunk, in chunk order
        keep = [latest[c["id"]] for c in chunks if current(latest.get(c["id"]), c, model)]
        EXPANSIONS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    if limit is not None:
        by_id = {c["id"]: c for c in chunks}
        for r in made:
            c = by_id[r["chunk_id"]]
            print(f"\n{' > '.join(c['heading_path'])} ({r['seconds']} s)")
            print(f"  passage: {' '.join(c['text'].split())[:160]}...")
            print(f"  context: {r['context']}")
            for q in r["questions"]:
                print(f"  - {q}")
    print(f"\n{len(made)} chunks expanded in {elapsed / 60:.1f} min"
          f"{f', {elapsed / len(made):.1f} s a chunk' if made else ''}; {len(failed)} failed")
    return failed


def describe_images(client, url, model, chunks, units, show):
    by_unit = {}
    for c in chunks:
        by_unit.setdefault(c["unit_id"], []).append(c)
    images = {}  # absolute URL: (unit, src, alt, chunk ids)
    for unit in units.values():
        for src, alt in find_images(unit["text"]):
            if needs_description(alt):
                ids = [c["id"] for c in by_unit.get(unit["id"], []) if src in c["text"]]
                images.setdefault(urljoin(unit["url"], src), (unit, src, alt, ids))
    done = read_jsonl(IMAGE_DESCRIPTIONS, "url")
    digest = model_digest(client, url, model)
    print(f"\nImages: {model} ({digest[:12]}), prompt {IMAGE_PROMPT_VERSION}: {len(images)} without useful alt text")

    keep, failed = [], []
    for image_url, (unit, src, alt, ids) in images.items():
        path = image_file(unit, src)
        if path is None or not path.exists():
            print(f"  skipped {image_url}: not in the checkout in {RAW}/")
            continue
        if path.suffix.lower() not in IMAGE_TYPES:
            print(f"  skipped {image_url}: {path.suffix} is not a type the vision model reads")
            continue
        data = path.read_bytes()
        record = done.get(image_url)
        if not current_image(record, sha256(data), model):
            heading_path = next((c["heading_path"] for c in by_unit.get(unit["id"], []) if c["id"] in ids),
                                [unit["title"]])
            began = time.perf_counter()
            try:
                description, seed = chat(client, url, model, IMAGE_SYSTEM, image_prompt(unit, src, heading_path),
                                         parse_description, images=[base64.b64encode(data).decode("ascii")])
            except RuntimeError as e:
                failed.append((image_url, str(e)))
                continue
            record = {"url": image_url, "src": src, "unit_id": unit["id"], "chunk_ids": ids, "alt": alt,
                      "image_sha256": sha256(data), "model": model, "model_digest": digest,
                      "prompt_version": IMAGE_PROMPT_VERSION, "options": {**OPTIONS, "seed": seed},
                      "description": description, "seconds": round(time.perf_counter() - began, 2),
                      "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if show:
                print(f"\n{image_url} ({record['seconds']} s, in {', '.join(ids) or 'no chunk'})\n  {description}")
        keep.append(record)
    IMAGE_DESCRIPTIONS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    print(f"{len(keep)} images described in {IMAGE_DESCRIPTIONS}; {len(failed)} failed")
    return failed


def main():
    parser = argparse.ArgumentParser(description="Generate matching-only text for chunks and images.")
    parser.add_argument("--limit", type=int, help="expand only this many chunks, and print what came back")
    parser.add_argument("--workers", type=int, default=4, help="pages in flight at once (match OLLAMA_NUM_PARALLEL)")
    args = parser.parse_args()

    import httpx
    url, model, vision_model = settings()
    chunks = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line]
    units = {u["id"]: u for u in map(json.loads, UNITS.read_text(encoding="utf-8").splitlines())}
    client = httpx.Client(limits=httpx.Limits(max_connections=args.workers))
    failed = expand_chunks(client, url, model, chunks, units, args.limit, args.workers)
    failed += describe_images(client, url, vision_model, chunks, units, show=args.limit is not None)
    for key, problem in failed[:10]:
        print(f"  {key}: {problem}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
