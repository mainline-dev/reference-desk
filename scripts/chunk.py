#!/usr/bin/env python3
"""Cut every unit in corpus/units.jsonl into chunks, written to corpus/chunks.jsonl.

Each chunk's `text` is an exact slice of its unit's text, unit["text"][start:end],
so the verifier can check quotes against it. `embed_text` is what gets embedded
and indexed: the heading path, a blank line, then the text.

Usage (from the repo root): ./scripts/chunk.py
"""
import json
import random
import re
import sys
from pathlib import Path

UNITS = Path("corpus/units.jsonl")
CHUNKS = Path("corpus/chunks.jsonl")

MAX_WORDS = 300          # ADR 0004: 200 to 400 tokens; 300 words plus the path stays under 512 (ADR 0007)
MIN_WORDS = 40           # ADR 0004: smaller chunks join the next one (our own rule)
OVERLAP_SENTENCES = 0    # ADR 0005: 0 or 2, decided by the eval
MIN_KEEP = 8             # drop chunks shorter than this: stray labels, empty sections (our own rule)
SKIP_SECTIONS = {"related content", "see also"}  # navigation, not content (our own rule)

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[*_`])")
LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def words(text):
    return len(text.split())


def lines_with_offsets(text):
    pos = 0
    for line in text.splitlines(keepends=True):
        yield pos, line
        pos += len(line)


def sections(text, title):
    """Split at every heading. Yields (heading_path, start, end) for each body.

    ADR 0004: headings first. Every level counts, so glossary terms (###) and
    twelve-factor rules (####) become their own sections. Headings inside code
    fences are ignored.
    """
    stack, start, in_fence = [], 0, False
    path = [title]
    for pos, line in lines_with_offsets(text):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        match = None if in_fence else HEADING.match(line.rstrip("\n"))
        if match:
            if pos > start:
                yield path, start, pos
            level, name = len(match.group(1)), match.group(2).strip()
            stack = [(lvl, n) for lvl, n in stack if lvl < level] + [(level, name)]
            path = [title] + [n for _, n in stack if n != title]
            start = pos + len(line)
    if len(text) > start:
        yield path, start, len(text)


def blocks(text, start, end):
    """Paragraphs as (start, end) spans. A code fence or table stays one block."""
    spans, block_start, in_fence, pos = [], None, False, start
    for line in text[start:end].splitlines(keepends=True):
        fence = FENCE.match(line)
        blank = not line.strip()
        if fence:
            in_fence = not in_fence
        if block_start is None and not blank:
            block_start = pos
        if blank and not in_fence and block_start is not None:
            spans.append((block_start, pos))
            block_start = None
        pos += len(line)
    if block_start is not None:
        spans.append((block_start, end))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def sentences(text, start, end):
    """Sentence spans within a paragraph. Code, tables and lists are not split.

    ADR 0004: never cut mid-sentence, because the verifier checks whole sentences.
    """
    block = text[start:end]
    if FENCE.match(block) or block.lstrip().startswith("|"):
        return [(start, end)]
    spans, last = [], 0
    for match in SENTENCE_END.finditer(block):
        spans.append((start + last, start + match.start()))
        last = match.end()
    spans.append((start + last, end))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def split_long(text, span):
    """A single span over MAX_WORDS (a big code block, table or run-on paragraph)
    is cut at line breaks, or at word boundaries if it has none, so nothing is
    cut off by the model's 512-token limit (ADR 0007)."""
    start, end = span
    if words(text[start:end]) <= MAX_WORDS:
        return [span]
    pieces, piece_start, pos = [], start, start
    marks = [m.end() for m in re.finditer(r"\n", text[start:end])] or [m.end() for m in re.finditer(r"\s+", text[start:end])]
    for mark in marks:
        cut = start + mark
        if words(text[piece_start:cut]) > MAX_WORDS and pos > piece_start:
            pieces.append((piece_start, pos))
            piece_start = pos
        pos = cut
    pieces.append((piece_start, end))
    return [(s, e) for s, e in pieces if text[s:e].strip()]


def pack(text, spans):
    """Group sentence spans into chunks of at most MAX_WORDS, with optional overlap."""
    chunks, current = [], []
    spans = [piece for span in spans for piece in split_long(text, span)]
    for span in spans:
        if current and words(text[current[0][0]:span[1]]) > MAX_WORDS:
            chunks.append((current[0][0], current[-1][1]))
            current = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
        current.append(span)
    if current:
        chunks.append((current[0][0], current[-1][1]))
    return chunks


def only_links(text):
    """True for link lists with almost no prose, like many cd-migration index pages."""
    prose = LINK.sub("", text)
    prose = re.sub(r"^\s*([-*+]|\d+\.)\s*", "", prose, flags=re.M)
    return len(LINK.findall(text)) >= 2 and words(prose) < 10


def chunk_unit(unit):
    text, title = unit["text"], unit["title"]
    merged = []
    for path, start, end in sections(text, title):
        spans = [s for b in blocks(text, start, end) for s in sentences(text, *b)]
        packed = pack(text, spans)
        # Join a small piece onto its neighbour, but only within its own section,
        # so a short glossary term or rule stays a chunk of its own (our own rule).
        while len(packed) > 1:
            small = next((i for i, (s, e) in enumerate(packed) if words(text[s:e]) < MIN_WORDS), None)
            if small is None:
                break
            j = small + 1 if small + 1 < len(packed) else small - 1
            a, b = sorted((small, j))
            packed[a:b + 1] = [(packed[a][0], packed[b][1])]
        merged += [(path, s, e) for s, e in packed]

    chunks = []
    for path, start, end in merged:
        body = text[start:end]
        if only_links(body) or words(body) < MIN_KEEP or path[-1].lower() in SKIP_SECTIONS:
            continue
        chunks.append({
            "id": f'{unit["id"]}#{len(chunks)}',
            "unit_id": unit["id"],
            "ord": len(chunks),
            "start": start,
            "end": end,
            "heading_path": path,
            "text": body,
            # ADR 0006: the heading path goes in front of what is embedded and indexed.
            "embed_text": " > ".join(path) + "\n\n" + body,
        })
    return chunks


def main():
    units = [json.loads(line) for line in UNITS.read_text(encoding="utf-8").splitlines() if line]
    chunks = [c for unit in units for c in chunk_unit(unit)]

    for c in chunks:  # every chunk must be an exact slice of its unit
        assert c["text"] == next(u for u in units if u["id"] == c["unit_id"])["text"][c["start"]:c["end"]]

    with CHUNKS.open("w", encoding="utf-8") as out:
        for c in chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")

    counts = {}
    for c in chunks:
        source = c["unit_id"].split("/")[0]
        counts[source] = counts.get(source, 0) + 1
    sizes = sorted(words(c["text"]) for c in chunks)
    print(f"{len(chunks)} chunks from {len(units)} units")
    for source, n in sorted(counts.items()):
        print(f"  {source}: {n}")
    print(f"words per chunk: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")

    if "--sample" in sys.argv:
        for c in random.sample(chunks, 5):
            print("\n---", " > ".join(c["heading_path"]), f'({words(c["text"])} words)')
            print(c["text"][:600])


if __name__ == "__main__":
    main()
