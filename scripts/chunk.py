#!/usr/bin/env -S uv run --quiet python
"""Cut every unit in corpus/units.jsonl into chunks, written to corpus/chunks.jsonl.

Each chunk's `text` is an exact slice of its unit's text, unit["text"][start:end],
so the verifier can check quotes against it. `embed_text` is what gets embedded
and indexed: the heading path, a blank line, then the text with each link
reduced to its words and any "Referenced in:" list left out. Every embed_text
fits in MAX_TOKENS by the embedding model's own tokeniser; a chunk that would
not is cut at sentence ends, line breaks or spaces until each piece does.

Usage (from the repo root): ./scripts/chunk.py, which runs it with uv in the
project environment (it needs transformers). Tests: uv run python -m unittest discover -s tests
"""
import json
import math
import random
import re
import sys
from pathlib import Path

UNITS = Path("corpus/units.jsonl")
CHUNKS = Path("corpus/chunks.jsonl")

MAX_WORDS = 300          # ADR 0004: 200 to 400 tokens. A target only: MAX_TOKENS is the hard limit (ADR 0007)
MIN_WORDS = 40           # ADR 0004: smaller chunks join the next one (our own rule)
OVERLAP_SENTENCES = 0    # ADR 0005: 0 or 2, decided by the eval
MIN_KEEP = 8             # drop chunks shorter than this: stray labels, empty sections (our own rule)
SKIP_SECTIONS = {"related content", "see also"}  # navigation, not content (our own rule)
TOKENISER = "BAAI/bge-small-en-v1.5"  # ADR 0007: the embedding model, measured with its own tokeniser
MAX_TOKENS = 480         # ADR 0007: the model reads 512 tokens, special tokens included; 480 leaves headroom

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[*_`])")
LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
REFERENCED_IN = re.compile(r"(?m)^[ \t]*Referenced in:[^\n]*(?:\n(?![ \t]*(?:\n|$))[^\n]*)*")


def words(text):
    return len(text.split())


def lines_with_offsets(text):
    pos = 0
    for line in text.splitlines(keepends=True):
        yield pos, line
        pos += len(line)


def fence_after(line, fence):
    """The fence still open after this line, or None outside code.

    As in CommonMark, a fence closes only on a run of its own character at least
    as long as the run that opened it, with nothing after it, so a ``` line
    inside a ```` block is code.
    """
    match = FENCE.match(line)
    if fence is None:
        return match.group(1) if match else None
    run = match.group(1) if match else ""
    if run[:1] == fence[0] and len(run) >= len(fence) and not line[match.end():].strip():
        return None
    return fence


def sections(text, title):
    """Split at every heading. Yields (heading_path, start, end) for each body.

    ADR 0004: headings first. Every level counts, so glossary terms (###) and
    twelve-factor rules (####) become their own sections. Headings inside code
    fences are ignored.
    """
    stack, start, fence = [], 0, None
    path = [title]
    for pos, line in lines_with_offsets(text):
        was_open = fence is not None
        fence = fence_after(line, fence)
        if was_open or fence is not None:
            continue  # a fence line, or a line inside one
        match = HEADING.match(line.rstrip("\n"))
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
    spans, block_start, fence, pos = [], None, None, start
    for line in text[start:end].splitlines(keepends=True):
        fence = fence_after(line, fence)
        blank = not line.strip()
        if block_start is None and not blank:
            block_start = pos
        if blank and fence is None and block_start is not None:
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


def embed_text(path, text):
    """ADR 0006: the heading path goes in front of what is embedded and indexed.

    Links keep their words and lose their URLs, and "Referenced in:" lists of
    pages are left out: both are navigation, not content (our own rule).
    """
    body = LINK.sub(r"\1", REFERENCED_IN.sub("", text))
    return " > ".join(path) + "\n\n" + re.sub(r"\n{3,}", "\n\n", body).strip()


def atoms(text, span, fits, marks=(r"\n", r"[ \t]+")):
    """A span that fits, else its lines, else its words, each an exact slice."""
    start, end = span
    if fits(start, end) or not marks:
        return [span]
    pieces, last = [], start
    for match in re.finditer(marks[0], text[start:end]):
        pieces.append((last, start + match.start()))
        last = start + match.end()
    pieces.append((last, end))
    pieces = [(s, e) for s, e in pieces if text[s:e].strip()]
    if len(pieces) == 1:
        return atoms(text, span, fits, marks[1:])
    return [atom for piece in pieces for atom in atoms(text, piece, fits, marks[1:])]


def fit(text, path, start, end, count):
    """ADR 0007: cut a chunk whose embed_text is over MAX_TOKENS into pieces of
    about equal size that fit, at sentence ends, else line breaks, else spaces."""
    tokens = lambda s, e: count(embed_text(path, text[s:e]))
    fits = lambda s, e: tokens(s, e) <= MAX_TOKENS
    spans = [a for b in blocks(text, start, end) for s in sentences(text, *b) for a in atoms(text, s, fits)]
    total = tokens(start, end)
    target = total / math.ceil(total / MAX_TOKENS)
    pieces = [list(spans[0])]
    for s, e in spans[1:]:
        if not fits(pieces[-1][0], e) or tokens(*pieces[-1]) >= target:
            pieces.append([s, e])
        else:
            pieces[-1][1] = e
    # A small piece, such as a last sentence, joins a neighbour if the two still
    # fit, so it is not lost to MIN_KEEP (our own rule).
    pieces = [tuple(piece) for piece in pieces]
    joined = True
    while joined and len(pieces) > 1:
        joined = False
        for i, (s, e) in enumerate(pieces):
            if words(text[s:e]) >= MIN_WORDS:
                continue
            for j in (i - 1, i + 1):
                a, b = sorted((i, j))
                if 0 <= j < len(pieces) and fits(pieces[a][0], pieces[b][1]):
                    pieces[a:b + 1] = [(pieces[a][0], pieces[b][1])]
                    joined = True
                    break
            if joined:
                break
    return pieces


def chunk_unit(unit, count):
    """The unit's chunks, and how many of them had to be cut to fit MAX_TOKENS."""
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

    chunks, cut = [], 0
    for path, start, end in merged:
        if path[-1].lower() in SKIP_SECTIONS:
            continue
        pieces = [(start, end)]
        if count(embed_text(path, text[start:end])) > MAX_TOKENS:
            pieces = fit(text, path, start, end, count)
            cut += 1
        for s, e in pieces:
            body = text[s:e]
            if only_links(body) or words(body) < MIN_KEEP:
                continue
            chunks.append({
                "id": f'{unit["id"]}#{len(chunks)}',
                "unit_id": unit["id"],
                "ord": len(chunks),
                "start": s,
                "end": e,
                "heading_path": path,
                "text": body,
                "embed_text": embed_text(path, body),
            })
    return chunks, cut


def token_counter():
    """Tokens in a string by TOKENISER, special tokens included (ADR 0007)."""
    from transformers import AutoTokenizer, logging

    logging.set_verbosity_error()  # no warning for each text over 512 tokens
    try:  # the cached copy, so a run needs no network
        tokeniser = AutoTokenizer.from_pretrained(TOKENISER, local_files_only=True)
    except OSError:  # not cached yet: download it once
        tokeniser = AutoTokenizer.from_pretrained(TOKENISER)
    return lambda text: len(tokeniser(text)["input_ids"])


def main():
    count = token_counter()
    units = [json.loads(line) for line in UNITS.read_text(encoding="utf-8").splitlines() if line]
    chunks, cut = [], 0
    for unit in units:
        found, n = chunk_unit(unit, count)
        chunks += found
        cut += n

    by_id = {unit["id"]: unit for unit in units}
    for c in chunks:  # every chunk must be an exact slice of its unit
        assert c["text"] == by_id[c["unit_id"]]["text"][c["start"]:c["end"]]
    tokens = sorted(count(c["embed_text"]) for c in chunks)
    assert tokens[-1] <= MAX_TOKENS, f"an embed_text has {tokens[-1]} tokens, over {MAX_TOKENS}"

    with CHUNKS.open("w", encoding="utf-8") as out:
        for c in chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")

    counts = {}
    for c in chunks:
        source = c["unit_id"].split("/")[0]
        counts[source] = counts.get(source, 0) + 1
    sizes = sorted(words(c["text"]) for c in chunks)
    print(f"{len(chunks)} chunks from {len(units)} units ({cut} cut to fit {MAX_TOKENS} tokens)")
    for source, n in sorted(counts.items()):
        print(f"  {source}: {n}")
    print(f"words per chunk: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print(f"tokens per embed_text: median {tokens[len(tokens) // 2]}, max {tokens[-1]}")

    if "--sample" in sys.argv:
        for c in random.sample(chunks, 5):
            print("\n---", " > ".join(c["heading_path"]), f'({words(c["text"])} words)')
            print(c["text"][:600])


if __name__ == "__main__":
    main()
