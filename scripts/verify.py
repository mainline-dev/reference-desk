#!/usr/bin/env -S uv run --quiet python
"""Check a saved answer's quotes against the passages they cite.

Each claim should quote one exact sentence in double quotes followed by its
passage number, like "Work integrates to trunk at least daily." [c2]. Every
part of the answer gets one of three outcomes:

  checked    a quote found in its passage's text (not embed_text), ignoring
             white space and link markup, nothing else: the model sees each
             [words](url) as its words, so that is what a quote is matched to
  unchecked  the model's own words in a sentence that also has a checked quote:
             shown dimmed, as the model's wording, not from a source
  flagged    a quote not in its passage or without a passage number, and the
             model's words in a sentence with no checked quote

Quote coverage is the share of the answer's words that are inside checked
quotes. The outcome counts, the coverage and every quote (with, when checked,
its chunk and the exact slice of the chunk's text it matched) are saved in the
answer file.

Usage (from the repo root): ./scripts/verify.py [corpus/answers/<file>.json]
With no file, checks the latest answer. Exits 1 if anything is flagged.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

CHUNKS = Path("corpus/chunks.jsonl")
UNITS = Path("corpus/units.jsonl")
ANSWERS = Path("corpus/answers")
NOT_IN_SOURCES = "Not in the sources."

# A quote in straight or curly double quotes, and the [cN] straight after it, if any.
QUOTE = re.compile(r'["“](?P<quote>[^"“”]+)["”](?:\s*\[c(?P<n>\d+)\])?')
CITATION = re.compile(r"\[c\d+\]")
LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")  # a Markdown link or image: its words, then (url)
LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:\d+[.)]|[-*+•])[ \t]+")  # "1. ", "- ": layout, not words
HELD = re.compile(r"\x00(\d+)\x00")
# Sentence breaks outside quotes: after . ! or ?, after a quote or citation when a
# new sentence starts, and at line breaks.
BREAK = re.compile(r"(?<=[.!?])\s+|(?<=[\]\x00])\s+(?=[A-Z\x00])|\n+")
WORD = re.compile(r"\w+(?:['’-]\w+)*")
DIM, PLAIN = ("\x1b[2m", "\x1b[0m") if sys.stdout.isatty() else ("", "")


def flat(text):
    return " ".join(text.split())


def reduce_links(text):
    """(text with each Markdown link or image reduced to its words, the index in
    text of each character of that, and the (start, end) of each link in text).

    The model is shown the reduced text, and quotes are found in it, so a quote
    can always be traced back to its exact slice of the original."""
    parts, where, links, last = [], [], [], 0
    for match in LINK.finditer(text):
        parts += [text[last:match.start()], match.group(1)]
        where += [*range(last, match.start()), *range(match.start(1), match.end(1))]
        links.append((match.start(), match.end()))
        last = match.end()
    parts.append(text[last:])
    where += range(last, len(text))
    return "".join(parts), where, links


def locate(quote, text):
    """The (start, end) slice of text that a quote matches, ignoring white space
    and link markup and nothing else, or None. A quote that starts or ends inside
    a link takes the whole link, so the slice never cuts through its markup."""
    target = flat(quote)
    if not target:
        return None
    reduced, where, links = reduce_links(text)
    squeezed, at = [], []  # reduced text with each run of white space as one space
    for match in re.finditer(r"\s+|\S+", reduced):
        if match.group().isspace():
            squeezed.append(" ")
            at.append(match.start())
        else:
            squeezed.append(match.group())
            at += range(match.start(), match.end())
    k = "".join(squeezed).find(target)
    if k < 0:
        return None
    start, end = where[at[k]], where[at[k + len(target) - 1]] + 1
    for link_start, link_end in links:
        if link_start <= start < link_end:
            start = link_start
        if link_start < end <= link_end:
            end = link_end
    return start, end


def words(text):
    """The words in some answer text, leaving out [cN] citations and list markers."""
    return WORD.findall(CITATION.sub(" ", LIST_MARKER.sub("", text)))


def sentences(answer):
    """(quotes, wording) for each sentence of the answer.

    quotes is [(quote, passage number or None)]. wording is the model's own
    words in the sentence, with its quotes and citations taken out, or "".
    Quotes are held aside first, so the full stops inside them do not split
    sentences.
    """
    quotes = []

    def hold(match):
        quotes.append((match.group("quote"), int(match.group("n")) if match.group("n") else None))
        return f"\x00{len(quotes) - 1}\x00"

    found = []
    for sentence in BREAK.split(QUOTE.sub(hold, LIST_MARKER.sub("", answer))):
        held = [quotes[int(i)] for i in HELD.findall(sentence)]
        wording = flat(CITATION.sub(" ", HELD.sub(" ", sentence)))
        wording = wording if re.search(r"\w", wording) else ""
        if held or wording:
            found.append((held, wording))
    return found


def find(quote, n, ids, chunks):
    """(the quote's chunk or None, the (start, end) of its slice of the chunk's text
    or None, and why it does not check, or None if it does)."""
    if n is None:
        return None, None, "the quote has no passage number"
    if not 1 <= n <= len(ids):
        return None, None, f"[c{n}] is not one of the {len(ids)} passages"
    chunk = chunks.get(ids[n - 1])
    if chunk is None:
        return None, None, f"{ids[n - 1]} is no longer in {CHUNKS}"
    span = locate(quote, chunk["text"])
    if span is None:
        return chunk, None, f"not in [c{n}] ({chunk['id']})"
    return chunk, span, None


def verify(path):
    """Print every part of a saved answer as checked, unchecked or flagged, save the
    counts and quote coverage in the answer file, and return 1 if anything is flagged."""
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    raw = CHUNKS.read_bytes()
    if record.get("chunks_sha256") and record["chunks_sha256"] != hashlib.sha256(raw).hexdigest():
        print(f"warning: {CHUNKS} has changed since this answer; checking against the current text\n")
    chunks = {c["id"]: c for c in map(json.loads, raw.decode("utf-8").splitlines())}
    units = {u["id"]: u for u in map(json.loads, UNITS.read_text(encoding="utf-8").splitlines())}
    ids, answer = record["chunks"], record["answer"]
    counts, covered, saved = {"checked": 0, "unchecked": 0, "flagged": 0}, 0, []

    print(f"Verifying {path}")
    if flat(answer) == NOT_IN_SOURCES:
        print(f'No claims: the answer is "{NOT_IN_SOURCES}"')
        coverage = None
    else:
        for quotes, wording in sentences(answer):
            found = [find(quote, n, ids, chunks) for quote, n in quotes]
            if wording:
                if any(problem is None for *_, problem in found):
                    counts["unchecked"] += 1
                    print(f"{DIM}unchecked  {wording}\n           model's wording, not from a source{PLAIN}")
                else:
                    counts["flagged"] += 1
                    reason = "the sentence quotes nothing" if not quotes else "its quote does not check"
                    print(f"flagged    {wording}\n           {reason}")
            for (quote, n), (chunk, span, problem) in zip(quotes, found):
                cite = f" [c{n}]" if n else ""
                if problem:
                    counts["flagged"] += 1
                    saved.append({"quote": quote, "passage": n, "outcome": "flagged", "why": problem})
                    print(f'flagged    "{quote}"{cite}\n           {problem}')
                    continue
                counts["checked"] += 1
                covered += len(words(quote))
                # The exact slice of the chunk's text, and of its unit's (chunk start + offset).
                saved.append({"quote": quote, "passage": n, "outcome": "checked", "chunk": chunk["id"],
                              "start": span[0], "end": span[1],
                              "unit_start": chunk["start"] + span[0], "unit_end": chunk["start"] + span[1]})
                unit = units.get(chunk["unit_id"], {})
                print(f'checked    "{quote}"{cite}\n'
                      f"           {unit.get('title', chunk['heading_path'][0])}, "
                      f"{unit.get('url') or unit.get('path', chunk['unit_id'])}\n"
                      f"           {unit.get('licence', '?')}. {unit.get('attribution', '?')}\n"
                      f"           {chunk['id']} text[{span[0]}:{span[1]}]")
        total = len(words(answer))
        coverage = round(covered / total, 3) if total else None
        print(f"\n{counts['checked']} checked, {counts['unchecked']} unchecked, {counts['flagged']} flagged. "
              f"Quote coverage {coverage:.0%} ({covered} of {total} words in checked quotes)")

    record["claims"], record["quote_coverage"], record["quotes"] = counts, coverage, saved
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if counts["flagged"] else 0


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        saved = sorted(ANSWERS.glob("*.json"))
        if not saved:
            sys.exit(f"No answers in {ANSWERS}")
        path = saved[-1]
    sys.exit(verify(path))


if __name__ == "__main__":
    main()
