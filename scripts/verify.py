#!/usr/bin/env -S uv run --quiet python
"""Check a saved answer's quotes against the passages they cite.

Each claim should quote one exact sentence in double quotes followed by its
passage number, like "Work integrates to trunk at least daily." [c2]. A quote
is checked if it appears in that chunk's text (not its embed_text), ignoring
only differences in white space. Anything else is flagged: a quote that is not
in its passage, a quote without a number, and a sentence that quotes nothing.

Usage (from the repo root): ./scripts/verify.py [corpus/answers/<file>.json]
With no file, checks the latest answer. Exits 1 if any claim is flagged.
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
HELD = re.compile(r"\x00(\d+)\x00")
# Sentence breaks outside quotes: after . ! or ?, after a citation when a new
# sentence starts, and at line breaks.
BREAK = re.compile(r"(?<=[.!?])\s+|(?<=[\]\x00])\s+(?=[A-Z\x00])|\n+")


def flat(text):
    return " ".join(text.split())


def claims(answer):
    """(quote or None, its passage number or None, the sentence) for each claim.

    Quotes are held aside first, so the full stops inside them do not split
    sentences. A sentence with no quote is a claim with no citation.
    """
    quotes = []

    def hold(match):
        quotes.append((match.group("quote"), int(match.group("n")) if match.group("n") else None))
        return f"\x00{len(quotes) - 1}\x00"

    found = []
    for sentence in BREAK.split(QUOTE.sub(hold, answer)):
        held = [quotes[int(i)] for i in HELD.findall(sentence)]
        if held:
            found += [(quote, n, sentence) for quote, n in held]
        elif re.search(r"\w", re.sub(r"\[c\d+\]", "", sentence)):
            shown = HELD.sub(lambda m: f'"{quotes[int(m.group(1))][0]}"', sentence)
            found.append((None, None, shown.strip()))
    return found


def verify(path):
    """Print each claim in a saved answer as checked or flagged. Returns 0, or 1 if any is flagged."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = CHUNKS.read_bytes()
    if record.get("chunks_sha256") and record["chunks_sha256"] != hashlib.sha256(raw).hexdigest():
        print(f"warning: {CHUNKS} has changed since this answer; checking against the current text\n")
    chunks = {c["id"]: c for c in map(json.loads, raw.decode("utf-8").splitlines())}
    units = {u["id"]: u for u in map(json.loads, UNITS.read_text(encoding="utf-8").splitlines())}
    ids, answer = record["chunks"], record["answer"]

    print(f"Verifying {path}")
    if flat(answer) == NOT_IN_SOURCES:
        print(f'No claims: the answer is "{NOT_IN_SOURCES}"')
        return 0

    checked = flagged = 0
    for quote, n, sentence in claims(answer):
        if quote is None:
            flagged += 1
            print(f"flagged  {sentence}\n         no quote with a passage number")
            continue
        cite = f"[c{n}]" if n else "(no number)"
        print(f'{"":9}"{quote}" {cite}')
        chunk = chunks.get(ids[n - 1]) if n and 1 <= n <= len(ids) else None
        if n is None:
            problem = "the quote has no passage number"
        elif chunk is None:
            problem = f"[c{n}] is not one of the {len(ids)} passages" if not 1 <= n <= len(ids) else \
                f"{ids[n - 1]} is no longer in {CHUNKS}"
        elif flat(quote) not in flat(chunk["text"]):
            problem = f"not in [c{n}] ({chunk['id']})"
        else:
            unit = units.get(chunk["unit_id"], {})
            checked += 1
            print(f"checked  {unit.get('title', chunk['heading_path'][0])}, "
                  f"{unit.get('url') or unit.get('path', chunk['unit_id'])}\n"
                  f"         {unit.get('licence', '?')}. {unit.get('attribution', '?')}")
            continue
        flagged += 1
        print(f"flagged  {problem}")

    print(f"\n{checked + flagged} claims: {checked} checked, {flagged} flagged")
    return 1 if flagged else 0


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
