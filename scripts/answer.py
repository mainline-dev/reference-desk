#!/usr/bin/env -S uv run --quiet python
"""Answer a question from the reference desk's passages, then verify every quote.

Runs the same fused search as search.py, puts the top PASSAGES of its top 20 in
the prompt as [c1] to [c8], asks the model (scripts/llm.py, set up in .env),
saves the answer to corpus/answers/<timestamp>.json and checks it with verify.py.

Usage (from the repo root): ./scripts/answer.py "question" [--prompt v1|v2]
The prompt version (default v1) is saved with the answer. Exits 1 if any claim is flagged.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import llm
import search
import verify

ANSWERS = Path("corpus/answers")
PASSAGES = 8  # our own rule: passages in the prompt, of search.py's fused top 20

RULES_V1 = [
    "Answer only from the passages. Use nothing you know from anywhere else.",
    "Every claim quotes one exact sentence from a passage, in double quotes, followed by the passage's "
    'number, like this: "Work integrates to trunk at least daily." [c2]',
    "Copy each quoted sentence exactly as it is in the passage.",
    f"If the passages do not answer the question, reply exactly: {verify.NOT_IN_SOURCES}",
    "When passages disagree, say so and name each source.",
]
RULES_V3 = [
    "Answer only from the passages. Use nothing you know from anywhere else.",
    'Each quote is one exact sentence copied from a passage, and "passage" is that passage\'s number, such as "c3".',
    '"summary" is optional: at most one sentence in your own words.',
    '"disagreement" is optional: which sources disagree, and on what.',
    'If the passages do not answer the question, reply exactly: {"quotes": []}',
]
INTRO = "You answer questions using only the numbered passages you are given."
JSON_FORM = ('Reply with JSON only, in this form:\n'
             '{"quotes": [{"quote": "...", "passage": "c3"}], "summary": "...", "disagreement": "..."}')

# Each version of the prompt, kept unchanged once used, so saved answers can be compared.
# v3 is structured: the model replies with JSON, and verify.py renders the answer.
RULES = {
    "v1": RULES_V1,
    "v2": RULES_V1 + ["Every sentence must contain a quote, except one sentence naming the sources when they disagree."],
    "v3": RULES_V3,
}
DEFAULT_PROMPT = "v1"


def system(version):
    form = f"{JSON_FORM}\n\n" if version in verify.STRUCTURED else ""
    return f"{INTRO}\n\n{form}Rules:\n" + "\n".join(f"- {rule}" for rule in RULES[version])


def prompt(question, passages):
    """The numbered passages, each with its title and heading path, then the question.

    Each passage is its chunk's text with links reduced to their words, the same
    text verify.py matches quotes against, so a quote never has to carry a URL."""
    parts = [f"[c{n}] {title}\nSection: {' > '.join(chunk['heading_path'])}\n\n"
             f"{verify.reduce_links(chunk['text'])[0].strip()}"
             for n, (chunk, title) in enumerate(passages, 1)]
    return "Passages:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"


def save(record):
    """Write the record to corpus/answers/<UTC timestamp>.json, never over another."""
    ANSWERS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path, n = ANSWERS / f"{stamp}.json", 1
    while path.exists():
        n += 1
        path = ANSWERS / f"{stamp}-{n}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Answer a question from the reference desk.")
    parser.add_argument("question")
    parser.add_argument("--prompt", choices=sorted(RULES), default=DEFAULT_PROMPT, help="prompt version")
    args = parser.parse_args()
    question, version = args.question, args.prompt
    config = llm.settings()  # before the search, so a missing key stops it straight away

    chunks, vectors, manifest, model = search.load()
    units = search.load_units()
    *_, fused = search.rankings(question, chunks, vectors, manifest, model)
    top = search.fused_order(fused)[:PASSAGES]
    passages = [(chunks[i], units.get(chunks[i]["unit_id"], {}).get("title", chunks[i]["heading_path"][0]))
                for i in top]

    reply, provider, reasoning = llm.chat(config, system(version), prompt(question, passages))
    path = save({
        "question": question,
        "prompt_version": version,
        "model": config["model"],
        "zdr": config["zdr"],
        "provider": provider,
        "reasoning_tokens": reasoning,
        "chunks": [chunks[i]["id"] for i in top],
        "chunks_sha256": manifest["chunks_sha256"],
        "answer": reply,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    print(f"Question: {question}\n")
    for n, (chunk, title) in enumerate(passages, 1):
        print(f"  [c{n}] {' > '.join(chunk['heading_path'])}")
    print(f"\nAnswer (prompt {version}, {config['model']}, zdr {'on' if config['zdr'] else 'off'}"
          f"{', served by ' + provider if provider else ''}, reasoning tokens {reasoning}):"
          f"\n\n{reply}\n\nSaved {path}\n")
    sys.exit(verify.verify(path))


if __name__ == "__main__":
    main()
