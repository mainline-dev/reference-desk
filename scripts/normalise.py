#!/usr/bin/env -S uv run --quiet python
"""Turn every source folder in corpus/raw/ into clean Markdown in corpus/docs/,
and every page into one line of corpus/units.jsonl.

Each directory in corpus/raw/ is a source, named after the directory. It can be
a git clone or a plain folder of Markdown or text files. The script works out
what it can:

  git clone   repo URL and commit (git), content folder and site URL (Hugo
              config, if there is one)
  plain folder  nothing beyond the files themselves

Every page gets the same front matter, in this order, with empty fields left
out and every other original field dropped:

  id              <source>/<page path>, the same from run to run
  title           the original title, else the first heading, else the file name
  description     the original summary, description or headline, else the first
                  paragraph, as plain text cut at about 300 characters
  section         the first folder of the page's path, below a leading docs/
  source, url, repo, commit, path
  source_sha256   SHA-256 of the original file, so a plain folder is still traceable
  licence, attribution

units.jsonl holds one JSON object per page, sorted by id: the same fields, plus
text (the cleaned body) and sha256 (of text). SOURCES.md, at the repo root, is
rewritten on every run: each source with pages, its repository, licence,
attribution and commit.

What it cannot work out is the licence: a repo's LICENSE file often covers its
code, not its content. So every source needs an entry in sources.toml with a
licence and an attribution. A directory without one is skipped, with a warning.

Usage (from the repo root): ./scripts/normalise.py, which runs it with uv in the
project environment (it needs PyYAML), or uv run scripts/normalise.py.
"""
import fnmatch
import hashlib
import html
import json
import posixpath
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
from collections import Counter
from pathlib import Path

import yaml

RAW = Path("corpus/raw")
OUT = Path("corpus/docs")
UNITS = Path("corpus/units.jsonl")
CONFIG = Path("sources.toml")
SOURCES = Path("SOURCES.md")
SOURCES_HEADER = (
    "This repository's code is MIT licensed (see LICENSE.md). The corpus is not included: scripts fetch "
    "and clean it locally. Each source keeps its own licence. The text is adapted for search: markup is "
    "removed and links are made absolute. Anything served from the corpus must credit the source, link to "
    "the original and say it was adapted."
)
HUGO_CONFIGS = ["hugo.toml", "config.toml", "hugo/hugo.toml", "hugo/config.toml"]
DEFAULT_INCLUDE = ["**/*.md", "**/*.txt"]
DESCRIPTION_KEYS = ["summary", "description", "headline"]
DESCRIPTION_LENGTH = 300
REQUIRED = ["id", "title", "licence", "attribution"]

FRONT = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.S)
RELREF = re.compile(r"\{\{[<%]\s*relref\s+\"([^\"]*)\"\s*[>%]\}\}")
PARAM = re.compile(r"\{\{[<%]\s*param\s+\"?(\w+)\"?\s*[>%]\}\}")
RISK = re.compile(r"\{\{[<%]\s*risk-indicator\s+level=\"(\w+)\"\s*[>%]\}\}")
SCOPE = re.compile(r"\{\{[<%]\s*scope-label\s+\"([\w-]+)\"\s*[>%]\}\}")
CARD_HEADER = re.compile(r"\{\{[<%]\s*card\b[^}]*?header=\"([^\"]*)\"[^}]*[>%]\}\}")
ANY_SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.S)
CODE_SHORTCODE = re.compile(
    r"^([ \t]*)\{\{[<%]\s*(card|tab)\b([^}]*?)\s*[>%]\}\}[ \t]*\n?(.*?)\n?[ \t]*\{\{[<%]\s*/\2\s*[>%]\}\}",
    re.S | re.M,
)
TABPANE = re.compile(r"\{\{[<%]\s*tabpane\b([^}]*?)\s*[>%]\}\}(.*?)\{\{[<%]\s*/tabpane\s*[>%]\}\}", re.S)
ATTRIBUTE = re.compile(r"(\w+)=(?:\"([^\"]*)\"|(\S+))")
NOT_PROSE = re.compile(r"#{1,6}\s|[-*+]\s|\d+[.)]\s|[>|<]|!\[|\{\{|(?:[-*_]\s*){3,}$")
FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
UNDERLINE = re.compile(r" {0,3}(=+|-+)[ \t]*")
NOT_TEXT = re.compile(r" {4}|\t| {0,3}(#{1,6}(\s|$)|[-*+]\s|\d+[.)]\s|>|\||<|(?:[-*_][ \t]*){3,}$|=+[ \t]*$)")
ATX = re.compile(r" {0,3}#{1,6}(\s|$)")


# --- discovery -------------------------------------------------------------

def git(folder, *args):
    return subprocess.run(
        ["git", "-C", str(folder), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def git_info(folder):
    """(repo URL, commit) for a clone, or (None, None) for a plain folder.

    Checks for folder/.git itself, so a plain folder inside this repo is not
    mistaken for part of it.
    """
    if not (folder / ".git").exists():
        return None, None
    try:
        url = git(folder, "remote", "get-url", "origin")
        url = re.sub(r"\.git$", "", re.sub(r"^git@github\.com:", "https://github.com/", url))
    except subprocess.CalledProcessError:
        url = None  # a local repo with no remote
    return url, git(folder, "rev-parse", "HEAD")


def hugo_site(folder):
    """(site root, baseURL) from the first Hugo config found, else (None, None)."""
    for name in HUGO_CONFIGS:
        path = folder / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            match = re.search(r"^\s*baseURL\s*=\s*[\"']([^\"']+)[\"']", text, re.M | re.I)
            return path.parent, (match.group(1).rstrip("/") if match else None)
    return None, None


def content_dir(folder, site_root):
    if not site_root:
        return folder
    content = site_root / "content"
    if not content.is_dir():
        return site_root
    # Multilingual sites keep English under content/en.
    if (content / "en").is_dir() and not any(content.glob("*.md")):
        return content / "en"
    return content


# --- cleaning --------------------------------------------------------------

def url_path(rel, trailing_slash=True):
    """content-relative file path -> site path, Hugo style.

    Without a trailing slash, an index is at its folder's path with none either,
    so the top index.md is "" (base_url itself).
    """
    path = re.sub(r"\.(md|txt)$", "", rel)
    if path.endswith("_index") or path.endswith("/index") or path == "index":
        path = posixpath.dirname(path)
    path = "/" + path.strip("/") + "/" if path.strip("/") else "/"
    return path if trailing_slash else path.rstrip("/")


def text_value(meta, key):
    """A front matter value as text, or "" if it is missing or not a single value."""
    value = meta.get(key)
    if value is None or isinstance(value, (list, dict)):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def first_heading(body):
    match = re.search(r"^#{1,3}\s+(.+?)\s*$", body, re.M)
    return match.group(1) if match else ""


def atx_headings(body):
    """Setext headings (paragraph text underlined with === or ---) as # and ## headings.

    An underline counts only straight after paragraph text, outside code, as
    CommonMark reads it. So a --- after a blank line, a list item, a quote or a
    table row stays a thematic break. A heading that wraps over several lines is
    joined into one, unless it runs on from a list item or a quote.
    """
    out, fence = [], None
    for line in body.split("\n"):
        if fence:
            if re.match(rf" {{0,3}}{fence[0]}{{{len(fence)},}}[ \t]*$", line):
                fence = None
        elif FENCE.match(line):
            fence = FENCE.match(line).group(1)
        elif UNDERLINE.fullmatch(line):
            start = len(out)
            while start and out[start - 1].strip() and not NOT_TEXT.match(out[start - 1]):
                start -= 1
            before = out[start - 1] if start else ""
            if start < len(out) and (not before.strip() or ATX.match(before)):
                text = " ".join(part.strip() for part in out[start:])
                out[start:] = [("#" if "=" in line else "##") + " " + text]
                continue
        out.append(line)
    return "\n".join(out)


def attributes(text):
    return {m.group(1): m.group(3) if m.group(2) is None else m.group(2) for m in ATTRIBUTE.finditer(text)}


def code_shortcodes(body):
    """Docsy shortcodes that show code, as the header in bold and a fenced block.

    Docsy highlights a card with code=true, and a tab unless it or its tabpane
    says text=true, as code in its lang (a tab falls back to the tabpane's).
    Stripping them would leave the code as loose text, its comments read as
    headings. Anything else is left for the rules below.
    """
    def fenced(match, pane):
        indent, kind, own, inner = match.groups()
        attrs = {**pane, **attributes(own)}
        if attrs.get("text") == "true" or (kind == "card" and attrs.get("code") != "true"):
            return match.group(0)
        code = textwrap.dedent(inner.strip("\n")).rstrip()
        fence = "`" * max(3, max((len(run) for run in re.findall(r"`+", code)), default=0) + 1)
        header = re.sub(r"\A\*\*(.*)\*\*\Z", r"\1", attrs.get("header", "").strip())
        lines = ([f"**{header}**", ""] if header else []) + [fence + attrs.get("lang", ""), *code.split("\n"), fence]
        # Blank lines round it, so the header is never read as part of a paragraph.
        return "\n".join([""] + [indent + line if line else line for line in lines] + [""])

    def pane(match):
        defaults = {k: v for k, v in attributes(match.group(1)).items() if k in ("lang", "text")}
        return CODE_SHORTCODE.sub(lambda tab: fenced(tab, defaults), match.group(2))

    body = TABPANE.sub(pane, body)
    return CODE_SHORTCODE.sub(lambda m: fenced(m, {}) if m.group(2) == "card" else m.group(0), body)


def clean(body, params, page_path, base_url, trailing_slash=True):
    body = code_shortcodes(body)
    if base_url:
        def relref(match):
            target, _, anchor = match.group(1).partition("#")
            if not target.startswith("/"):
                target = posixpath.normpath(posixpath.join(page_path, "..", target))
            path = url_path(target.lstrip("/"), trailing_slash)
            return base_url + path + ("#" + anchor if anchor else "")

        body = RELREF.sub(relref, body)
        body = re.sub(r"\]\(/", f"]({base_url}/", body)  # site-relative links -> absolute
    body = PARAM.sub(lambda m: text_value(params, m.group(1)), body)
    body = RISK.sub(lambda m: f"Risk level: {m.group(1)}.", body)
    body = SCOPE.sub(lambda m: f"Scope: {m.group(1)}.", body)
    body = CARD_HEADER.sub(lambda m: m.group(1), body)
    body = ANY_SHORTCODE.sub("", body)  # closing tags, tabs, alerts, figures: keep inner text
    body = atx_headings(body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip() + "\n"


# --- front matter ----------------------------------------------------------

def plain(text):
    """Markdown or HTML as plain text on one line."""
    text = ANY_SHORTCODE.sub("", text)
    text = re.sub(r"`+([^`]*)`+", r"\1", text)  # code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links
    text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)  # reference links
    text = re.sub(r"<[^>]+>", "", text)  # HTML tags
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)  # bold
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)  # italics
    text = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"\1", text)
    text = re.sub(r"\\([!-/:-@\[-`{-~])", r"\1", text)  # backslash escapes
    return " ".join(html.unescape(text).split())


def first_paragraph(markdown):
    """The first paragraph of prose: not a heading, list, table, quote, code, HTML or image."""
    lines, fence, skipping = [], None, False
    for line in markdown.split("\n"):
        stripped = line.strip()
        if fence:
            if stripped.startswith(fence):
                fence = None
            continue
        opening = re.match(r"`{3,}|~{3,}", stripped)
        if opening:
            if lines:
                break
            fence, skipping = opening.group(0), False
        elif not stripped:
            if lines:
                break
            skipping = False
        elif lines and re.fullmatch(r"=+|-+", stripped):
            lines, skipping = [], True  # an underline: the lines above were a heading
        elif lines:
            lines.append(stripped)
        elif skipping or NOT_PROSE.match(stripped) or line.startswith(("    ", "\t")):
            skipping = True  # the rest of this block is not prose either
        else:
            lines.append(stripped)
    return " ".join(lines)


def shorten(text, limit=DESCRIPTION_LENGTH):
    """Text cut on a word boundary at about limit characters, marked … if a sentence is cut."""
    if len(text) <= limit:
        return text
    head = text[: limit + 1]
    cut = head.rsplit(" ", 1)[0] if " " in head else text[:limit]
    return cut if cut[-1] in ".!?" else cut.rstrip(" ,;:-") + "…"


def describe(meta, markdown):
    for key in DESCRIPTION_KEYS:
        text = plain(text_value(meta, key))
        if text:
            return shorten(text)
    return shorten(plain(first_paragraph(markdown)))


def section_of(rel):
    """The first folder of a content-relative path, below a leading docs/ folder."""
    folders = rel.split("/")[:-1]
    if folders[:1] == ["docs"]:
        folders = folders[1:]
    return folders[0] if folders else ""


def front_matter(fields):
    return yaml.safe_dump(fields, sort_keys=False, allow_unicode=True, width=1_000_000)


# --- main ------------------------------------------------------------------

def normalise(folder, entry):
    """Write a source's pages to corpus/docs. Returns (a summary line, its units)."""
    name = folder.name
    for key in ("licence", "attribution"):
        if not entry.get(key):
            sys.exit(f'{CONFIG}: ["{name}"] needs a {key}')

    repo, commit = git_info(folder)
    site_root, base_url = hugo_site(folder)
    base_url = entry.get("base_url", base_url)
    trailing_slash = entry.get("trailing_slash", True)
    root = folder / entry["content"] if "content" in entry else content_dir(folder, site_root)
    content_rel = root.relative_to(folder).as_posix().strip(".")
    blob = f"{repo}/blob/{commit}/{content_rel}".rstrip("/") if repo and commit else None

    include = entry.get("include", DEFAULT_INCLUDE)
    exclude = entry.get("exclude", [])
    pages = sorted({p for pattern in include for p in root.glob(pattern) if p.is_file()})

    units = []
    for page in pages:
        rel = page.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in exclude) or "/.git/" in f"/{rel}":
            continue
        raw = page.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        match = FRONT.match(text)
        front, body = (match.group(1), match.group(2)) if match else ("", text)
        try:
            meta = yaml.safe_load(front) if front.strip() else {}
        except yaml.YAMLError as error:
            sys.exit(f"{page}: front matter is not valid YAML\n{error}")
        meta = meta if isinstance(meta, dict) else {}
        if meta.get("draft") is True:
            continue

        path = url_path(rel)  # with the slash: names the output file, resolves relrefs
        if base_url:
            page_url = base_url + url_path(rel, trailing_slash)
        elif blob:
            page_url = f"{blob}/{rel}"
            body = re.sub(r"\]\(\./([\w/-]+\.md)", lambda m: f"]({blob}/{m.group(1)}", body)
        else:
            page_url = entry.get("url", "")  # plain folder: only if sources.toml gives one

        title = text_value(meta, "title").strip()
        title = title or first_heading(atx_headings(body)) or page.stem.replace("-", " ").replace("_", " ")
        markdown = clean(body, {**meta, "title": title}, path, base_url, trailing_slash)
        unit_id = f"{name}/{path.strip('/') or 'index'}"

        fields = {
            "id": unit_id,
            "title": title,
            "description": describe(meta, markdown),
            "section": section_of(rel),
            "source": name,
            "url": page_url,
            "repo": repo,
            "commit": commit,
            "path": "/".join(x for x in (content_rel, rel) if x),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "licence": entry["licence"],
            "attribution": entry["attribution"],
        }
        fields = {key: value for key, value in fields.items() if value}

        out = OUT / f"{unit_id}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"---\n{front_matter(fields)}---\n\n{markdown}", encoding="utf-8")
        sha256 = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        units.append({**fields, "text": markdown, "sha256": sha256})

    version = f"at {commit[:12]}" if commit else "(plain folder)"
    source = {"name": name, "repo": repo, "commit": commit, "pages": len(units),
              "licence": entry["licence"], "attribution": entry["attribution"]}
    return f"{name}: {len(units)} pages {version}", units, source


def write_sources(sources):
    """SOURCES.md: where the corpus comes from, under what licence, at which commit.

    Sources that contribute no pages (such as a placeholder) are left out.
    """
    cell = lambda value: str(value).replace("|", "\\|")
    lines = ["# Sources", "", SOURCES_HEADER, "",
             "| Source | Repository or folder | Licence | Attribution | Commit |",
             "|---|---|---|---|---|"]
    for s in sorted(sources, key=lambda s: s["name"]):
        if s["pages"]:
            where = s["repo"] or f"corpus/raw/{s['name']} (a plain folder)"
            lines.append(f"| {cell(s['name'])} | {cell(where)} | {cell(s['licence'])} | "
                         f"{cell(s['attribution'])} | {s['commit'] or 'n/a'} |")
    lines += ["", "Generated by scripts/normalise.py on every run, from sources.toml and each source's commit."]
    SOURCES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check(units):
    """Stop if a page lacks a required field or shares its id with another."""
    problems = []
    for unit in units:
        missing = [key for key in REQUIRED if not unit.get(key)]
        if not unit.get("url") and not unit.get("path"):
            missing.append("url or path")
        if missing:
            problems.append(f"{unit.get('id', '?')}: no {', '.join(missing)}")
    for unit_id, count in sorted(Counter(unit.get("id") for unit in units).items()):
        if count > 1:
            problems.append(f"{unit_id}: id shared by {count} pages")
    if problems:
        sys.exit("check failed:\n  " + "\n  ".join(problems))


def main():
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    if OUT.exists():
        shutil.rmtree(OUT)

    units, sources = [], []
    folders = sorted(p for p in RAW.iterdir() if p.is_dir() and not p.name.startswith("."))
    for folder in folders:
        entry = config.get(folder.name)
        if entry is None:
            print(f"skip {folder.name}: no entry in {CONFIG} (add its licence and attribution)")
            continue
        summary, found, source = normalise(folder, entry)
        print(summary)
        units.extend(found)
        sources.append(source)

    for name in sorted(set(config) - {p.name for p in folders}):
        print(f"note: {CONFIG} lists {name}, but corpus/raw/{name} is missing")

    check(units)
    units.sort(key=lambda unit: unit["id"])
    with UNITS.open("w", encoding="utf-8") as out:
        for unit in units:
            out.write(json.dumps(unit, ensure_ascii=False) + "\n")
    print(f"check: {len(units)} pages, ids unique, none missing id, title, url or path, licence or attribution")
    print(f"{UNITS}: {len(units)} units")
    write_sources(sources)
    print(f"{SOURCES}: {sum(1 for s in sources if s['pages'])} sources")

    samples = []
    for unit in units:
        if unit["source"] not in {sample["source"] for sample in samples}:
            samples.append(unit)
    for unit in samples[:3]:
        fields = {key: value for key, value in unit.items() if key not in ("text", "sha256")}
        print(f"\n---\n{front_matter(fields)}---")


if __name__ == "__main__":
    main()
