#!/usr/bin/env python3
"""Turn every source folder in corpus/raw/ into clean Markdown in corpus/docs/.

Each directory in corpus/raw/ is a source, named after the directory. It can be
a git clone or a plain folder of Markdown or text files. The script works out
what it can:

  git clone   repo URL and commit (git), content folder and site URL (Hugo
              config, if there is one)
  plain folder  nothing beyond the files themselves

Every page gets provenance in its front matter, including a SHA-256 of the
original file, so a plain folder is still traceable.

What it cannot work out is the licence: a repo's LICENSE file often covers its
code, not its content. So every source needs an entry in sources.toml with a
licence and an attribution. A directory without one is skipped, with a warning.

Usage (from the repo root): python3 scripts/normalise.py
Needs Python 3.11 or later (tomllib).
"""
import fnmatch
import hashlib
import posixpath
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

RAW = Path("corpus/raw")
OUT = Path("corpus/docs")
CONFIG = Path("sources.toml")
HUGO_CONFIGS = ["hugo.toml", "config.toml", "hugo/hugo.toml", "hugo/config.toml"]
DEFAULT_INCLUDE = ["**/*.md", "**/*.txt"]

FRONT = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.S)
RELREF = re.compile(r"\{\{[<%]\s*relref\s+\"([^\"]*)\"\s*[>%]\}\}")
PARAM = re.compile(r"\{\{[<%]\s*param\s+\"?(\w+)\"?\s*[>%]\}\}")
RISK = re.compile(r"\{\{[<%]\s*risk-indicator\s+level=\"(\w+)\"\s*[>%]\}\}")
SCOPE = re.compile(r"\{\{[<%]\s*scope-label\s+\"([\w-]+)\"\s*[>%]\}\}")
CARD_HEADER = re.compile(r"\{\{[<%]\s*card\b[^}]*?header=\"([^\"]*)\"[^}]*[>%]\}\}")
ANY_SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.S)


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


def front_value(front, key):
    match = re.search(rf"^{key}:\s*[\"']?(.*?)[\"']?\s*$", front, re.M)
    return match.group(1) if match else ""


def first_heading(body):
    match = re.search(r"^#{1,3}\s+(.+?)\s*$", body, re.M)
    return match.group(1) if match else ""


def clean(body, front, page_path, base_url, trailing_slash=True):
    if base_url:
        def relref(match):
            target, _, anchor = match.group(1).partition("#")
            if not target.startswith("/"):
                target = posixpath.normpath(posixpath.join(page_path, "..", target))
            path = url_path(target.lstrip("/"), trailing_slash)
            return base_url + path + ("#" + anchor if anchor else "")

        body = RELREF.sub(relref, body)
        body = re.sub(r"\]\(/", f"]({base_url}/", body)  # site-relative links -> absolute
    body = PARAM.sub(lambda m: front_value(front, m.group(1)), body)
    body = RISK.sub(lambda m: f"Risk level: {m.group(1)}.", body)
    body = SCOPE.sub(lambda m: f"Scope: {m.group(1)}.", body)
    body = CARD_HEADER.sub(lambda m: m.group(1), body)
    body = ANY_SHORTCODE.sub("", body)  # closing tags, tabs, alerts, figures: keep inner text
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip() + "\n"


def quoted(value):
    return '"' + str(value).replace('"', "'") + '"'


# --- main ------------------------------------------------------------------

def normalise(folder, entry):
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

    written = 0
    for page in pages:
        rel = page.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in exclude) or "/.git/" in f"/{rel}":
            continue
        raw = page.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        match = FRONT.match(text)
        front, body = (match.group(1), match.group(2)) if match else ("", text)
        if re.search(r"^draft:\s*true", front, re.M):
            continue

        path = url_path(rel)  # with the slash: names the output file, resolves relrefs
        if base_url:
            page_url = base_url + url_path(rel, trailing_slash)
        elif blob:
            page_url = f"{blob}/{rel}"
            body = re.sub(r"\]\(\./([\w/-]+\.md)", lambda m: f"]({blob}/{m.group(1)}", body)
        else:
            page_url = entry.get("url", "")  # plain folder: only if sources.toml gives one

        if not front_value(front, "title"):
            title = first_heading(body) or page.stem.replace("-", " ").replace("_", " ")
            front = (front + "\n" if front else "") + f"title: {quoted(title)}"

        fields = {
            "source": name,
            "source_url": page_url,
            "source_repo": repo,
            "source_commit": commit,
            "source_path": "/".join(x for x in (content_rel, rel) if x),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "licence": entry["licence"],
            "attribution": entry["attribution"],
        }
        front = "\n".join([front] + [f"{k}: {quoted(v)}" for k, v in fields.items() if v])

        out = OUT / name / f"{path.strip('/') or 'index'}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"---\n{front}\n---\n\n{clean(body, front, path, base_url, trailing_slash)}", encoding="utf-8")
        written += 1

    version = f"at {commit[:12]}" if commit else "(plain folder)"
    return f"{name}: {written} pages {version}"


def main():
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    if OUT.exists():
        shutil.rmtree(OUT)

    folders = sorted(p for p in RAW.iterdir() if p.is_dir() and not p.name.startswith("."))
    for folder in folders:
        entry = config.get(folder.name)
        if entry is None:
            print(f"skip {folder.name}: no entry in {CONFIG} (add its licence and attribution)")
            continue
        print(normalise(folder, entry))

    for name in sorted(set(config) - {p.name for p in folders}):
        print(f"note: {CONFIG} lists {name}, but corpus/raw/{name} is missing")


if __name__ == "__main__":
    main()
