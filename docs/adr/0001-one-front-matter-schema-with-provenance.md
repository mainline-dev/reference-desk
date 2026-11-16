# One front matter schema, with provenance, for every page

Status: Accepted

## Context and Problem Statement

The corpus comes from Hugo sites, Markdown repositories and plain
folders, each with its own front matter. Search, chunking and
answers need the same fields from every page. Anything served from
the corpus has to credit its source, link to the original and keep
to the source's licence, so a page that cannot be traced to its
source, commit and licence cannot be served.

## Considered Options

- Keep each source's own front matter and map fields when reading
- A separate provenance table beside the pages
- One schema, written by normalise.py, with provenance on every page

## Decision Outcome

One schema for every page, in this order: id, title, description,
section, source, url, repo, commit, path, source_sha256, licence and
attribution. Empty fields are left out and every other original
field is dropped. The front matter is written with a YAML dumper,
and the same fields with the cleaned text go to corpus/units.jsonl.

Licence and attribution are declared for each source in
sources.toml, not read from the repository, because a repository's
licence often covers its code and not its content.

This follows the practice of documenting a dataset's provenance and
terms beside it. Gebru et al. (2021) "propose that every dataset be
accompanied with a datasheet that documents its motivation,
composition, collection process, recommended uses, and so on".

### Consequences

Every page and every chunk can be traced to the file and commit it
came from, and checked against its SHA-256. A run stops if a page
has no id, title, url or path, licence or attribution, or if two
pages share an id. Fields a source had that are not in the schema
are lost; adding one changes the schema for every source.

## References

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach,
H., Daumé III, H. and Crawford, K. (2021). Datasheets for datasets.
Communications of the ACM, 64(12), 86-92.
https://doi.org/10.1145/3458723 (also arXiv:1803.09010)
