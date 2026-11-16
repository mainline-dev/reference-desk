1. Add LICENSE: MIT, copyright "2026 Kipple Pty Ltd (trading as Mainline Systems)" (match the holder used in the slopbouncer and plugins repos if different).

2. Have normalise.py write SOURCES.md on every run, from sources.toml and each source's commit: name, repo or folder, licence, attribution, commit. Header text:
   "This repository's code is MIT licensed (see LICENSE). The corpus is not included: scripts fetch and clean it locally. Each source keeps its own licence. The text is adapted for search: markup is removed and links are made absolute. Anything served from the corpus must credit the source, link to the original and say it was adapted."

3. README: a short Licence section pointing to LICENSE and SOURCES.md.

4. Add to ADR 0002: every chunk served to anyone carries its attribution, licence, original URL and "adapted for search", as CC BY 4.0 requires.

