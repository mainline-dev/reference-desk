"""Tests for scripts/chunk.py. Run from the repo root: uv run python -m unittest discover -s tests"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("chunker", Path(__file__).parent.parent / "scripts" / "chunk.py")
chunker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunker)


class Fences(unittest.TestCase):
    # A four-backtick block holding a three-backtick block, as a Markdown sample
    # that shows a code block would. The ``` lines must not close the outer fence.
    TEXT = (
        "Intro.\n"
        "\n"
        "````markdown\n"
        "```python\n"
        "# not a heading\n"
        "```\n"
        "\n"
        "## Still code\n"
        "````\n"
        "\n"
        "## Real heading\n"
        "\n"
        "Body.\n"
    )

    def test_headings_inside_the_block_are_not_sections(self):
        paths = [path for path, _, _ in chunker.sections(self.TEXT, "Page")]
        self.assertEqual(paths, [["Page"], ["Page", "Real heading"]])

    def test_the_block_stays_one_block_across_its_blank_line(self):
        spans = chunker.blocks(self.TEXT, 0, len(self.TEXT))
        code = [self.TEXT[s:e] for s, e in spans if self.TEXT[s:e].startswith("````")]
        self.assertEqual(code, ["````markdown\n```python\n# not a heading\n```\n\n## Still code\n````\n"])

    def test_a_longer_run_closes_and_a_tilde_run_does_not(self):
        self.assertIsNone(chunker.fence_after("`````\n", "````"))
        self.assertEqual(chunker.fence_after("~~~~\n", "````"), "````")
        self.assertEqual(chunker.fence_after("```` not a close\n", "````"), "````")


if __name__ == "__main__":
    unittest.main()
