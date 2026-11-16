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


class Tables(unittest.TestCase):
    TEXT = (
        "| Objection      | Response        |\n"
        "|----------------|:----------------|\n"
        "| Too big        | [Split it](https://example.test/split) |\n"
        "| No time        | Make time       |\n"
    )

    def test_embed_text_has_minimal_separators_and_no_cell_padding(self):
        self.assertEqual(
            chunker.embed_text(["Page"], self.TEXT),
            "Page\n\n| Objection | Response |\n|---|---|\n| Too big | Split it |\n| No time | Make time |",
        )

    def test_a_piece_starting_inside_a_table_repeats_its_header(self):
        start = self.TEXT.index("| No time")
        header = chunker.table_header(chunker.tables(self.TEXT), start)
        self.assertEqual(
            chunker.embed_text(["Page"], self.TEXT[start:], header),
            "Page\n\n| Objection | Response |\n| No time | Make time |",
        )

    def test_a_piece_starting_at_the_header_gets_no_copy(self):
        self.assertIsNone(chunker.table_header(chunker.tables(self.TEXT), 0))

    def test_rows_inside_code_are_left_alone(self):
        code = "```\n| a    | b |\n|------|---|\n```"
        self.assertEqual(chunker.embed_text(["Page"], code), "Page\n\n" + code)
        self.assertEqual(chunker.tables(code), [])


if __name__ == "__main__":
    unittest.main()
