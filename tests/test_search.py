"""Tests for the fused order in scripts/search.py: at most PER_PAGE passages from a page.
Run from the repo root: uv run python -m unittest discover -s tests"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).parent.parent / "scripts"

# search.py imports numpy, rank-bm25 and embed.py, which loads the model; fused_order needs none of them.
stubs = {name: types.ModuleType(name) for name in ("numpy", "rank_bm25", "embed")}
stubs["rank_bm25"].BM25Okapi = object
with mock.patch.dict(sys.modules, stubs):
    spec = importlib.util.spec_from_file_location("search_under_test", SCRIPTS / "search.py")
    search = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(search)

CHUNKS = [{"unit_id": u} for u in "aaabac"]  # chunks 0, 1, 2 and 4 are from page a
FUSED = {i: 10 - i for i in range(6)}       # best first: 0, 1, 2, 3, 4, 5


class FusedOrder(unittest.TestCase):
    def test_a_pages_third_and_later_passages_are_left_out(self):
        self.assertEqual(search.PER_PAGE, 2)
        self.assertEqual(search.fused_order(FUSED, CHUNKS, top=None), [0, 1, 3, 5])

    def test_the_top_is_counted_after_the_cap(self):
        self.assertEqual(search.fused_order(FUSED, CHUNKS, top=3), [0, 1, 3])

    def test_the_default_top_is_TOP(self):
        chunks = [{"unit_id": str(i)} for i in range(30)]
        self.assertEqual(len(search.fused_order({i: -i for i in range(30)}, chunks)), search.TOP)

    def test_ties_go_to_the_lower_index(self):
        self.assertEqual(search.fused_order({2: 1.0, 1: 1.0, 0: 1.0}, [{"unit_id": u} for u in "xyz"]), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
