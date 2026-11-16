"""Tests for the retrieval metrics in scripts/eval.py, on a small case worked by hand.
Run from the repo root: uv run python -m unittest discover -s tests"""
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

# eval.py imports search.py, which loads the embedding model; the metrics need none of it.
sys.modules.setdefault("search", types.ModuleType("search"))
spec = importlib.util.spec_from_file_location("evaluation", Path(__file__).parent.parent / "scripts" / "eval.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

# Ten chunks from five pages. The search ranked them in this order.
CHUNKS = [{"unit_id": u} for u in ["x", "a", "x", "b", "y", "a", "z", "c", "y", "w"]]
ORDER = list(range(10))


class Metrics(unittest.TestCase):
    def test_pages_rank_by_their_best_chunk(self):
        self.assertEqual(evaluation.page_order(ORDER, CHUNKS), {"x": 1, "a": 2, "b": 3, "y": 4, "z": 5, "c": 6, "w": 7})

    def test_recall_mrr_and_ndcg(self):
        grades = {"a": 2, "c": 2, "z": 1}  # pages a and c answer; z is related
        got = evaluation.scores(ORDER, CHUNKS, grades)
        self.assertEqual(got["recall@8"], 1.0)  # a (chunk 2) and c (chunk 8) are both in the top 8 chunks
        self.assertEqual(evaluation.scores(ORDER[:5], CHUNKS, grades)["recall@8"], 0.5)  # only a, if c is not retrieved
        self.assertEqual(got["MRR"], 1 / 2)  # a is the 2nd page
        dcg = 3 / math.log2(3) + 1 / math.log2(6) + 3 / math.log2(7)  # a at 2, z at 5, c at 6
        ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(got["nDCG@10"], dcg / ideal)

    def test_best_chunk_rank_is_the_first_chunk_from_a_grade_2_page(self):
        self.assertEqual(evaluation.best_chunk_rank(ORDER, CHUNKS, {"c": 2, "z": 1}), 8)
        self.assertIsNone(evaluation.best_chunk_rank(ORDER, CHUNKS, {"missing": 2, "a": 1}))

    def test_a_question_whose_pages_are_never_found_scores_zero(self):
        got = evaluation.scores(ORDER, CHUNKS, {"missing": 2})
        self.assertEqual((got["recall@8"], got["recall@20"], got["MRR"], got["nDCG@10"]), (0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
