"""Tests for the pure parts of scripts/experiments.py and eval.py's comparison.
Run from the repo root: uv run python -m unittest discover -s tests"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))  # experiments.py imports chunk.py by name
stub = sys.modules.setdefault("search", types.ModuleType("search"))  # search.py loads the model; not needed here
stub.RRF_K = 60


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiments = load("experiments_under_test", "experiments.py")
evaluation = load("evaluation_for_compare", "eval.py")


class Fusion(unittest.TestCase):
    def test_equal_weights_reward_agreement(self):
        # chunk 1 is 2nd in both lists (2/62), so it beats 0 and 2, each 1st in one list only (1/61)
        self.assertEqual(experiments.fuse({0: 1, 1: 2}, {2: 1, 1: 2}), [1, 0, 2])

    def test_weighting_embeddings_two_to_one(self):
        # 0 tops the embeddings list, 2 tops BM25; with 2:1 the embeddings' first wins
        self.assertEqual(experiments.fuse({0: 1}, {2: 1}, (2, 1)), [0, 2])
        self.assertEqual(experiments.fuse({0: 1}, {2: 1}, (1, 2)), [2, 0])


class CapPerPage(unittest.TestCase):
    def test_at_most_two_passages_from_a_page(self):
        chunks = [{"unit_id": u} for u in ["a", "a", "a", "b", "a", "c"]]
        self.assertEqual(experiments.cap_per_page(list(range(6)), chunks), [0, 1, 3, 5])


def row(id_, recall8, mrr):
    return {"id": id_, "scores": {"fused": {"recall@8": recall8, "MRR": mrr}}}


class Compare(unittest.TestCase):
    def test_fixed_broken_better_worse_and_the_verdict(self):
        before = [row("q1", 0, 0.05), row("q2", 0, 0.1), row("q3", 0, 0.0), row("q4", 1, 1.0), row("q5", 0.5, 0.5)]
        after = [row("q1", 1, 1.0), row("q2", 0.5, 0.5), row("q3", 1, 0.25), row("q4", 0, 0.1), row("q5", 0.5, 0.5)]
        result = evaluation.compare(after, before)
        self.assertEqual(result["fixed"], ["q1", "q2", "q3"])
        self.assertEqual(result["broken"], ["q4"])
        self.assertEqual([c[0] for c in result["better"]], ["q1", "q2", "q3"])
        self.assertEqual([c[0] for c in result["worse"]], ["q4"])
        self.assertEqual(result["same"], 1)
        self.assertTrue(result["wins"])  # 3 fixed, 1 broken

    def test_two_broken_loses_however_many_are_fixed(self):
        before = [row(f"q{i}", 0, 0.0) for i in range(4)] + [row("q8", 1, 1.0), row("q9", 1, 1.0)]
        after = [row(f"q{i}", 1, 1.0) for i in range(4)] + [row("q8", 0, 0.0), row("q9", 0, 0.0)]
        self.assertFalse(evaluation.compare(after, before)["wins"])


if __name__ == "__main__":
    unittest.main()
