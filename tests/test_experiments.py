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


def best_first(fused):
    return sorted(fused, key=lambda i: (-fused[i], i))


class Fusion(unittest.TestCase):
    def test_equal_weights_reward_agreement(self):
        # chunk 1 is 2nd in both lists (2/62), so it beats 0 and 2, each 1st in one list only (1/61)
        self.assertEqual(best_first(experiments.fuse([{0: 1, 1: 2}, {2: 1, 1: 2}])), [1, 0, 2])

    def test_weights(self):
        # 0 tops the embeddings list, 2 tops BM25; the heavier list's first wins
        self.assertEqual(best_first(experiments.fuse([{0: 1}, {2: 1}], (2, 1))), [0, 2])
        self.assertEqual(best_first(experiments.fuse([{0: 1}, {2: 1}], (1, 2))), [2, 0])

    def test_a_third_list_counts_like_the_others(self):
        self.assertEqual(best_first(experiments.fuse([{0: 1}, {1: 1}, {1: 1}])), [1, 0])

    def test_as_scores_keeps_an_order(self):
        self.assertEqual(best_first(experiments.as_scores([5, 2, 9])), [5, 2, 9])


class BM25Weight(unittest.TestCase):
    N = 1000

    def test_a_rare_term_gives_full_weight(self):
        self.assertEqual(experiments.bm25_weight(["how", "buildkit"], {"how": 600, "buildkit": 2}, self.N),
                         (1.0, "buildkit"))

    def test_only_common_terms_give_the_low_weight(self):
        self.assertEqual(experiments.bm25_weight(["how", "to", "the"], {"how": 300, "to": 900, "the": 950}, self.N),
                         (experiments.LOW, "how"))

    def test_in_between_it_rises_linearly_in_idf(self):
        idf = lambda df: experiments.idf(df, self.N)
        share = (idf(30) - idf(100)) / (idf(10) - idf(100))  # 3% of chunks, between 10% and 1%
        weight, rarest = experiments.bm25_weight(["pipeline"], {"pipeline": 30}, self.N)
        self.assertEqual(rarest, "pipeline")
        self.assertAlmostEqual(weight, experiments.LOW + (1 - experiments.LOW) * share)
        self.assertTrue(experiments.LOW < weight < 1)

    def test_terms_in_no_chunk_are_left_out(self):
        self.assertEqual(experiments.bm25_weight(["kubernetes"], {}, self.N), (experiments.LOW, None))
        self.assertEqual(experiments.bm25_weight(["kubernetes", "how"], {"how": 300}, self.N), (experiments.LOW, "how"))


class TopPages(unittest.TestCase):
    def test_the_first_passage_from_each_of_the_first_three_pages(self):
        chunks = [{"unit_id": u} for u in "aabacbd"]
        self.assertEqual(experiments.top_pages(list(range(7)), chunks), [0, 2, 4])
        self.assertEqual(experiments.top_pages([1, 0], chunks), [1])  # fewer pages than asked for


class Drift(unittest.TestCase):
    def test_only_questions_fed_back_no_answer_page_count_and_worse_is_marked(self):
        def r(id_, question, recall8, mrr):
            return {**row(id_, recall8, mrr), "question": question, "type": "plain"}
        before = [r("q1", "one", 0.5, 0.5), r("q2", "two", 0.5, 0.5), r("q3", "three", 0.5, 0.5)]
        after = [r("q1", "one", 0.5, 0.25), r("q2", "two", 1.0, 1.0), r("q3", "three", 0.0, 0.1)]
        feedback = {"one": ["x", "y"], "two": ["x"], "three": ["a", "y"]}
        grades = {"q1": {"a": 2, "x": 1}, "q2": {"a": 2}, "q3": {"a": 2}}
        found = evaluation.drift(after, before, feedback, grades)
        self.assertEqual([(d["id"], d["worse"]) for d in found], [("q1", True), ("q2", False)])  # q3 fed back a
        self.assertEqual(found[0]["pages"], [["x", 1], ["y", None]])


class WithContext(unittest.TestCase):
    def test_context_goes_between_the_heading_path_and_the_text(self):
        self.assertEqual(experiments.with_context("Title > Section\n\nFirst.\n\nSecond.", "It is about X."),
                         "Title > Section\n\nIt is about X.\n\nFirst.\n\nSecond.")


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

    def test_none_broken_and_recall_up_wins_without_any_fixed(self):
        before = [row("q1", 0.5, 0.5), row("q2", 1, 1.0)]
        after = [row("q1", 1, 0.5), row("q2", 1, 1.0)]
        result = evaluation.compare(after, before)
        self.assertEqual(result["fixed"], [])
        self.assertEqual(result["recall@8"], [0.75, 1.0])
        self.assertTrue(result["wins"])

    def test_none_broken_and_recall_level_loses(self):
        rows = [row("q1", 0.5, 0.5), row("q2", 1, 1.0)]
        self.assertFalse(evaluation.compare(rows, rows)["wins"])

    def test_one_broken_and_recall_up_loses(self):
        before = [row("q1", 0.5, 0.5), row("q2", 0.5, 0.5), row("q3", 0.5, 0.5)]
        after = [row("q1", 0, 0.0), row("q2", 1, 1.0), row("q3", 1, 1.0)]
        self.assertFalse(evaluation.compare(after, before)["wins"])


if __name__ == "__main__":
    unittest.main()
