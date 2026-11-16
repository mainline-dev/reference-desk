"""Tests for how scripts/verify.py reads claims out of an answer.
Run from the repo root: uv run python -m unittest discover -s tests"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("verifier", Path(__file__).parent.parent / "scripts" / "verify.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class Claims(unittest.TestCase):
    def test_a_quote_and_its_number(self):
        found = verifier.claims('Teams merge daily: "Work integrates to trunk at least daily." [c2]')
        self.assertEqual([(q, n) for q, n, _ in found], [("Work integrates to trunk at least daily.", 2)])

    def test_full_stops_inside_quotes_do_not_split_claims(self):
        found = verifier.claims('"One. Two." [c1] "Three." [c3]')
        self.assertEqual([(q, n) for q, n, _ in found], [("One. Two.", 1), ("Three.", 3)])

    def test_curly_quotes_count(self):
        found = verifier.claims("“Work integrates daily.” [c4]")
        self.assertEqual([(q, n) for q, n, _ in found], [("Work integrates daily.", 4)])

    def test_a_quote_without_a_number_has_none(self):
        found = verifier.claims('"Work integrates daily."')
        self.assertEqual([(q, n) for q, n, _ in found], [("Work integrates daily.", None)])

    def test_a_sentence_that_quotes_nothing_is_a_claim_with_no_citation(self):
        found = verifier.claims('"Work integrates daily." [c1] Teams should integrate often.')
        self.assertEqual([(q, n) for q, n, _ in found], [("Work integrates daily.", 1), (None, None)])
        self.assertEqual(found[1][2], "Teams should integrate often.")

    def test_white_space_differences_are_ignored_and_nothing_else(self):
        passage = "Work integrates\nto trunk   at least daily."
        self.assertIn(verifier.flat("Work integrates to trunk at least daily."), verifier.flat(passage))
        self.assertNotIn(verifier.flat("Work integrates to trunk daily."), verifier.flat(passage))


if __name__ == "__main__":
    unittest.main()
