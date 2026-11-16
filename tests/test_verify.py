"""Tests for how scripts/verify.py reads an answer into quotes and the model's own words.
Run from the repo root: uv run python -m unittest discover -s tests"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("verifier", Path(__file__).parent.parent / "scripts" / "verify.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class Sentences(unittest.TestCase):
    def test_a_quote_its_number_and_the_models_wording_around_it(self):
        self.assertEqual(
            verifier.sentences('Teams merge daily: "Work integrates to trunk at least daily." [c2]'),
            [([("Work integrates to trunk at least daily.", 2)], "Teams merge daily:")],
        )

    def test_full_stops_inside_quotes_do_not_split_sentences(self):
        self.assertEqual(
            verifier.sentences('"One. Two." [c1] "Three." [c3]'),
            [([("One. Two.", 1)], ""), ([("Three.", 3)], "")],
        )

    def test_curly_quotes_count(self):
        self.assertEqual(verifier.sentences("“Work integrates daily.” [c4]"),
                         [([("Work integrates daily.", 4)], "")])

    def test_a_quote_without_a_number_has_none(self):
        self.assertEqual(verifier.sentences('"Work integrates daily."'), [([("Work integrates daily.", None)], "")])

    def test_a_sentence_that_quotes_nothing_is_wording_alone(self):
        self.assertEqual(
            verifier.sentences('"Work integrates daily." [c1] Teams should integrate often.'),
            [([("Work integrates daily.", 1)], ""), ([], "Teams should integrate often.")],
        )

    def test_list_markers_are_layout_not_wording(self):
        self.assertEqual(
            verifier.sentences('Several:\n\n1. "Daily." [c3]\n2. "Twice daily." [c4]\n- "Hourly." [c1]'),
            [([], "Several:"), ([("Daily.", 3)], ""), ([("Twice daily.", 4)], ""), ([("Hourly.", 1)], "")],
        )

    def test_white_space_differences_are_ignored_and_nothing_else(self):
        passage = "Work integrates\nto trunk   at least daily."
        self.assertIn(verifier.flat("Work integrates to trunk at least daily."), verifier.flat(passage))
        self.assertNotIn(verifier.flat("Work integrates to trunk daily."), verifier.flat(passage))


class Links(unittest.TestCase):
    # As in the DORA continuous integration page: a link broken over lines, then "(PDF)".
    TEXT = ("on a regular basis. The [2015 State of DevOps\nReport](https://dora.dev/research/2015/report.pdf#page=20)\n"
            "(PDF) shows that teams perform better when developers merge their work into\ntrunk at least daily.")

    def test_links_reduce_to_their_words_and_keep_their_positions(self):
        reduced, where, links = verifier.reduce_links(self.TEXT)
        self.assertIn("The 2015 State of DevOps\nReport\n(PDF) shows", reduced)
        self.assertEqual([self.TEXT[i] for i in where], list(reduced))
        self.assertEqual(len(links), 1)

    def test_a_quote_of_the_reduced_text_maps_to_its_exact_slice(self):
        quote = "The 2015 State of DevOps Report (PDF) shows that teams perform better when developers merge their work into trunk at least daily."
        start, end = verifier.locate(quote, self.TEXT)
        self.assertEqual(self.TEXT[start:end], self.TEXT[self.TEXT.index("The ["):])

    def test_a_quote_starting_or_ending_inside_a_link_takes_the_whole_link(self):
        start, end = verifier.locate("2015 State of DevOps Report", self.TEXT)
        self.assertEqual(self.TEXT[start:end], "[2015 State of DevOps\nReport](https://dora.dev/research/2015/report.pdf#page=20)")

    def test_link_markup_is_the_only_thing_besides_white_space_that_is_ignored(self):
        self.assertIsNone(verifier.locate("The 2015 State of DevOps Report shows that teams perform better", self.TEXT))
        self.assertIsNone(verifier.locate("A 2015 State of DevOps Report (PDF) shows", self.TEXT))
        self.assertIsNone(verifier.locate("https://dora.dev/research/2015/report.pdf", self.TEXT))

    def test_text_without_links_matches_ignoring_white_space(self):
        start, end = verifier.locate("one  two\nthree", "zero one two three four")
        self.assertEqual((start, end), (5, 18))


class Coverage(unittest.TestCase):
    def test_words_leave_out_citations_and_list_markers(self):
        answer = '1. It says: "Work integrates to trunk daily." [c2]'
        self.assertEqual(len(verifier.words(answer)), 7)
        self.assertEqual(len(verifier.words("Work integrates to trunk daily.")), 5)


if __name__ == "__main__":
    unittest.main()
