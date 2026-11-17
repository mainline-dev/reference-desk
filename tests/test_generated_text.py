"""Generated text (scripts/expand.py) is for matching only: it must never reach
answer.py's passages or the text the verifier checks quotes against.
Run from the repo root: uv run python -m unittest discover -s tests"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).parent.parent / "scripts"


def load(name, file, stub_names=()):
    stubs = {n: types.ModuleType(n) for n in stub_names}
    with mock.patch.dict(sys.modules, stubs), mock.patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


# answer.py imports llm.py (the model client) and search.py (the embedding model); the prompt needs neither.
answer = load("answer_under_test", "answer.py", ("llm", "search"))
verifier = load("verifier_for_generated_text", "verify.py")
expand = load("expand_under_test", "expand.py")
experiments = load("experiments_for_generated_text", "experiments.py", ("search",))

TEXT = "Work integrates to trunk at least daily."
CONTEXT = "GENERATED: this passage explains how often work reaches trunk."
QUESTIONS = ["GENERATED: how often should we merge?", "GENERATED: is merging once a week enough?"]
CHUNK = {
    "id": "minimumcd/trunk#0",
    "unit_id": "minimumcd/trunk",
    "text": TEXT,
    "heading_path": ["Trunk-Based Development", "Integration"],
    # What a variant indexes: context and questions added to embed_text, as experiments.py builds it.
    "embed_text": experiments.with_context("Trunk-Based Development > Integration\n\n" + TEXT, CONTEXT)
                  + "\n\n" + "\n".join(QUESTIONS),
    # And an expansion record's fields copied onto the chunk, in case code ever merges them.
    "questions": QUESTIONS,
    "context": CONTEXT,
}


class GeneratedTextStaysOutOfAnswers(unittest.TestCase):
    def test_the_prompt_holds_the_chunk_text_and_nothing_generated(self):
        prompt = answer.prompt("How often should we integrate?", [(CHUNK, "Trunk-Based Development")])
        self.assertIn(TEXT, prompt)
        self.assertNotIn("GENERATED", prompt)

    def test_the_verifier_checks_quotes_against_the_text_only(self):
        ids, chunks = [CHUNK["id"]], {CHUNK["id"]: CHUNK}
        self.assertIsNone(verifier.find(TEXT, 1, ids, chunks)[2])
        for generated in [CONTEXT, *QUESTIONS]:
            self.assertIsNotNone(verifier.find(generated, 1, ids, chunks)[2], generated)

    def test_no_default_script_reads_the_expansions_or_image_descriptions(self):
        # Only expand.py writes them and only experiments.py reads them; search, answers and
        # verification never open them.
        for name in ("answer.py", "verify.py", "search.py", "chunk.py", "embed.py", "normalise.py", "llm.py"):
            source = (SCRIPTS / name).read_text(encoding="utf-8").lower()
            for word in ("expansion", "image-description", "image_description"):
                self.assertNotIn(word, source, name)

    def test_an_image_description_stays_out_of_the_prompt(self):
        chunk = {**CHUNK, "text": TEXT + "\n\n![image](ta-image1.png)",
                 "description": "GENERATED: a bar chart of test automation by team."}
        prompt = answer.prompt("How often should we integrate?", [(chunk, "Trunk-Based Development")])
        self.assertIn(TEXT, prompt)
        self.assertNotIn("GENERATED", prompt)

    def test_with_context_leaves_the_chunk_text_alone(self):
        chunk = dict(CHUNK)
        experiments.with_context(chunk["embed_text"], CONTEXT)
        self.assertEqual(chunk["text"], TEXT)


class ExpansionReplies(unittest.TestCase):
    GOOD = '{"questions": ["a?", "b?", "c?", "d?", "e?"], "context": "It is  about trunk."}'

    def test_a_good_reply(self):
        self.assertEqual(expand.parse(self.GOOD), (["a?", "b?", "c?", "d?", "e?"], "It is about trunk."))

    def test_bad_replies_say_what_is_wrong(self):
        for reply, problem in [
            ("not json", "not JSON"),
            ('["a?"]', "not a JSON object"),
            ('{"questions": ["a?", "b?", "c?", "d?"], "context": "x"}', "4 questions"),
            ('{"questions": ["a?", "A?", "c?", "d?", "e?"], "context": "x"}', "repeat"),
            ('{"questions": ["a?", "b?", "c?", "d?", "e?"], "context": " "}', "no context"),
            ('{"questions": ["a?", "b?", "c?", "d?", "e?"], "context": "' + "word " * 81 + '"}', "context is over"),
        ]:
            with self.assertRaises(ValueError, msg=reply) as caught:
                expand.parse(reply)
            self.assertIn(problem, str(caught.exception))

    def test_the_prompt_puts_the_page_before_the_passage_with_links_reduced(self):
        unit = {"title": "Trunk", "text": "Intro. See [the guide](https://example.org/guide).\n\n" + TEXT}
        prompt = expand.user_prompt(unit, CHUNK)
        self.assertIn("See the guide.", prompt)
        self.assertNotIn("https://", prompt)
        self.assertLess(prompt.index("Intro."), prompt.index('section "Trunk-Based Development > Integration"'))

    def test_a_record_is_current_only_for_the_same_text_model_and_prompt(self):
        record = {"text_sha256": expand.text_sha256(TEXT), "model": "m", "prompt_version": expand.PROMPT_VERSION}
        self.assertTrue(expand.current(record, CHUNK, "m"))
        self.assertFalse(expand.current(record, CHUNK, "other"))
        self.assertFalse(expand.current({**record, "prompt_version": "e0"}, CHUNK, "m"))
        self.assertFalse(expand.current(record, {**CHUNK, "text": TEXT + " Changed."}, "m"))
        self.assertFalse(expand.current(None, CHUNK, "m"))
        self.assertEqual(expand.text_sha256(TEXT), experiments.text_sha256(TEXT))


class Images(unittest.TestCase):
    PAGE = "\n".join([
        "Intro ![image](one.png) and ![](two.png) and ![A chart of lead time](three.png).",
        '<img src="four.svg" alt="Strangler fig"> <img src=\'five.png\'>',
        "```mermaid",
        "graph LR; A-->B",
        "![image](in-mermaid.png)",
        "```",
        "````",
        "```",
        "![image](in-a-longer-fence.png)",
        "````",
        "After ![Image](six.png)",
    ])

    def test_images_outside_code_fences_are_found_and_mermaid_is_left_alone(self):
        self.assertEqual(expand.find_images(self.PAGE), [
            ("one.png", "image"), ("two.png", ""), ("three.png", "A chart of lead time"),
            ("four.svg", "Strangler fig"), ("five.png", ""), ("six.png", "Image")])

    def test_only_empty_or_image_alt_text_needs_a_description(self):
        needing = [src for src, alt in expand.find_images(self.PAGE) if expand.needs_description(alt)]
        self.assertEqual(needing, ["one.png", "two.png", "five.png", "six.png"])

    def test_the_file_comes_from_the_checkout_and_nowhere_else(self):
        unit = {"source": "dora.dev", "path": "hugo/content/capabilities/wip-limits/index.md"}
        root = (expand.RAW / "dora.dev").resolve()
        self.assertEqual(expand.image_file(unit, "wip-1.png"),
                         root / "hugo/content/capabilities/wip-limits/wip-1.png")
        self.assertIsNone(expand.image_file(unit, "https://example.org/x.png"))
        self.assertIsNone(expand.image_file(unit, "/images/x.png"))
        self.assertIsNone(expand.image_file(unit, "../../../../../../etc/passwd"))

    def test_a_description_is_one_paragraph(self):
        self.assertEqual(expand.parse_description("A bar chart.\n\nIt shows WIP."), "A bar chart. It shows WIP.")
        for bad in ["", "  \n ", "word " * 201]:
            with self.assertRaises(ValueError):
                expand.parse_description(bad)

    def test_a_description_is_current_only_for_the_same_image_model_and_prompt(self):
        record = {"image_sha256": "abc", "model": "m", "prompt_version": expand.IMAGE_PROMPT_VERSION}
        self.assertTrue(expand.current_image(record, "abc", "m"))
        self.assertFalse(expand.current_image(record, "def", "m"))
        self.assertFalse(expand.current_image({**record, "prompt_version": "i0"}, "abc", "m"))


if __name__ == "__main__":
    unittest.main()
