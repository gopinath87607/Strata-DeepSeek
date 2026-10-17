"""Tests for the DeepSeek-V4 `joyai-llm` pre-tokenizer and the control-token table its GGUF lacks.

    python -m unittest tools.test_tokenizer_joyai -v

The split tests and the type synthesis run on synthetic vocabularies and always run.  The golden vectors
and the differential check need the artifact's tokenizer metadata: `reference/ds4-tokenizer.gguf` (a
metadata-only GGUF written from the first megabytes of the real file by
`tools/dsv4/make_tokenizer_gguf.py`), or `$DS4_TOKENIZER_GGUF`.  The differential additionally needs the
reference C oracle at `$DS4_TOK_ORACLE` (built from `reference/ds4-dfm-rs/tests/parity/
tokenizer_c_oracle.c`); it is skipped when that path is unset.

The goldens were produced by that oracle, not by this module: `encode` is ordinary byte-level BPE and
`render` is the same text with the engine's special-token scan.  They pin the two behaviours that a
plausible-looking implementation gets wrong: a run of spaces before CJK, and the engine's rule that every
non-ASCII byte counts as a letter.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import strata_tokenizer as ST  # noqa: E402

GGUF = Path(os.environ.get("DS4_TOKENIZER_GGUF", ROOT / "reference" / "ds4-tokenizer.gguf"))
ORACLE = os.environ.get("DS4_TOK_ORACLE")


def byte_tokenizer(pre: str = "joyai-llm") -> ST.Tokenizer:
    """The 256 byte tokens plus a handful of literals, no merges: enough to test dispatch and specials."""
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    return ST.Tokenizer(tokens, [], [1] * 256, pre)


class Split(unittest.TestCase):
    def check(self, text, pieces):
        self.assertEqual(ST.joyai_split(text), pieces)

    def test_digits_are_groups_of_three_then_the_rest(self):
        self.check("1234567890", ["123", "456", "789", "0"])

    def test_a_space_joins_the_letters_after_it(self):
        self.check("a b", ["a", " b"])
        self.check("a  b", ["a", " ", " b"])

    def test_the_last_space_of_a_run_goes_with_cjk(self):
        # The engine's rule that neither regex reading of the llama.cpp triplet reproduces: pattern 2 would
        # cut 漢字 out first (sequential), and a single alternation would merge the space into 漢字.
        self.check(" 漢字", [" 漢字"])
        self.check("   漢字", ["  ", " 漢字"])
        self.check(" " * 7 + "漢字", [" " * 6, " 漢字"])

    def test_whitespace_runs_end_at_the_last_newline(self):
        self.check("a\n\n\nb", ["a", "\n\n\n", "b"])
        self.check("  \n  x", ["  \n", " ", " x"])

    def test_punctuation_before_letters_and_after_a_space(self):
        self.check("Hello, world!", ["Hello", ",", " world", "!"])
        self.check("(x)", ["(x", ")"])      # `[punct][A-Za-z]+` is one piece

    def test_every_non_ascii_byte_is_a_letter_but_cjk_is_isolated(self):
        self.check("😀é", ["😀é"])          # documented approximation, not \p{L}
        self.check("漢é", ["漢", "é"])

    def test_nothing_is_dropped(self):
        for s in ["", " ", "\x00\x01", "a\tb", "（）12…", "é" * 40, "x" * 200]:
            self.assertEqual("".join(ST.joyai_split(s)), s)


class Types(unittest.TestCase):
    def test_the_deepseek_control_literals_are_marked_and_others_are_not(self):
        tokens = ["a", "<｜User｜>", "<think>", "<|endoftext|>", "<｜System｜>", "b"]
        types = ST.family_control_types("joyai-llm", tokens)
        self.assertEqual([types[i] for i in (0, 1, 2, 3, 4, 5)], [1, 3, 3, 1, 3, 1])

    def test_other_pres_leave_the_metadata_path_alone(self):
        self.assertIsNone(ST.family_control_types("qwen35", ["a"]))


class Dispatch(unittest.TestCase):
    def test_an_unknown_pre_is_refused_rather_than_silently_qwen(self):
        with self.assertRaises(ValueError):
            byte_tokenizer("llama-bpe")

    def test_the_split_decides_where_merges_may_apply(self):
        # `Ġ Ġ` merges two spaces only when the pre-tokenizer put them in one piece.  With three spaces the
        # engine splits "  " + " b", so the merge fires on the first two; the third stays in " b".
        tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + ["ĠĠ"]
        tok = ST.Tokenizer(tokens, ["Ġ Ġ"], [1] * len(tokens), "joyai-llm")
        self.assertEqual(tok.encode("a  b"), [97, 32, 32, 98])
        self.assertEqual(tok.encode("a   b"), [97, 256, 32, 98])

    def test_the_ascii_marker_aliases_map_to_bos_and_eos(self):
        tokens = ["x", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"]
        tok = ST.Tokenizer(tokens, [], [1, 3, 3], "joyai-llm")
        self.assertEqual(tok.encode("<|endoftext|>", parse_special=True), [2])
        self.assertEqual(tok.encode("<|beginoftext|>", parse_special=True), [1])
        self.assertEqual(tok.encode("<|startoftext|>", parse_special=True), [1])


GOLDEN_ENCODE = [
    ("Hello, world!", [19923, 14, 2058, 3]),
    ("   漢字", [262, 223, 29069, 2024]),
    (" 漢字", [223, 29069, 2024]),
    ("a  b", [67, 223, 291]),
    ("a b", [67, 291]),
    ("1234567890", [6895, 18009, 25744, 18]),
    ("😀é", [28927, 225, 619]),
    ("（）12…", [32336, 736, 1248]),
    ("def f(x):\n\treturn x  # comment\n", [3465, 285, 4042, 3395, 5562, 1527, 223, 1823, 7006, 201]),
    ("é́ combining", [71, 17793, 17793, 26179]),
]

GOLDEN_RENDER = [
    ("<｜begin▁of▁sentence｜><｜User｜>hi</think> 漢字", [0, 128803, 6366, 128822, 223, 29069, 2024]),
    ("<|endoftext|>", [1]),
]


@unittest.skipUnless(GGUF.is_file(), "no DeepSeek tokenizer GGUF (reference/ds4-tokenizer.gguf or $DS4_TOKENIZER_GGUF)")
class Real(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ST.Tokenizer.from_gguf(GGUF)

    def test_the_artifact_declares_joyai_and_carries_no_token_types(self):
        self.assertEqual(self.tok.pre, "joyai-llm")
        self.assertEqual(len(self.tok.tokens), 129280)
        names = sorted(self.tok.special_tokens)
        self.assertEqual(self.tok.special_tokens["<｜User｜>"], 128803)
        self.assertIn("<｜System｜>", names)         # the V4.1 token, absent from the C checkout's table
        self.assertIn("<|endoftext|>", names)       # ASCII alias, not a vocabulary token

    def test_encode_goldens(self):
        for text, ids in GOLDEN_ENCODE:
            self.assertEqual(self.tok.encode(text), ids, text)

    def test_render_goldens(self):
        for text, ids in GOLDEN_RENDER:
            self.assertEqual(self.tok.encode(text, parse_special=True), ids, text)

    def test_special_tokens_decode_as_their_literal(self):
        self.assertEqual(self.tok.decode([128803]), "<｜User｜>")
        self.assertEqual(self.tok.decode([0]), "<｜begin▁of▁sentence｜>")

    @unittest.skipUnless(ORACLE, "set $DS4_TOK_ORACLE to the reference tokenizer oracle")
    def test_against_the_reference_oracle(self):
        cases = [t for t, _ in GOLDEN_ENCODE] + [t for t, _ in GOLDEN_RENDER] + [
            "\x01\x7f control", "café naïve", "مرحبا", "한국어 텍스트", "ひらがな カタカナ", "x" * 500,
            "     漢字", "  \t lead", "end   ", " non-breaking space", "😀 漢 x",
        ]
        for text in cases:
            for cmd, parse in (("encode", False), ("render", True)):
                out = subprocess.run([ORACLE, "deepseek4", str(GGUF), cmd, text.encode().hex()],
                                     check=True, capture_output=True, text=True).stdout.split()
                want = [int(x) for x in out[1:]]
                got = self.tok.encode(text, parse_special=parse)
                if cmd == "render" and "<｜System｜>" in text:
                    continue    # the C source in this checkout predates the engine's <｜System｜> row
                self.assertEqual(got, want, "%s %r" % (cmd, text))


if __name__ == "__main__":
    unittest.main()
