"""Each configuration pitfall from docs/09-failure-modes.md, induced and caught.

These are the tests that give the checks their credibility: for every pitfall, the guide's
configuration passes and a single changed setting fails. They also serve as executable evidence
for the guide's claims — if a recommendation stops being load-bearing, a test here goes red.
"""

from sentencepiece_ocr_guide.checks.cross_script import cross_script_pieces
from sentencepiece_ocr_guide.checks.digit_pieces import digit_pieces
from sentencepiece_ocr_guide.checks.protected_symbols import protected_symbols
from sentencepiece_ocr_guide.checks.result import Status
from sentencepiece_ocr_guide.checks.round_trip import round_trip
from sentencepiece_ocr_guide.checks.unknown import no_unknown
from sentencepiece_ocr_guide.checks.whitespace import no_phantom_prefix
from tests.pitfalls.conftest import repeated, train

FULLWIDTH_CORPUS = repeated(
    ["Ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３", "ﬁnal ﬂour oﬃce", "plain ascii text"]
)
CJK_CORPUS = repeated(["光学字符识别系统", "日本語の文字認識", "한국어 문자 인식"])
LATIN_CORPUS = repeated(["the quick brown fox", "jumps over the lazy dog"])
LATEX_CORPUS = repeated(
    [r"\operatorname{argmax} f(x)", r"\frac{a}{b} + \sqrt{c}", r"\operatorname{tr}(A)"]
)
DIGIT_CORPUS = repeated(["Invoice 100 total 250", "Sum 100 and 250 is 350", "Page 100 of 250"])
MIXED_SCRIPT_CORPUS = repeated(["ABC文字ABC文字", "文字ABC文字ABC"])


class TestNormalizationMismatch:
    """Failure mode #2 — the tokenizer normalizes more than ground truth does."""

    samples = ["Ｆｕｌｌｗｉｄｔｈ ＡＢＣ", "ﬁnal oﬃce"]

    def test_identity_normalization_round_trips(self) -> None:
        tokenizer = train(FULLWIDTH_CORPUS)

        assert round_trip(self.samples).run(tokenizer).status is Status.PASSED

    def test_nfkc_normalization_destroys_the_labels(self) -> None:
        tokenizer = train(FULLWIDTH_CORPUS, normalization_rule_name="nmt_nfkc")

        result = round_trip(self.samples).run(tokenizer)

        assert result.status is Status.FAILED
        assert result.evidence


class TestPhantomPrefix:
    """Failure mode behind docs/04 — a spurious token at the start of every CJK label."""

    samples = ["光学字符识别系统", "日本語の文字認識"]

    def test_guide_config_leaves_cjk_lines_unprefixed(self) -> None:
        tokenizer = train(CJK_CORPUS)

        assert no_phantom_prefix(self.samples).run(tokenizer).status is Status.PASSED

    def test_add_dummy_prefix_injects_a_leading_space(self) -> None:
        tokenizer = train(CJK_CORPUS, add_dummy_prefix=True)

        result = no_phantom_prefix(self.samples).run(tokenizer)

        assert result.status is Status.FAILED
        assert len(result.evidence) == len(self.samples)

    def test_round_trip_cannot_see_this_defect(self) -> None:
        """Why the check exists separately: decoding strips the prefix, so round-trip passes."""
        tokenizer = train(CJK_CORPUS, add_dummy_prefix=True)

        assert round_trip(self.samples).run(tokenizer).status is Status.PASSED


class TestUnknownTokens:
    """Failure mode #1 — an <unk> label makes the character permanently unreadable."""

    samples = ["文字", "the quick brown fox"]

    def test_byte_fallback_covers_unseen_characters(self) -> None:
        tokenizer = train(LATIN_CORPUS)

        assert no_unknown(self.samples).run(tokenizer).status is Status.PASSED

    def test_without_byte_fallback_unseen_characters_become_unk(self) -> None:
        tokenizer = train(LATIN_CORPUS, byte_fallback=False)

        result = no_unknown(self.samples).run(tokenizer)

        assert result.status is Status.FAILED
        assert "文字" in result.evidence[0]


class TestCommandAtomicity:
    """docs/06 — a LaTeX command longer than max_sentencepiece_length can never merge.

    This is the case that makes user_defined_symbols mandatory rather than merely helpful: no
    amount of corpus frequency will produce a piece longer than the cap.
    """

    command = r"\operatorname"

    def test_declared_commands_stay_atomic(self) -> None:
        tokenizer = train(LATEX_CORPUS, user_defined_symbols=[self.command])

        assert protected_symbols([self.command]).run(tokenizer).status is Status.PASSED

    def test_undeclared_long_commands_fragment(self) -> None:
        tokenizer = train(LATEX_CORPUS)

        result = protected_symbols([self.command]).run(tokenizer)

        assert result.status is Status.FAILED
        assert "splits into" in result.evidence[0]


class TestDigitMerging:
    """Failure mode #12 — the model memorizes frequent numbers instead of reading digits."""

    def test_split_digits_keeps_numbers_at_character_level(self) -> None:
        tokenizer = train(DIGIT_CORPUS, split_digits=True)

        assert digit_pieces(max_length=1).run(tokenizer).status is Status.PASSED

    def test_without_split_digits_frequent_numbers_become_single_pieces(self) -> None:
        tokenizer = train(DIGIT_CORPUS)

        result = digit_pieces(max_length=1).run(tokenizer)

        assert result.status is Status.FAILED
        assert any("100" in item for item in result.evidence)


class TestCrossScriptMerges:
    """Failure mode #11 — a piece spanning two writing systems."""

    def test_split_by_unicode_script_blocks_the_merge(self) -> None:
        tokenizer = train(MIXED_SCRIPT_CORPUS)

        assert cross_script_pieces().run(tokenizer).status is Status.PASSED

    def test_without_it_latin_and_han_fuse(self) -> None:
        tokenizer = train(MIXED_SCRIPT_CORPUS, split_by_unicode_script=False)

        result = cross_script_pieces().run(tokenizer)

        assert result.status is Status.FAILED
        assert "Han+Latin" in result.evidence[0]
