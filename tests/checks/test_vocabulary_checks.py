"""Checks that inspect the vocabulary rather than encodings."""

from sentencepiece_ocr_guide.checks.cross_script import cross_script_pieces
from sentencepiece_ocr_guide.checks.digit_pieces import digit_pieces
from sentencepiece_ocr_guide.checks.result import Status
from tests.checks.conftest import FakeTokenizer


def test_digit_pieces_flags_a_merged_number() -> None:
    """Failure mode #12 — the model memorizes 100 instead of reading digits."""
    tokenizer = FakeTokenizer(pieces=("1", "0", "100", "a"))

    result = digit_pieces(max_length=1).run(tokenizer)

    assert result.status is Status.FAILED
    assert "'100'" in result.evidence[0]


def test_digit_pieces_respects_a_raised_ceiling() -> None:
    tokenizer = FakeTokenizer(pieces=("1", "100"))

    assert digit_pieces(max_length=3).run(tokenizer).status is Status.PASSED


def test_digit_pieces_ignores_byte_and_control_pieces() -> None:
    """Byte pieces render as <0x31>; naive matching would read digits inside them."""
    tokenizer = FakeTokenizer(pieces=("<0x31>", "<0x32>"), byte_pieces=("<0x31>", "<0x32>"))

    assert digit_pieces(max_length=1).run(tokenizer).status is Status.PASSED


def test_cross_script_flags_a_latin_han_merge() -> None:
    """Failure mode #11 — what split_by_unicode_script is supposed to prevent."""
    tokenizer = FakeTokenizer(pieces=("a", "文", "a文"))

    result = cross_script_pieces().run(tokenizer)

    assert result.status is Status.FAILED
    assert "Han" in result.evidence[0] and "Latin" in result.evidence[0]


def test_cross_script_accepts_single_script_pieces_with_punctuation() -> None:
    """Punctuation belongs to no script and must not make ordinary pieces look mixed."""
    tokenizer = FakeTokenizer(pieces=("don't", "word.", "文。"))

    assert cross_script_pieces().run(tokenizer).status is Status.PASSED


def test_cross_script_treats_digit_letter_merges_as_configurable() -> None:
    tokenizer = FakeTokenizer(pieces=("a1",))

    assert cross_script_pieces(digits_are_a_script=True).run(tokenizer).status is Status.FAILED
    assert cross_script_pieces(digits_are_a_script=False).run(tokenizer).status is Status.PASSED
