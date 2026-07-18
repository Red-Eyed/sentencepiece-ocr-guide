"""Checks that inspect what encoding produces."""

from sentencepiece_ocr_guide.checks.byte_fallback import byte_fallback_rate
from sentencepiece_ocr_guide.checks.fertility import fertility
from sentencepiece_ocr_guide.checks.protected_symbols import protected_symbols
from sentencepiece_ocr_guide.checks.result import Status
from sentencepiece_ocr_guide.checks.unknown import no_unknown
from sentencepiece_ocr_guide.checks.whitespace import no_phantom_prefix
from tests.checks.conftest import FakeTokenizer, vocabulary_for


def test_no_unknown_flags_a_character_outside_the_vocabulary() -> None:
    """Failure mode #1 — in OCR this character becomes permanently unreadable."""
    tokenizer = FakeTokenizer(pieces=vocabulary_for("abc"))

    result = no_unknown(["abc", "abcZ"]).run(tokenizer)

    assert result.status is Status.FAILED
    assert "abcZ" in result.evidence[0]


def test_no_unknown_passes_when_every_character_is_covered() -> None:
    tokenizer = FakeTokenizer(pieces=vocabulary_for("abcZ"))

    assert no_unknown(["abcZ"]).run(tokenizer).status is Status.PASSED


def test_protected_symbols_flags_a_fragmenting_command() -> None:
    """Failure mode behind docs/06 — \\frac splitting into pieces."""
    tokenizer = FakeTokenizer(pieces=(*vocabulary_for("\\frac"), "\\sqrt"))

    result = protected_symbols(["\\frac", "\\sqrt"]).run(tokenizer)

    assert result.status is Status.FAILED
    assert len(result.evidence) == 1
    assert "\\\\frac" in result.evidence[0] or "frac" in result.evidence[0]


def test_protected_symbols_skips_when_none_are_declared() -> None:
    result = protected_symbols([]).run(FakeTokenizer(pieces=("a",)))

    assert result.status is Status.SKIPPED


def test_byte_fallback_rate_measures_the_share_of_byte_tokens() -> None:
    tokenizer = FakeTokenizer(pieces=("a", "<0xE6>"), byte_pieces=("<0xE6>",))

    passing = byte_fallback_rate(["aaa"], max_rate=0.0).run(tokenizer)
    assert passing.status is Status.PASSED

    failing = byte_fallback_rate(["a<0xE6>"], max_rate=0.1).run(tokenizer)
    assert failing.status is Status.FAILED
    assert "50.00%" in failing.summary


def test_byte_fallback_rate_skips_when_samples_produce_no_tokens() -> None:
    result = byte_fallback_rate([""], max_rate=0.01).run(FakeTokenizer(pieces=("a",)))

    assert result.status is Status.SKIPPED


def test_phantom_prefix_flags_a_dummy_prefix_tokenizer() -> None:
    """Failure mode the round-trip check cannot see — decoding strips the prefix again."""
    tokenizer = FakeTokenizer(
        pieces=("▁hello", "hello"),
        normalizer=lambda text: " " + text,  # what add_dummy_prefix=True does
    )

    result = no_phantom_prefix(["hello"]).run(tokenizer)

    assert result.status is Status.FAILED
    assert "▁hello" in result.evidence[0]


def test_phantom_prefix_ignores_samples_that_really_start_with_space() -> None:
    tokenizer = FakeTokenizer(pieces=("▁hello",))

    result = no_phantom_prefix([" hello"]).run(tokenizer)

    assert result.status is Status.SKIPPED


def test_fertility_reports_tokens_per_character() -> None:
    tokenizer = FakeTokenizer(pieces=vocabulary_for("abcd"))

    result = fertility("latin", ["abcd"], max_tokens_per_char=1.0).run(tokenizer)

    assert result.status is Status.PASSED
    assert "1.00 tokens/char" in result.summary


def test_fertility_fails_when_a_script_fragments() -> None:
    character_level = FakeTokenizer(pieces=vocabulary_for("abcd"))

    result = fertility("devanagari", ["abcd"], max_tokens_per_char=0.5).run(character_level)

    assert result.status is Status.FAILED
    assert "fragmenting" in result.summary
