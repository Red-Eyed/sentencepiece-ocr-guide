import unicodedata

from sentencepiece_ocr_guide.checks.nfc import nfc_vocabulary
from sentencepiece_ocr_guide.checks.result import Status
from tests.checks.conftest import FakeTokenizer

COMPOSED = unicodedata.normalize("NFC", "café")
DECOMPOSED = unicodedata.normalize("NFD", "café")


def test_passes_on_a_fully_composed_vocabulary() -> None:
    tokenizer = FakeTokenizer(pieces=(COMPOSED, "a", "b"))

    assert nfc_vocabulary().run(tokenizer).status is Status.PASSED


def test_flags_a_decomposed_piece() -> None:
    tokenizer = FakeTokenizer(pieces=(DECOMPOSED, "a"))

    result = nfc_vocabulary().run(tokenizer)

    assert result.status is Status.FAILED
    assert len(result.evidence) == 1


def test_evidence_escapes_both_forms_because_they_look_identical() -> None:
    tokenizer = FakeTokenizer(pieces=(DECOMPOSED,))

    evidence = nfc_vocabulary().run(tokenizer).evidence[0]

    assert "\\u0301" in evidence  # combining acute, in the piece as stored
    assert "\\xe9" in evidence  # precomposed e-acute, what it should have been


def test_lone_combining_marks_are_not_flagged() -> None:
    """Devanagari virama and Arabic tashkeel are NFC-stable and legitimately stand alone."""
    tokenizer = FakeTokenizer(pieces=("्", "ّ", "क"))

    assert nfc_vocabulary().run(tokenizer).status is Status.PASSED
