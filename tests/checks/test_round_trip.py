import unicodedata

from sentencepiece_ocr_guide.checks.result import Status
from sentencepiece_ocr_guide.checks.round_trip import round_trip
from tests.checks.conftest import FakeTokenizer, vocabulary_for


def test_passes_when_every_sample_survives() -> None:
    samples = ["hello", "world"]
    tokenizer = FakeTokenizer(pieces=vocabulary_for(*samples))

    result = round_trip(samples).run(tokenizer)

    assert result.status is Status.PASSED


def test_fails_when_tokenizer_normalizes_more_than_ground_truth() -> None:
    """Failure mode #2 — an NFKC tokenizer against un-normalized labels."""
    samples = ["Ｆｕｌｌ", "ﬁnal"]
    tokenizer = FakeTokenizer(
        pieces=vocabulary_for("Full", "final"),
        normalizer=lambda text: unicodedata.normalize("NFKC", text),
    )

    result = round_trip(samples).run(tokenizer)

    assert result.status is Status.FAILED
    assert len(result.evidence) == 2
    assert "Ｆｕｌｌ" in result.evidence[0]


def test_fails_when_a_character_is_missing_from_the_vocabulary() -> None:
    tokenizer = FakeTokenizer(pieces=vocabulary_for("abc"))

    result = round_trip(["abcZ"]).run(tokenizer)

    assert result.status is Status.FAILED


def test_skips_without_samples_rather_than_passing() -> None:
    """An un-run check must never report success — that is the failure it exists to catch."""
    result = round_trip([]).run(FakeTokenizer(pieces=("a",)))

    assert result.status is Status.SKIPPED
    assert "no samples" in result.summary
