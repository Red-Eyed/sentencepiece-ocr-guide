"""The mixed NFC/NFD corpus, and why round-trip cannot see it.

Failure mode #4 in docs/09-failure-modes.md. These tests pin down a claim that is easy to get
backwards: `identity` does not *cause* this failure, it *propagates* one already present in the
corpus — and it does so while every round-trip assertion still passes.
"""

import unicodedata

from sentencepiece_ocr_guide.checks.nfc import nfc_vocabulary
from sentencepiece_ocr_guide.checks.result import Status
from sentencepiece_ocr_guide.checks.round_trip import round_trip
from tests.pitfalls.conftest import repeated, train

COMPOSED = unicodedata.normalize("NFC", "café résumé naïve")
DECOMPOSED = unicodedata.normalize("NFD", "café résumé naïve")

MIXED_CORPUS = repeated([COMPOSED, DECOMPOSED, "plain ascii line"])
CLEAN_CORPUS = repeated([COMPOSED, "plain ascii line"])


def test_the_two_forms_render_alike_but_are_different_text() -> None:
    assert COMPOSED != DECOMPOSED
    assert len(COMPOSED) < len(DECOMPOSED)


class TestMixedCorpus:
    def test_identity_gives_the_same_text_two_different_encodings(self) -> None:
        """The damage: one image, two possible label sequences, signal split between them."""
        tokenizer = train(MIXED_CORPUS)

        assert tokenizer.encode(COMPOSED) != tokenizer.encode(DECOMPOSED)

    def test_round_trip_is_blind_to_it(self) -> None:
        """Both forms survive encode/decode, so the guide's highest-value check stays green."""
        tokenizer = train(MIXED_CORPUS)

        result = round_trip([COMPOSED, DECOMPOSED]).run(tokenizer)

        assert result.status is Status.PASSED

    def test_the_vocabulary_check_catches_it(self) -> None:
        tokenizer = train(MIXED_CORPUS)

        result = nfc_vocabulary().run(tokenizer)

        assert result.status is Status.FAILED
        assert any("\\u0301" in item for item in result.evidence)

    def test_an_nfc_corpus_is_clean(self) -> None:
        tokenizer = train(CLEAN_CORPUS)

        assert nfc_vocabulary().run(tokenizer).status is Status.PASSED


class TestNfkcIsTheWrongFix:
    """NFKC does unify the two forms — by destroying distinctions OCR needs."""

    def test_nfkc_unifies_the_encodings(self) -> None:
        tokenizer = train(MIXED_CORPUS, normalization_rule_name="nmt_nfkc")

        assert tokenizer.encode(COMPOSED) == tokenizer.encode(DECOMPOSED)

    def test_but_it_folds_fullwidth_and_ligatures(self) -> None:
        """The cost: visual evidence present in the image is erased from the label space."""
        corpus = repeated(["Ｆｕｌｌｗｉｄｔｈ ＡＢＣ", "ﬁnal oﬃce", "plain ascii"])
        tokenizer = train(corpus, normalization_rule_name="nmt_nfkc")

        result = round_trip(["Ｆｕｌｌｗｉｄｔｈ ＡＢＣ", "ﬁnal oﬃce"]).run(tokenizer)

        assert result.status is Status.FAILED

    def test_nfc_at_ingestion_is_lossless_where_nfkc_is_not(self) -> None:
        """Why the fix belongs in the corpus: NFC preserves what NFKC folds."""
        fullwidth = "Ｆｕｌｌ ＡＢＣ"
        ligature = "ﬁnal"

        assert unicodedata.normalize("NFC", fullwidth) == fullwidth
        assert unicodedata.normalize("NFC", ligature) == ligature
        assert unicodedata.normalize("NFKC", fullwidth) != fullwidth
        assert unicodedata.normalize("NFKC", ligature) != ligature
