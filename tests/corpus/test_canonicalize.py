"""The canonicalizer: what it must collapse, what it must never touch.

The preserve cases are the important half. A canonicalizer that folds too much is harder to
notice than one that folds too little, because the corpus still looks clean afterwards.
"""

import unicodedata

import pytest

from sentencepiece_ocr_guide.corpus.axes import SOFT_HYPHEN
from sentencepiece_ocr_guide.corpus.canonicalize import canonicalizer, is_canonical

canonicalize = canonicalizer()


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        ("NFD composes", unicodedata.normalize("NFD", "café"), "café"),
        ("presentation kaf", "ﻛ", "ك"),
        ("BOM stripped", "a﻿b", "ab"),
        ("zero-width space stripped", "a​b", "ab"),
        ("NBSP becomes a space", "a b", "a b"),
    ],
)
def test_collapses_encoding_differences(label: str, text: str, expected: str) -> None:
    assert canonicalize(text) == expected, label


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("fullwidth", "Ａ"),
        ("ligature", "ﬁ"),
        ("Arabic-Indic digit", "٣"),
        ("curly quote", "“"),
        ("minus sign", "−"),
        ("ZWNJ", "mi‌rood"),
        ("ZWJ", "a‍b"),
        ("ideographic space", "a　b"),
        ("Latin A", "A"),
        ("Cyrillic A", "А"),
        ("Greek Alpha", "Α"),
    ],
)
def test_preserves_character_differences(label: str, text: str) -> None:
    assert canonicalize(text) == text, label


def test_homoglyphs_stay_distinct() -> None:
    """Appearance underdetermines Unicode — context decides, so all three must survive."""
    homoglyphs = ["A", "А", "Α"]

    assert len({canonicalize(char) for char in homoglyphs}) == 3


@pytest.mark.parametrize(
    "text",
    [
        unicodedata.normalize("NFD", "café résumé"),
        "ﻛﺘﺎﺏ",
        "Ａﬁ٣−",
        "mi‌rood",
        "a﻿ b",
        "文字 100",
        r"\frac{1}{2}",
        "",
    ],
)
def test_is_idempotent(text: str) -> None:
    """What makes `line == canonicalize(line)` a valid write-time assertion."""
    once = canonicalize(text)

    assert canonicalize(once) == once


def test_bom_is_stripped_before_the_presentation_form_fold() -> None:
    """U+FEFF sits inside the Forms-B range; the wrong order would fold it instead."""
    assert canonicalize("a﻿ب") == "aب"


class TestSoftHyphen:
    """The row that proves the exception list is contextual, not universal."""

    def test_untouched_by_default(self) -> None:
        assert canonicalize(f"exam{SOFT_HYPHEN}") == f"exam{SOFT_HYPHEN}"

    def test_line_final_becomes_a_real_hyphen_when_opted_in(self) -> None:
        line_final = canonicalizer(decide=("soft_hyphen_line_final",))

        assert line_final(f"exam{SOFT_HYPHEN}") == "exam-"

    def test_mid_line_is_stripped_when_opted_in(self) -> None:
        mid_line = canonicalizer(decide=("soft_hyphen_mid_line",))

        assert mid_line(f"ex{SOFT_HYPHEN}am") == "exam"

    def test_mid_line_leaves_a_line_final_one_alone(self) -> None:
        mid_line = canonicalizer(decide=("soft_hyphen_mid_line",))

        assert mid_line(f"ex{SOFT_HYPHEN}am{SOFT_HYPHEN}") == f"exam{SOFT_HYPHEN}"


def test_unknown_decide_axis_is_an_error_not_a_silent_no_op() -> None:
    """A typo here would quietly disable a transform you believed was running."""
    with pytest.raises(ValueError, match="unknown DECIDE axes: soft_hyphen_typo"):
        canonicalizer(decide=("soft_hyphen_typo",))


def test_is_canonical_reports_the_write_time_invariant() -> None:
    assert is_canonical("café", canonicalize)
    assert not is_canonical(unicodedata.normalize("NFD", "café"), canonicalize)
