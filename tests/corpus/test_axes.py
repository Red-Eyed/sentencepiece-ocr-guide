"""Properties the scanner's fast paths depend on.

`Axis.affects` skips work in two ways: pure-ASCII lines short-circuit to False, and a line
sharing no character with an axis's `triggers` skips the transform. Both are invisible when
correct and silent when wrong — a broken trigger set does not raise, it just stops detecting.
These tests are what make the optimisation safe to keep.
"""

import unicodedata

import pytest

from sentencepiece_ocr_guide.corpus.axes import (
    DEFAULT_AXES,
    SOFT_HYPHEN,
    ZERO_WIDTH_JOINER,
    ZERO_WIDTH_NON_JOINER,
    Axis,
)

ASCII_SAMPLES = [
    "",
    " ",
    "the quick brown fox",
    "Invoice 4827193 total 1284.65",
    r"\frac{-b \pm \sqrt{b^2-4ac}}{2a}",
    "tabs\tand\nnewlines",
    "!@#$%^&*()_+-=[]{}|;':\",./<>?`~",
    "0123456789",
    "a" * 500,
]

NON_ASCII_SAMPLES = [
    unicodedata.normalize("NFD", "café résumé naïve"),
    unicodedata.normalize("NFC", "café résumé naïve"),
    "ﻛﺘﺎﺏ",
    "كتاب",
    "Ｆｕｌｌｗｉｄｔｈ　ＡＢＣ　１２３",
    "ﬁnal ﬂour oﬃce",
    "٣٤٥ ०१२",
    "“curly” and ‘single’ — dashes − too",
    f"mi{ZERO_WIDTH_NON_JOINER}rood",
    f"a{ZERO_WIDTH_JOINER}b",
    f"exam{SOFT_HYPHEN}",
    f"ex{SOFT_HYPHEN}am",
    "a b",
    "a﻿b",
    "a​b",
    "a　b",
    "光学字符识别系统",
    "日本語の文字認識",
    "प्रकाशिक वर्ण पहचान",
    "ระบบรู้จำอักขระ",
    b"bad \xff\xfe bytes".decode("utf-8", errors="surrogateescape"),
]

ALL_SAMPLES = ASCII_SAMPLES + NON_ASCII_SAMPLES


@pytest.mark.parametrize("axis", DEFAULT_AXES, ids=lambda axis: axis.name)
@pytest.mark.parametrize("text", ASCII_SAMPLES)
def test_no_axis_changes_pure_ascii(axis: Axis, text: str) -> None:
    """The invariant behind the scanner's biggest short-circuit.

    An axis that fired on ASCII would be silently skipped for every ASCII line in the corpus.
    """
    assert axis.transform(text) == text


@pytest.mark.parametrize("axis", DEFAULT_AXES, ids=lambda axis: axis.name)
@pytest.mark.parametrize("text", ALL_SAMPLES)
def test_triggers_never_cause_a_false_negative(axis: Axis, text: str) -> None:
    """`affects` must agree with running the transform, for every axis and every sample.

    This is what stops a too-narrow `triggers` set from quietly disabling detection: the fast
    path is allowed to be conservative, never wrong.
    """
    assert axis.affects(text) == (axis.transform(text) != text)


@pytest.mark.parametrize("axis", DEFAULT_AXES, ids=lambda axis: axis.name)
def test_trigger_sets_are_live(axis: Axis) -> None:
    """A trigger set must contain characters the transform actually reacts to.

    Only the conservative direction is asserted. A range-derived trigger set legitimately
    includes unassigned codepoints, and a context-sensitive transform legitimately leaves a
    trigger character alone in isolation — over-broad triggers cost speed, never accuracy.
    Characters are tested embedded in context for that reason.
    """
    if axis.triggers is None:
        pytest.skip(f"{axis.name} enumerates no trigger set")

    live = [char for char in axis.triggers if _changes_in_some_context(axis, char)]

    assert live, f"{axis.name} declares triggers but transforms none of them"


def _changes_in_some_context(axis: Axis, char: str) -> bool:
    """Position matters: the line-final soft-hyphen axis only fires at the end of a line."""
    contexts = (f"a{char}b", f"ab{char}", f"{char}ab", char)
    return any(axis.transform(context) != context for context in contexts)


@pytest.mark.parametrize("axis", DEFAULT_AXES, ids=lambda axis: axis.name)
def test_transforms_are_idempotent(axis: Axis) -> None:
    """Canonicalization composes these; a non-idempotent axis would break the write assertion."""
    for text in ALL_SAMPLES:
        once = axis.transform(text)

        assert axis.transform(once) == once


def test_surrogates_survive_every_transform_without_raising() -> None:
    """Undecodable bytes must not crash the scan — they are reported, not repaired."""
    line = b"bad \xff\xfe bytes".decode("utf-8", errors="surrogateescape")

    for axis in DEFAULT_AXES:
        axis.affects(line)  # must not raise
