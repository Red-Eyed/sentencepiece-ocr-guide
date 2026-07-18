"""The axes on which two sources can disagree about how to encode the same text.

An axis is a named, pure `str -> str` transform plus a verdict on what should happen to it. A
line is *affected* by an axis when the transform changes it, which is all the scanner needs; the
canonicalizer reuses the same transforms for the subset it is allowed to apply.

The verdict is what keeps the two uses honest. `PRESERVE` axes carry a transform purely so they
can be *detected* — `canonicalize` filters on `Action` and can never fold one by accident. See
docs/09-failure-modes.md for the reasoning behind each verdict.
"""

import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

SOFT_HYPHEN = "­"
ZERO_WIDTH_NON_JOINER = "‌"
ZERO_WIDTH_JOINER = "‍"


class Action(StrEnum):
    """What should be done about text that varies along an axis."""

    COLLAPSE = "collapse"
    """Not a difference in the text. `canonicalize` folds these."""

    PRESERVE = "preserve"
    """A genuine character difference that must survive. Reported, never folded."""

    DECIDE = "decide"
    """Depends on your sources and rendering context. Opt in explicitly, after measuring."""


@dataclass(frozen=True)
class Axis:
    """One way two encodings of the same text can differ."""

    name: str
    action: Action
    rationale: str
    transform: Callable[[str], str]

    def affects(self, line: str) -> bool:
        return self.transform(line) != line


def _strip(*codepoints: int) -> Callable[[str], str]:
    table: dict[int, str | None] = {codepoint: None for codepoint in codepoints}

    def transform(text: str) -> str:
        return text.translate(table)

    return transform


def _replace(mapping: Mapping[int, str]) -> Callable[[str], str]:
    table: dict[int, str | None] = dict(mapping)

    def transform(text: str) -> str:
        return text.translate(table)

    return transform


def _fold_ranges(*ranges: range) -> Callable[[str], str]:
    """NFKC-fold only the characters inside `ranges`, leaving everything else untouched."""
    members = frozenset(codepoint for span in ranges for codepoint in span)

    def transform(text: str) -> str:
        return "".join(
            unicodedata.normalize("NFKC", char) if ord(char) in members else char for char in text
        )

    return transform


def _compose(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _ascii_digit(char: str) -> str:
    if char.isascii() or unicodedata.category(char) != "Nd":
        return char
    return str(unicodedata.decimal(char))


def _fold_non_ascii_digits(text: str) -> str:
    return "".join(_ascii_digit(char) for char in text)


def _soft_hyphen_line_final(text: str) -> str:
    """A trailing soft hyphen was rendered as a real hyphen — the line broke there."""
    if text.endswith(SOFT_HYPHEN):
        return text[: -len(SOFT_HYPHEN)] + "-"
    return text


def _soft_hyphen_mid_line(text: str) -> str:
    """A soft hyphen anywhere but the end was never drawn."""
    if not text:
        return text
    head, last = text[:-1], text[-1]
    return head.replace(SOFT_HYPHEN, "") + last


_TYPOGRAPHIC_PUNCTUATION = {
    0x2018: "'",
    0x2019: "'",
    0x201C: '"',
    0x201D: '"',
    0x2010: "-",
    0x2011: "-",
    0x2012: "-",
    0x2013: "-",
    0x2014: "-",
    0x2015: "-",
    0x2212: "-",
}

# Order matters for the collapsing subset: the BOM sits inside the Arabic Forms-B range, so it
# must be stripped before that fold runs, and NFC must run last because folding can emit
# decomposed sequences.
DEFAULT_AXES: tuple[Axis, ...] = (
    Axis(
        name="zero_width_non_content",
        action=Action.COLLAPSE,
        rationale="BOM and zero-width space are not page content",
        transform=_strip(0xFEFF, 0x200B),
    ),
    Axis(
        name="arabic_presentation_forms",
        action=Action.COLLAPSE,
        rationale="legacy pre-shaped glyphs; shaping is derivable from context",
        transform=_fold_ranges(range(0xFB50, 0xFE00), range(0xFE70, 0xFF00)),
    ),
    Axis(
        name="nbsp",
        action=Action.COLLAPSE,
        rationale="differs only in line-breaking, which a line image cannot show",
        transform=_replace({0x00A0: " "}),
    ),
    Axis(
        name="nfc_composition",
        action=Action.COLLAPSE,
        rationale="canonically equivalent — not a difference in the text",
        transform=_compose,
    ),
    Axis(
        name="soft_hyphen_line_final",
        action=Action.DECIDE,
        rationale="rendered as a real hyphen where the line breaks",
        transform=_soft_hyphen_line_final,
    ),
    Axis(
        name="soft_hyphen_mid_line",
        action=Action.DECIDE,
        rationale="never drawn away from a line break",
        transform=_soft_hyphen_mid_line,
    ),
    Axis(
        name="fullwidth_forms",
        action=Action.PRESERVE,
        rationale="different characters with visibly wider glyphs",
        transform=_fold_ranges(range(0xFF01, 0xFF61), range(0xFFE0, 0xFFE7)),
    ),
    Axis(
        name="ligatures",
        action=Action.PRESERVE,
        rationale="typography the model can see in the image",
        transform=_fold_ranges(range(0xFB00, 0xFB50)),
    ),
    Axis(
        name="non_ascii_digits",
        action=Action.PRESERVE,
        rationale="Arabic-Indic and other digits are visibly distinct from ASCII",
        transform=_fold_non_ascii_digits,
    ),
    Axis(
        name="typographic_punctuation",
        action=Action.PRESERVE,
        rationale="curly quotes and dash widths are distinct glyphs",
        transform=_replace(_TYPOGRAPHIC_PUNCTUATION),
    ),
    Axis(
        name="ideographic_space",
        action=Action.PRESERVE,
        rationale="U+3000 is a visibly wider space",
        transform=_replace({0x3000: " "}),
    ),
    Axis(
        name="zero_width_joiners",
        action=Action.PRESERVE,
        rationale="ZWJ and ZWNJ change the shape of neighbouring letters",
        transform=_strip(ord(ZERO_WIDTH_NON_JOINER), ord(ZERO_WIDTH_JOINER)),
    ),
)


def axes_with(action: Action, axes: tuple[Axis, ...] = DEFAULT_AXES) -> tuple[Axis, ...]:
    return tuple(axis for axis in axes if axis.action is action)
