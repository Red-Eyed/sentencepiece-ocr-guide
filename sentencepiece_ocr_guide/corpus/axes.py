"""The axes on which two sources can disagree about how to encode the same text.

An axis is a named, pure `str -> str` transform plus a verdict on what should happen to it. A
line is *affected* by an axis when the transform changes it, which is all the scanner needs; the
canonicalizer reuses the same transforms for the subset it is allowed to apply.

The verdict is what keeps the two uses honest. `PRESERVE` axes carry a transform purely so they
can be *detected* — `canonicalize` filters on `Action` and can never fold one by accident. See
docs/09-failure-modes.md for the reasoning behind each verdict.

**Every axis here fires only on non-ASCII input.** That invariant is what makes scanning a real
corpus affordable: a pure-ASCII line cannot be affected by any axis, so the scanner skips all of
them with one C-level test. `tests/corpus/test_axes.py` asserts it for every axis, because a new
axis that broke it would silently stop being detected on ASCII lines.

Axes also carry the exact set of characters that can trigger them, bound to the transform at
construction so the two cannot drift. When a line contains none of them, the transform is
skipped entirely.
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
    """One way two encodings of the same text can differ.

    `triggers` is the exact set of characters that can make `transform` change a line, or `None`
    when that set is impractical to enumerate. It is only ever an optimisation: `affects` still
    confirms with the transform itself, so an over-broad trigger set costs speed, never accuracy.
    """

    name: str
    action: Action
    rationale: str
    transform: Callable[[str], str]
    triggers: frozenset[str] | None = None

    def affects(self, line: str) -> bool:
        if line.isascii():
            return False
        if self.triggers is not None and self.triggers.isdisjoint(line):
            return False
        return self.transform(line) != line


def _chars(*codepoints: int) -> frozenset[str]:
    return frozenset(chr(codepoint) for codepoint in codepoints)


def _strip_axis(name: str, action: Action, rationale: str, *codepoints: int) -> Axis:
    """Delete `codepoints`. Triggers are exactly those codepoints."""
    table: dict[int, str | None] = {codepoint: None for codepoint in codepoints}

    def transform(text: str) -> str:
        return text.translate(table)

    return Axis(name, action, rationale, transform, _chars(*codepoints))


def _replace_axis(name: str, action: Action, rationale: str, mapping: Mapping[int, str]) -> Axis:
    """Rewrite `mapping`'s codepoints. Triggers are exactly its keys."""
    table: dict[int, str | None] = dict(mapping)

    def transform(text: str) -> str:
        return text.translate(table)

    return Axis(name, action, rationale, transform, _chars(*mapping))


def _fold_axis(name: str, action: Action, rationale: str, *ranges: range) -> Axis:
    """NFKC-fold only the characters inside `ranges`. Triggers are exactly those ranges."""
    members = frozenset(codepoint for span in ranges for codepoint in span)

    def transform(text: str) -> str:
        return "".join(
            unicodedata.normalize("NFKC", char) if ord(char) in members else char for char in text
        )

    return Axis(name, action, rationale, transform, frozenset(map(chr, members)))


def _compose(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _ascii_digit(char: str) -> str:
    # `str.isdecimal` is the C-level equivalent of `unicodedata.category(char) == "Nd"`.
    if char.isascii() or not char.isdecimal():
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
    _strip_axis(
        "zero_width_non_content",
        Action.COLLAPSE,
        "BOM and zero-width space are not page content",
        0xFEFF,
        0x200B,
    ),
    _fold_axis(
        "arabic_presentation_forms",
        Action.COLLAPSE,
        "legacy pre-shaped glyphs; shaping is derivable from context",
        range(0xFB50, 0xFE00),
        range(0xFE70, 0xFF00),
    ),
    _fold_axis(
        "hebrew_presentation_forms",
        Action.COLLAPSE,
        "legacy precomposed letter+point glyphs; identical to the logical sequence",
        range(0xFB1D, 0xFB50),
    ),
    _replace_axis(
        "nbsp",
        Action.COLLAPSE,
        "differs only in line-breaking, which a line image cannot show",
        {0x00A0: " "},
    ),
    Axis(
        name="nfc_composition",
        action=Action.COLLAPSE,
        rationale="canonically equivalent — not a difference in the text",
        transform=_compose,
        # Composability is a property of sequences, not of an enumerable character set.
        # `normalize` is C-level and measured at ~1% of scan time, so no fast path is needed.
        triggers=None,
    ),
    Axis(
        name="soft_hyphen_line_final",
        action=Action.DECIDE,
        rationale="rendered as a real hyphen where the line breaks",
        transform=_soft_hyphen_line_final,
        triggers=frozenset(SOFT_HYPHEN),
    ),
    Axis(
        name="soft_hyphen_mid_line",
        action=Action.DECIDE,
        rationale="never drawn away from a line break",
        transform=_soft_hyphen_mid_line,
        triggers=frozenset(SOFT_HYPHEN),
    ),
    _fold_axis(
        "fullwidth_forms",
        Action.PRESERVE,
        "different characters with visibly wider glyphs",
        range(0xFF01, 0xFF61),
        range(0xFFE0, 0xFFE7),
    ),
    _fold_axis(
        "ligatures",
        Action.PRESERVE,
        "typography the model can see in the image",
        # Latin (FB00–FB06) and Armenian (FB13–FB17) ligatures only. The Hebrew presentation
        # forms that share this block are a legacy encoding artifact, not typography, and are
        # collapsed by their own axis above.
        range(0xFB00, 0xFB1D),
    ),
    Axis(
        name="non_ascii_digits",
        action=Action.PRESERVE,
        rationale="Arabic-Indic and other digits are visibly distinct from ASCII",
        transform=_fold_non_ascii_digits,
        # ~700 scattered codepoints across Unicode; enumerating them at import costs more than
        # the per-character `str.isdecimal` test it would save.
        triggers=None,
    ),
    _replace_axis(
        "typographic_punctuation",
        Action.PRESERVE,
        "curly quotes and dash widths are distinct glyphs",
        _TYPOGRAPHIC_PUNCTUATION,
    ),
    _replace_axis(
        "ideographic_space",
        Action.PRESERVE,
        "U+3000 is a visibly wider space",
        {0x3000: " "},
    ),
    _strip_axis(
        "zero_width_joiners",
        Action.PRESERVE,
        "ZWJ and ZWNJ change the shape of neighbouring letters",
        ord(ZERO_WIDTH_NON_JOINER),
        ord(ZERO_WIDTH_JOINER),
    ),
)


def axes_with(action: Action, axes: tuple[Axis, ...] = DEFAULT_AXES) -> tuple[Axis, ...]:
    return tuple(axis for axis in axes if axis.action is action)
