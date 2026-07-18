"""Unicode script classification.

`unicodedata` exposes categories but not scripts, so the ranges the guide cares about are
tabulated here. This is deliberately not a complete implementation of UAX #24 — it covers the
scripts discussed in the guide and answers `OTHER` elsewhere, which is enough to detect a piece
that straddles two writing systems.

`COMMON` (punctuation, whitespace, symbols) is not a script: it appears legitimately inside
pieces of every script and must never count toward a cross-script merge.
"""

from bisect import bisect_right
from enum import StrEnum


class Script(StrEnum):
    LATIN = "Latin"
    GREEK = "Greek"
    CYRILLIC = "Cyrillic"
    HEBREW = "Hebrew"
    ARABIC = "Arabic"
    DEVANAGARI = "Devanagari"
    BENGALI = "Bengali"
    TAMIL = "Tamil"
    THAI = "Thai"
    HAN = "Han"
    HIRAGANA = "Hiragana"
    KATAKANA = "Katakana"
    HANGUL = "Hangul"
    DIGIT = "Digit"
    COMMON = "Common"
    OTHER = "Other"


# (first_codepoint, script) — sorted; a codepoint belongs to the last entry at or below it.
# Gaps between listed blocks fall through to OTHER via the explicit sentinel entries.
_RANGES: tuple[tuple[int, Script], ...] = (
    (0x0000, Script.COMMON),
    (0x0030, Script.DIGIT),  # ASCII 0-9
    (0x003A, Script.COMMON),
    (0x0041, Script.LATIN),
    (0x005B, Script.COMMON),
    (0x0061, Script.LATIN),
    (0x007B, Script.COMMON),
    (0x00C0, Script.LATIN),  # Latin-1 Supplement letters onward
    (0x0250, Script.LATIN),  # IPA Extensions
    (0x0370, Script.GREEK),
    (0x0400, Script.CYRILLIC),
    (0x0530, Script.OTHER),  # Armenian
    (0x0590, Script.HEBREW),
    (0x0600, Script.ARABIC),
    (0x0700, Script.OTHER),  # Syriac and others
    (0x0900, Script.DEVANAGARI),
    (0x0980, Script.BENGALI),
    (0x0A00, Script.OTHER),  # Gurmukhi, Gujarati, Oriya
    (0x0B80, Script.TAMIL),
    (0x0C00, Script.OTHER),  # Telugu, Kannada, Malayalam, Sinhala
    (0x0E00, Script.THAI),
    (0x0E80, Script.OTHER),
    (0x1100, Script.HANGUL),  # Hangul Jamo
    (0x1200, Script.OTHER),
    (0x1E00, Script.LATIN),  # Latin Extended Additional
    (0x1F00, Script.GREEK),  # Greek Extended
    (0x2000, Script.COMMON),  # punctuation, symbols, arrows, math operators
    (0x2E80, Script.HAN),  # CJK radicals through unified ideographs
    (0x3040, Script.HIRAGANA),
    (0x30A0, Script.KATAKANA),
    (0x3100, Script.HAN),  # Bopomofo, CJK strokes, compatibility
    (0x3130, Script.HANGUL),  # Hangul Compatibility Jamo
    (0x3190, Script.HAN),
    (0x4E00, Script.HAN),  # CJK Unified Ideographs
    (0xA000, Script.OTHER),
    (0xAC00, Script.HANGUL),  # Hangul Syllables
    (0xD7B0, Script.OTHER),
    (0xF900, Script.HAN),  # CJK Compatibility Ideographs
    (0xFB00, Script.LATIN),  # Alphabetic Presentation Forms (ligatures)
    (0xFB1D, Script.HEBREW),
    (0xFB50, Script.ARABIC),  # Arabic Presentation Forms-A
    (0xFE00, Script.COMMON),
    (0xFE70, Script.ARABIC),  # Arabic Presentation Forms-B
    (0xFF00, Script.COMMON),  # Fullwidth forms — refined below
    (0x10000, Script.OTHER),
    (0x20000, Script.HAN),  # CJK Extension B and beyond
    (0x30000, Script.OTHER),
)

_STARTS: tuple[int, ...] = tuple(start for start, _ in _RANGES)
_SCRIPTS: tuple[Script, ...] = tuple(script for _, script in _RANGES)

# Fullwidth block duplicates ASCII; classify its members as the script they mirror so a
# fullwidth digit still reads as DIGIT. See docs/04-scripts.md.
_FULLWIDTH_DIGITS = range(0xFF10, 0xFF1A)
_FULLWIDTH_UPPER = range(0xFF21, 0xFF3B)
_FULLWIDTH_LOWER = range(0xFF41, 0xFF5B)
_HALFWIDTH_KATAKANA = range(0xFF66, 0xFFA0)


def script_of(char: str) -> Script:
    """The script a single character belongs to."""
    codepoint = ord(char)

    if codepoint in _FULLWIDTH_DIGITS:
        return Script.DIGIT
    if codepoint in _FULLWIDTH_UPPER or codepoint in _FULLWIDTH_LOWER:
        return Script.LATIN
    if codepoint in _HALFWIDTH_KATAKANA:
        return Script.KATAKANA

    index = bisect_right(_STARTS, codepoint) - 1
    return _SCRIPTS[index]


def scripts_in(text: str) -> frozenset[Script]:
    """Every script present in `text`, `COMMON` excluded.

    `COMMON` is dropped because punctuation and whitespace belong to no writing system —
    counting them would make every ordinary piece look cross-script.
    """
    return frozenset(script_of(char) for char in text) - {Script.COMMON}
