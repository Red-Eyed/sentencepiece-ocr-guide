import pytest

from sentencepiece_ocr_guide.checks.scripts import Script, script_of, scripts_in


@pytest.mark.parametrize(
    ("char", "expected"),
    [
        ("a", Script.LATIN),
        ("Z", Script.LATIN),
        ("é", Script.LATIN),
        ("5", Script.DIGIT),
        ("α", Script.GREEK),
        ("д", Script.CYRILLIC),
        ("ع", Script.ARABIC),
        ("א", Script.HEBREW),
        ("क", Script.DEVANAGARI),
        ("ব", Script.BENGALI),
        ("த", Script.TAMIL),
        ("ก", Script.THAI),
        ("文", Script.HAN),
        ("ひ", Script.HIRAGANA),
        ("カ", Script.KATAKANA),
        ("한", Script.HANGUL),
        (" ", Script.COMMON),
        (".", Script.COMMON),
        ("\\", Script.COMMON),
    ],
)
def test_script_of_classifies_representative_characters(char: str, expected: Script) -> None:
    assert script_of(char) is expected


@pytest.mark.parametrize(
    ("char", "expected"),
    [
        ("１", Script.DIGIT),  # fullwidth digit still reads as a digit
        ("Ａ", Script.LATIN),  # fullwidth Latin still reads as Latin
        ("ﾊ", Script.KATAKANA),  # halfwidth katakana still reads as katakana
    ],
)
def test_fullwidth_forms_classify_as_the_script_they_mirror(char: str, expected: Script) -> None:
    assert script_of(char) is expected


def test_scripts_in_excludes_common() -> None:
    assert scripts_in("hello, world!") == {Script.LATIN}


def test_scripts_in_detects_a_mixed_piece() -> None:
    assert scripts_in("a文") == {Script.LATIN, Script.HAN}


def test_latex_command_is_single_script() -> None:
    """Backslash and braces are COMMON, so \\frac must not look cross-script."""
    assert scripts_in(r"\frac{}") == {Script.LATIN}
