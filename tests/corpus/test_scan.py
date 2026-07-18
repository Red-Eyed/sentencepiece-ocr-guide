"""The scanner: does it find the right axes, attribute them, and rank them correctly."""

import unicodedata

from sentencepiece_ocr_guide.checks.result import Remedy, Severity, Status
from sentencepiece_ocr_guide.corpus.canonicalize import canonicalizer
from sentencepiece_ocr_guide.corpus.scan import scan_corpus

DECOMPOSED = unicodedata.normalize("NFD", "café")


def _result(report, axis_name: str):
    return next(item for item in report.results if item.check == f"axis[{axis_name}]")


def test_clean_corpus_passes_every_collapsible_axis() -> None:
    report = scan_corpus({"clean": ["café", "plain ascii", "文字"]})

    assert report.ok
    assert _result(report, "nfc_composition").status is Status.PASSED


def test_detects_decomposed_text_as_a_blocker() -> None:
    report = scan_corpus({"vendor_b": [DECOMPOSED, "clean line"]})

    result = _result(report, "nfc_composition")

    assert result.status is Status.FAILED
    assert result.severity is Severity.BLOCKER
    assert result.remedy is Remedy.FIX_CORPUS
    assert not report.ok


def test_attributes_variation_to_the_source_that_caused_it() -> None:
    """Variation is usually one broken extractor, not a diffuse problem."""
    report = scan_corpus(
        {
            "good_source": ["café", "clean"],
            "broken_extractor": [DECOMPOSED, DECOMPOSED, DECOMPOSED],
        }
    )

    evidence = _result(report, "nfc_composition").evidence

    assert evidence[0].startswith("broken_extractor: 3 / 3 lines")
    assert len(evidence) == 1  # the clean source contributes nothing


def test_preserve_axes_are_reported_but_never_fail() -> None:
    """Fullwidth text is confirmation you have CJK data, not a defect."""
    report = scan_corpus({"cjk": ["Ｆｕｌｌｗｉｄｔｈ", "ＡＢＣ"]})

    result = _result(report, "fullwidth_forms")

    assert result.status is Status.PASSED
    assert result.severity is Severity.INFO
    assert result.remedy is Remedy.NOT_APPLICABLE
    assert "preserve, do not fold" in result.summary
    assert report.ok


def test_soft_hyphen_positions_are_counted_separately() -> None:
    """9,117 line-final versus 226 mid-line is the number that makes the call for you."""
    report = scan_corpus({"pdf": ["exam­", "ex­am", "ex­am"]})

    line_final = _result(report, "soft_hyphen_line_final")
    mid_line = _result(report, "soft_hyphen_mid_line")

    assert line_final.severity is Severity.HIGH
    assert "1 of 3 lines" in line_final.summary
    assert "2 of 3 lines" in mid_line.summary


def test_ranking_puts_blockers_above_the_rest() -> None:
    report = scan_corpus({"mixed": [DECOMPOSED, "exam­", "Ｆｕｌｌ"]})

    ranked = [result.check for result in report.ranked()]

    assert ranked[0] == "axis[nfc_composition]"
    assert report.worst_severity() is Severity.BLOCKER


def test_empty_sources_skip_rather_than_pass() -> None:
    report = scan_corpus({})

    assert report.results[0].status is Status.SKIPPED
    assert report.ok  # a skip does not fail the run, but it is not a pass either


def test_canonicalizing_clears_every_collapsible_axis() -> None:
    """The proof the stage did what it was meant to: re-scan and the blockers are gone."""
    canonicalize = canonicalizer()
    dirty = [DECOMPOSED, "a﻿b", "a b", "ﻛﺘﺎﺏ"]

    before = scan_corpus({"raw": dirty})
    after = scan_corpus({"canonicalized": [canonicalize(line) for line in dirty]})

    assert not before.ok
    assert after.ok


def test_canonicalizing_leaves_preserve_axes_untouched() -> None:
    canonicalize = canonicalizer()
    visible = ["Ｆｕｌｌ", "ﬁ", "٣", "mi‌rood"]

    before = scan_corpus({"raw": visible})
    after = scan_corpus({"canonicalized": [canonicalize(line) for line in visible]})

    for axis in ("fullwidth_forms", "ligatures", "non_ascii_digits", "zero_width_joiners"):
        assert _result(before, axis).summary == _result(after, axis).summary
