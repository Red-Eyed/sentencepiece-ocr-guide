"""Rewriting a stream of lines into canonical form."""

import unicodedata

import pytest

from sentencepiece_ocr_guide.corpus.canonicalize import canonicalizer
from sentencepiece_ocr_guide.corpus.rewrite import (
    RewriteRun,
    RewriteTally,
    UndecodableLineError,
    rewrite_lines,
)
from sentencepiece_ocr_guide.corpus.scan import scan_corpus

DECOMPOSED = unicodedata.normalize("NFD", "café")
UNDECODABLE = b"bad \xff\xfe bytes".decode("utf-8", errors="surrogateescape")

canonicalize = canonicalizer()


def _rewrite(lines, drop_undecodable=False):
    tally = RewriteTally()
    written = list(
        rewrite_lines(lines, canonicalize, tally, "src", drop_undecodable=drop_undecodable)
    )
    return written, tally


def test_composes_decomposed_lines() -> None:
    written, tally = _rewrite([DECOMPOSED, "already clean"])

    assert written == ["café", "already clean"]
    assert tally.read == 2
    assert tally.changed == 1


def test_leaves_a_clean_corpus_byte_identical() -> None:
    clean = ["café", "光学字符识别", "Ｆｕｌｌ", "ﬁ", "٣"]

    written, tally = _rewrite(clean)

    assert written == clean
    assert tally.changed == 0


def test_refuses_undecodable_lines_by_default() -> None:
    """Normalizing cannot recover corrupt bytes, so carrying them through would launder them."""
    with pytest.raises(UndecodableLineError) as raised:
        _rewrite(["fine", UNDECODABLE])

    assert raised.value.line_number == 2
    assert "fix the extractor" in str(raised.value)


def test_drops_undecodable_lines_only_when_asked() -> None:
    written, tally = _rewrite(["fine", UNDECODABLE, "also fine"], drop_undecodable=True)

    assert written == ["fine", "also fine"]
    assert tally.dropped == 1
    assert tally.written == 2


def test_output_is_valid_utf8_and_encodable() -> None:
    """The point of refusing: what comes out can always be written to a UTF-8 file."""
    written, _ = _rewrite(["fine", UNDECODABLE, "café"], drop_undecodable=True)

    for line in written:
        line.encode("utf-8")  # must not raise


def test_rewriting_establishes_the_scan_invariant() -> None:
    """Canonicalize, then re-scan: every COLLAPSE axis must read zero."""
    dirty = [DECOMPOSED, "a﻿b", "a b", "ﻛﺘﺎﺏ", chr(0xFB30)]

    assert not scan_corpus({"raw": dirty}).ok

    written, _ = _rewrite(dirty)

    assert scan_corpus({"canonicalized": written}).ok


def test_rewriting_is_idempotent() -> None:
    dirty = [DECOMPOSED, "a﻿b", "a b", "ﻛﺘﺎﺏ"]

    once, _ = _rewrite(dirty)
    twice, tally = _rewrite(once)

    assert twice == once
    assert tally.changed == 0


class TestRewriteRun:
    def test_accumulates_totals_across_sources(self) -> None:
        run = RewriteRun()

        list(rewrite_lines([DECOMPOSED], canonicalize, run.tally_for("a"), "a"))
        list(rewrite_lines([DECOMPOSED, "clean"], canonicalize, run.tally_for("b"), "b"))

        assert run.changed == 2
        assert run.per_source["b"].read == 2

    def test_summary_mentions_drops_only_when_they_happened(self) -> None:
        clean = RewriteTally(read=10, written=10, changed=2)
        lossy = RewriteTally(read=10, written=9, changed=2, dropped=1)

        assert "dropped" not in clean.summary()
        assert "1 dropped (invalid UTF-8)" in lossy.summary()
