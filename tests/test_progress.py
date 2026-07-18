"""The progress bar's contract with the streams it wraps.

Nothing here asserts on rendering — the bar disables itself when stderr is not a terminal, which
under pytest it never is. What matters is that wrapping a stream is *invisible* to the scan: the
same lines, in the same order, pulled just as lazily, and no new way to fail. A progress bar that
crashed the scan it was measuring, or that drained a corpus into memory to measure it, would be
worse than no bar at all.
"""

import pytest

from sentencepiece_ocr_guide.progress import byte_progress, line_bytes

# What `_stream_lines` hands back for a byte that was not valid UTF-8: decoding with
# `surrogateescape` parks it in a lone surrogate rather than raising.
UNDECODABLE = b"caf\xe9 latte".decode("utf-8", "surrogateescape")


@pytest.mark.parametrize(
    "line",
    ["plain ascii text", "", "Привіт світе", "שלום עולם", "இது ஒரு தமிழ்", "x²+y²  — ±∞"],
)
def test_line_bytes_matches_what_the_line_occupied_on_disk(line: str) -> None:
    """The bar's total comes from `st_size`, so its increments must be real byte counts.

    Counting characters instead would leave a non-Latin corpus reporting a fraction of its
    true size and a bar that finished at 40%.
    """
    assert line_bytes(line) == len(line.encode("utf-8"))


def test_line_bytes_survives_bytes_that_were_never_valid_utf8() -> None:
    """The regression this file exists for.

    Corpora full of invalid UTF-8 are the tool's entire subject, so such lines reach the bar
    on any real run. A plain `.encode("utf-8")` here raises `UnicodeEncodeError` on the lone
    surrogate and takes down the scan — the error handler is load-bearing, not decoration.
    """
    assert line_bytes(UNDECODABLE) == len(b"caf\xe9 latte")


def test_tracking_yields_the_same_lines_unchanged() -> None:
    """The wrapper observes; it must not transform. Undecodable lines included, since the
    scanner detects them by inspecting exactly the surrogates it is handed."""
    lines = ["ascii", "Привіт світе", UNDECODABLE, ""]

    with byte_progress(total_bytes=64, description="test") as track:
        assert list(track(lines)) == lines


def test_tracking_does_not_read_the_stream_up_front() -> None:
    """`scan_corpus` streams so a multi-gigabyte corpus never lands in memory at once.

    Wrapping the source in anything eager — a `list`, a length pre-pass to size the bar —
    would quietly undo that, and the failure would only show on a corpus too big to test.
    """
    read: list[str] = []

    def source():
        for line in ["a", "b", "c"]:
            read.append(line)
            yield line

    with byte_progress(total_bytes=3, description="test") as track:
        tracked = track(source())
        assert read == [], "wrapping alone must not pull a line"

        assert next(tracked) == "a"
        assert read == ["a"], "one line consumed should pull exactly one line"


def test_an_empty_corpus_is_not_a_special_case() -> None:
    """A scan that discovers no files still opens a bar, with a total of zero."""
    with byte_progress(total_bytes=0, description="test") as track:
        assert list(track([])) == []
