"""Showing that a long scan is moving.

A corpus scan over a few hundred megabytes runs for minutes with nothing on screen, which is
indistinguishable from a hang. This attaches a bar to the line streams *at the call site* — the
scanner, the chunker and the thread pool are untouched and know nothing about it. Progress is an
observation the caller wants, so it belongs to the caller, not threaded through as a parameter
every layer would then have to carry.

Bytes are the unit because the total is knowable before reading anything (`stat().st_size`); a
line count is not, and a bar with no total shows neither a percentage nor an estimate.

The bar renders to stderr and disables itself when stderr is not a terminal, so a `--json` run
piped to a file stays exactly as parseable as it was before.
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

from tqdm import tqdm

Track = Callable[[Iterable[str]], Iterator[str]]


@contextmanager
def byte_progress(total_bytes: int, description: str) -> Iterator[Track]:
    """Yield a pass-through wrapper that advances one shared bar as lines flow through it.

    One bar spans every stream handed to the wrapper, so a corpus of a thousand shards reports
    as a single job rather than a thousand bars scrolling past.

    `disable=None` is tqdm's "decide for me": a bar on an interactive terminal, silence when
    stderr is redirected.
    """
    with tqdm(
        total=total_bytes,
        desc=description,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        disable=None,
        leave=False,
    ) as bar:

        def track(lines: Iterable[str]) -> Iterator[str]:
            for line in lines:
                bar.update(line_bytes(line))
                yield line

        yield track

        # Blank lines and newline characters never reach the wrapper, so the tally always lands
        # a little short of the file size. Close at the total rather than leaving a bar that
        # stops at 99% on a run that in fact read everything.
        bar.update(max(0, bar.total - bar.n))


def line_bytes(line: str) -> int:
    """How many bytes this line occupied on disk.

    The ASCII fast path is the point: `count_chunk` is built around a corpus being mostly ASCII,
    and encoding every line to measure it would spend more time than the scan it is measuring.
    For ASCII, one character is one byte and the C-level `isascii` check settles it.
    """
    if line.isascii():
        return len(line)
    return len(line.encode("utf-8", "surrogateescape"))
