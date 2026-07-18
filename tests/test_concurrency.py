"""Parallel execution helpers.

The property that matters is not speed — it is that turning parallelism on changes nothing
observable. A report that differed between `--jobs 1` and `--jobs 9` would be worse than a slow
one, because you could no longer trust either.
"""

import os
import threading
import unicodedata

import pytest

from sentencepiece_ocr_guide.concurrency import default_workers, run_parallel
from sentencepiece_ocr_guide.corpus.scan import _counted, count_chunk, scan_corpus

DECOMPOSED = unicodedata.normalize("NFD", "café")


def test_default_workers_is_at_least_one_and_never_oversubscribes() -> None:
    """On heterogeneous machines this is the performance-core count, not every core.

    Loading efficiency cores measured *slower* than not threading at all, so the default is
    capped by what the machine can actually run fast.
    """
    workers = default_workers()

    assert workers >= 1
    assert workers <= (os.cpu_count() or 1)


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_run_parallel_preserves_input_order(workers: int) -> None:
    assert run_parallel(lambda n: n * 2, range(50), workers) == [n * 2 for n in range(50)]


def test_counting_does_not_read_the_corpus_up_front() -> None:
    """Draining the reader ahead of the fold would pull a whole corpus into memory.

    `buffersize` is the only thing stopping `Executor.map` from doing exactly that, so this
    pulls a single count and asserts the reader is still a bounded distance in front of it.
    """
    read: list[str] = []

    def counting_source():
        for n in range(4_000):
            read.append(str(n))
            yield str(n)

    counted = _counted({"a": counting_source()}, (), workers=4, chunk_lines=1)
    next(counted)

    assert len(read) < 4_000, "the whole corpus was buffered before the first count came back"


def test_scan_exceptions_propagate_rather_than_being_swallowed() -> None:
    """A failure in the work means the work is broken — a bug to surface, not a count to report."""

    def explode():
        yield "fine"
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        scan_corpus({"a": explode()}, workers=4, chunk_lines=1)


def test_counting_actually_uses_multiple_threads() -> None:
    seen: set[int] = set()
    barrier = threading.Barrier(4, timeout=5)

    def record(_: int) -> int:
        barrier.wait()  # only returns if four threads arrive together
        seen.add(threading.get_ident())
        return 0

    run_parallel(record, range(8), workers=4)

    assert len(seen) >= 2


def test_count_chunk_is_indifferent_to_batch_type() -> None:
    """Chunks arrive from `itertools.batched` as tuples; the counter must not assume lists."""
    lines = (DECOMPOSED, "clean ascii")

    assert count_chunk("a", lines, ()).scanned == count_chunk("a", list(lines), ()).scanned


class TestScanIsUnaffectedByParallelism:
    lines = [DECOMPOSED, "clean ascii", "Ｆｕｌｌ", "ﻛﺘﺎﺏ", "exam­", "光学字符"] * 40

    def _report(self, workers: int, chunk_lines: int = 20_000):
        return scan_corpus(
            {"a": list(self.lines), "b": list(self.lines)},
            workers=workers,
            chunk_lines=chunk_lines,
        )

    def test_parallel_matches_sequential(self) -> None:
        assert self._report(workers=8).model_dump() == self._report(workers=1).model_dump()

    def test_chunking_matches_a_single_chunk(self) -> None:
        """A file split across many chunks must count the same as one read whole."""
        chunked = self._report(workers=4, chunk_lines=7)

        assert chunked.model_dump() == self._report(workers=1).model_dump()

    def test_a_single_source_still_splits_across_workers(self) -> None:
        """One big file is the common corpus shape; per-source dispatch would not parallelize."""
        one_file = scan_corpus({"only": list(self.lines)}, workers=4, chunk_lines=10)

        assert one_file.model_dump() == scan_corpus({"only": list(self.lines)}).model_dump()

    def test_an_empty_source_does_not_disturb_the_counts(self) -> None:
        """A file that yields no chunks must not break the merge or the denominator."""
        with_empty = scan_corpus({"empty": [], "full": list(self.lines)}, workers=4)
        without = scan_corpus({"full": list(self.lines)}, workers=4)

        assert f"{len(self.lines)} lines" in with_empty.results[0].summary
        assert with_empty.model_dump() == without.model_dump()


def test_evidence_order_is_stable_when_sources_tie() -> None:
    """Ranking equal sources on count alone would leave their order down to how the counts
    folded — and a report that changes with --jobs cannot be diffed."""
    identical = [DECOMPOSED] * 200
    sources = {name: list(identical) for name in ("d", "c", "b", "a")}

    runs = {
        scan_corpus(
            {name: list(lines) for name, lines in sources.items()}, workers=4, chunk_lines=10
        )
        .results[1]
        .evidence
        for _ in range(6)
    }

    assert len(runs) == 1, f"evidence order varied between runs: {runs}"
