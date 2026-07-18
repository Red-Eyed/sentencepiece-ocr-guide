"""Parallel execution helpers.

The property that matters is not speed — it is that turning parallelism on changes nothing
observable. A report that differed between `--jobs 1` and `--jobs 9` would be worse than a slow
one, because you could no longer trust either.
"""

import os
import threading
import time
import unicodedata

import pytest

from sentencepiece_ocr_guide.concurrency import (
    batched,
    default_workers,
    run_parallel,
    stream_parallel_unordered,
)
from sentencepiece_ocr_guide.corpus.scan import scan_corpus

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


def test_stream_parallel_does_not_exhaust_its_input_up_front() -> None:
    """`list(items)` here would pull a whole corpus into memory; the point is that it does not."""
    consumed = []

    def counting_source():
        for n in range(100):
            consumed.append(n)
            yield n

    results = stream_parallel_unordered(lambda n: n, counting_source(), workers=4, in_flight=8)
    next(results)

    assert len(consumed) < 100, "input was fully consumed before the first result"


@pytest.mark.parametrize("workers", [1, 4])
def test_exceptions_propagate_rather_than_being_swallowed(workers: int) -> None:
    def explode(n: int) -> int:
        if n == 7:
            raise ValueError("boom")
        return n

    with pytest.raises(ValueError, match="boom"):
        list(stream_parallel_unordered(explode, range(20), workers))


def test_stream_parallel_actually_uses_multiple_threads() -> None:
    seen: set[int] = set()
    barrier = threading.Barrier(4, timeout=5)

    def record(_: int) -> int:
        barrier.wait()  # only returns if four threads arrive together
        seen.add(threading.get_ident())
        return 0

    list(stream_parallel_unordered(record, range(8), workers=4, in_flight=8))

    assert len(seen) >= 2


@pytest.mark.parametrize(("size", "expected"), [(1, 5), (2, 3), (5, 1), (10, 1)])
def test_batched_groups_without_losing_items(size: int, expected: int) -> None:
    batches = list(batched(range(5), size))

    assert len(batches) == expected
    assert [item for batch in batches for item in batch] == list(range(5))


def test_batched_on_empty_input_yields_nothing() -> None:
    assert list(batched([], 10)) == []


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


class TestUnorderedStreaming:
    """Used by the scan, where chunk counts are summed and order is irrelevant."""

    def test_returns_every_result(self) -> None:
        results = stream_parallel_unordered(lambda n: n * 2, range(50), workers=4)

        assert sorted(results) == sorted(n * 2 for n in range(50))

    def test_does_not_stall_behind_one_slow_item(self) -> None:
        """Head-of-line blocking is what makes ordered yielding collapse on slow cores.

        Asserting on result *order* rather than elapsed time: the slow item is submitted first,
        so an ordered consumer would have to emit it first. Here it must come out last.
        """

        def uneven(n: int) -> int:
            time.sleep(0.2 if n == 0 else 0.0)
            return n

        results = list(stream_parallel_unordered(uneven, range(8), workers=4, in_flight=8))

        assert results[0] != 0, "the slow item blocked the fast ones behind it"
        assert results[-1] == 0
        assert sorted(results) == list(range(8))

    def test_exceptions_still_propagate(self) -> None:
        def explode(n: int) -> int:
            if n == 5:
                raise ValueError("boom")
            return n

        with pytest.raises(ValueError, match="boom"):
            list(stream_parallel_unordered(explode, range(20), workers=4))


def test_evidence_order_is_stable_when_sources_tie() -> None:
    """Counts arrive in completion order, so ranking on count alone would let equal sources
    swap places between runs — and a report that changes with --jobs cannot be diffed."""
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
