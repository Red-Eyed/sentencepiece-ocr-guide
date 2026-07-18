"""Running independent work across threads.

This project targets free-threaded CPython (see `.python-version`), where threads execute
Python bytecode genuinely in parallel. The corpus workload is a good fit: sources are
independent, the per-line work is CPU-bound, and the shared state — the axis definitions and
their trigger sets — is immutable.

On a GIL build every helper here still works and still produces correct results; it simply
stops being faster. Nothing degrades, so callers need no build-specific branch.

Threads rather than processes, because the units of work carry open file handles and lazy
generators that cannot be pickled, the axis trigger sets would be rebuilt in every process
(macOS spawns rather than forks), and the results are counts that must be merged centrally
anyway.

Measured on an Apple M5 (4 performance + 6 efficiency cores), scanning a single 400k-line file:

    jobs      1      2      3      4      5      6      8
    speedup   1.00x  1.46x  1.83x  2.08x  1.00x  1.07x  1.11x

Throughput climbs to the performance-core count and then falls off a cliff back to the serial
rate. The cliff is not an artefact of this code: it sits at the same place for chunk sizes from
1k to 20k lines, and switching from ordered to completion-order results did not move it. So the
default is the performance-core count, not every core — asking for more is measurably worse
than asking for four.
"""

import os
import subprocess
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from itertools import islice
from typing import TypeVar

Item = TypeVar("Item")
Result = TypeVar("Result")

RESERVED_CORES = 1
IN_FLIGHT_PER_WORKER = 2


def default_workers() -> int:
    """How many threads to use when the caller does not say.

    On a machine where every core is equivalent this is one fewer than the core count, leaving
    room for the interactive session that launched it.

    Heterogeneous machines need the performance-core count instead, and the difference is not
    academic. Measured on an Apple M5 (4 performance + 6 efficiency cores) scanning 400k lines:
    4 threads ran 2.07x faster than serial, while 9 — one fewer than `os.cpu_count()` — ran at
    1.03x, barely better than not threading at all. Efficiency cores are slow enough that
    loading them costs more than the parallelism they add.
    """
    performance = _performance_cores()
    if performance is not None:
        return max(1, performance)
    return max(1, (os.cpu_count() or 1) - RESERVED_CORES)


def _performance_cores() -> int | None:
    """The count of fast cores, when the platform will say. `None` when it will not."""
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return None


def run_parallel(
    work: Callable[[Item], Result],
    items: Iterable[Item],
    workers: int,
) -> list[Result]:
    """Apply `work` to every item, in parallel when `workers` allows it.

    Results are returned in input order, so a parallel run and a sequential one produce
    byte-identical reports. Exceptions propagate rather than being collected: a failure here
    means the work itself is broken, which is a bug to surface, not a result to report.
    """
    materialized = list(items)
    if workers <= 1 or len(materialized) <= 1:
        return [work(item) for item in materialized]

    with ThreadPoolExecutor(max_workers=min(workers, len(materialized))) as pool:
        return list(pool.map(work, materialized))


def stream_parallel_unordered(
    work: Callable[[Item], Result],
    items: Iterable[Item],
    workers: int,
    in_flight: int | None = None,
) -> Iterator[Result]:
    """Like `run_parallel`, but never materializes the input, and yields as results arrive.

    `run_parallel` is fine for a known-small list such as the files in a directory. It is the
    wrong shape for corpus lines: `list(items)` there would pull an entire multi-gigabyte corpus
    into memory, defeating the streaming reader that feeds it. At most `in_flight` units of work
    are queued at once, so memory stays bounded by the chunk size regardless of corpus size.

    **Results arrive in completion order, not input order.** Use it only where the caller
    combines them commutatively — summing counts, for instance. Yielding in input order would
    make a fast worker wait on whichever unit happens to be slowest, and nothing here needs it.

    (CPython 3.14 added a `buffersize` argument to `Executor.map` that bounds the input the same
    way. This is written out longhand so the module keeps working on older interpreters, where
    it simply stops being faster.)
    """
    if workers <= 1:
        for item in items:
            yield work(item)
        return

    limit = in_flight or workers * IN_FLIGHT_PER_WORKER
    pending: set[Future[Result]] = set()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for item in items:
            pending.add(pool.submit(work, item))
            if len(pending) >= limit:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()

        for future in as_completed(pending):
            yield future.result()


def batched(items: Iterable[Item], size: int) -> Iterator[list[Item]]:
    """Group a stream into fixed-size lists.

    Chunking is what lets a *single* large file parallelize. Dispatching per source would leave
    one worker doing everything when the corpus is one big file, which is the common case.
    """
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch
