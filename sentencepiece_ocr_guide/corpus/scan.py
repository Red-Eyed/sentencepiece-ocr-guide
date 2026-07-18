"""Measuring which axes actually vary, and in which source.

You cannot canonicalize what you have not characterised: the exception list beyond NFC is a
judgement about *your* extractors, and guessing at it is how a soft hyphen that was on the page
gets stripped. This produces the evidence that judgement needs.

Results reuse `CheckResult` so the corpus checklist and the model checklist render, rank and
exit identically. The mapping from `Action` to severity is the whole ranking:

- `COLLAPSE` variation is a **BLOCKER** — the label space is split before training starts.
- `DECIDE` variation is **HIGH** — it needs a call, and the counts are how you make it.
- `PRESERVE` variation is **INFO** — a non-zero count is expected and is not a defect.

Sources are supplied as iterables of lines, so the core needs no filesystem and a caller can
stream a file of any size: each source is consumed exactly once, with every axis counted in the
same pass.

Chunks are counted independently and folded afterwards, which is what makes the scan safe to
run across threads — no counter is touched by two workers, and the fold is commutative, so a
parallel run and a sequential one produce identical reports.
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

from sentencepiece_ocr_guide.checks.result import (
    MAX_EVIDENCE_ITEMS,
    CheckResult,
    Remedy,
    Report,
    Severity,
    Status,
)
from sentencepiece_ocr_guide.concurrency import batched, stream_parallel_unordered
from sentencepiece_ocr_guide.corpus.axes import DEFAULT_AXES, Action, Axis
from sentencepiece_ocr_guide.corpus.undecodable import has_undecodable_bytes

INVALID_UTF8 = "invalid_utf8"

# Large enough that per-chunk dispatch overhead is negligible, small enough that in-flight
# chunks stay a bounded slice of memory rather than a fraction of the corpus.
DEFAULT_CHUNK_LINES = 20_000

_SEVERITY_FOR_ACTION: dict[Action, Severity] = {
    Action.COLLAPSE: Severity.BLOCKER,
    Action.DECIDE: Severity.HIGH,
    Action.PRESERVE: Severity.INFO,
}

_REMEDY_FOR_ACTION: dict[Action, Remedy] = {
    Action.COLLAPSE: Remedy.FIX_CORPUS,
    Action.DECIDE: Remedy.FIX_CORPUS,
    Action.PRESERVE: Remedy.NOT_APPLICABLE,
}


@dataclass(frozen=True)
class ChunkCount:
    """What one chunk of lines contributed, labelled with the source it came from.

    Produced on a worker thread and never mutated, so handing it back to the consumer needs no
    lock. `_Totals` does the folding.
    """

    source: str
    scanned: int
    undecodable: int
    per_axis: Mapping[str, int] = field(default_factory=dict)


def scan_corpus(
    sources: Mapping[str, Iterable[str]],
    axes: tuple[Axis, ...] = DEFAULT_AXES,
    workers: int = 1,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
) -> Report:
    """Count, per axis and per source, how many lines change under that axis.

    Work is dispatched in chunks of lines rather than whole sources, so a corpus that is one
    large file parallelizes just as well as one that is a thousand shards.
    """
    if not sources:
        return Report(results=(_skipped("no sources supplied"),))

    chunks = _iter_chunks(sources, chunk_lines)
    # Completion order, not input order: counts are summed, and addition does not care which
    # worker finished first. The report stays deterministic because `_Totals` is keyed by source
    # and rendered in `sources` order.
    counted = stream_parallel_unordered(
        lambda chunk: count_chunk(chunk[0], chunk[1], axes), chunks, workers
    )

    totals = _Totals(sources)
    for chunk in counted:
        totals.add(chunk)

    return Report(results=_results_from(totals, axes))


def _iter_chunks(
    sources: Mapping[str, Iterable[str]], chunk_lines: int
) -> Iterator[tuple[str, list[str]]]:
    """Lazily label each batch of lines with the source it came from."""
    for source, lines in sources.items():
        for batch in batched(lines, chunk_lines):
            yield source, batch


def count_chunk(source: str, lines: Iterable[str], axes: tuple[Axis, ...]) -> ChunkCount:
    """Count one batch of lines in a single pass.

    The ASCII short-circuit is the difference between scanning a real corpus in seconds and in
    minutes: no axis can fire on a pure-ASCII line, so one C-level test replaces every transform
    for what is typically a large share of the corpus.
    """
    per_axis: dict[str, int] = {}
    scanned = 0
    undecodable = 0

    for line in lines:
        scanned += 1
        if line.isascii():
            continue

        if has_undecodable_bytes(line):
            undecodable += 1

        for axis in axes:
            if axis.affects(line):
                per_axis[axis.name] = per_axis.get(axis.name, 0) + 1

    return ChunkCount(source=source, scanned=scanned, undecodable=undecodable, per_axis=per_axis)


class _Totals:
    """Running totals, accumulated as chunk counts arrive.

    Mutable on purpose. Chunks are counted on worker threads into immutable `ChunkCount`s; this
    is folded by the consumer alone, on one thread, where copying a frozen record per chunk
    would buy no safety and cost an allocation for every chunk in the corpus.

    Axis counts are keyed axis-then-source, which is the shape the report wants, so rendering
    needs no second pass to invert them.
    """

    def __init__(self, sources: Iterable[str]) -> None:
        self.scanned: dict[str, int] = dict.fromkeys(sources, 0)
        self.undecodable: dict[str, int] = dict.fromkeys(self.scanned, 0)
        self.by_axis: dict[str, dict[str, int]] = {}

    def add(self, chunk: ChunkCount) -> None:
        self.scanned[chunk.source] += chunk.scanned
        self.undecodable[chunk.source] += chunk.undecodable

        for axis_name, count in chunk.per_axis.items():
            per_source = self.by_axis.setdefault(axis_name, {})
            per_source[chunk.source] = per_source.get(chunk.source, 0) + count


def _results_from(totals: _Totals, axes: tuple[Axis, ...]) -> tuple[CheckResult, ...]:
    return (
        _invalid_utf8_result(totals.undecodable, totals.scanned),
        *(_result_for(axis, totals.by_axis.get(axis.name, {}), totals.scanned) for axis in axes),
    )


def _invalid_utf8_result(undecodable: dict[str, int], scanned: dict[str, int]) -> CheckResult:
    """Undecodable bytes are corrupt data, not an encoding preference — canonicalizing cannot
    recover the intended text, so this reports rather than offering a transform."""
    total = sum(undecodable.values())
    total_scanned = sum(scanned.values())

    if total == 0:
        return CheckResult(
            check=INVALID_UTF8,
            status=Status.PASSED,
            summary=f"every one of {total_scanned:,} lines decoded as valid UTF-8",
            severity=Severity.BLOCKER,
            remedy=Remedy.FIX_CORPUS,
        )

    return CheckResult(
        check=INVALID_UTF8,
        status=Status.FAILED,
        summary=(
            f"{total:,} of {total_scanned:,} lines contain bytes that are not valid UTF-8 — "
            "fix the extractor; these bytes cannot be recovered by normalizing"
        ),
        evidence=_evidence(undecodable, scanned),
        severity=Severity.BLOCKER,
        remedy=Remedy.FIX_CORPUS,
    )


def _result_for(axis: Axis, per_source: dict[str, int], scanned: dict[str, int]) -> CheckResult:
    total = sum(per_source.values())
    return CheckResult(
        check=f"axis[{axis.name}]",
        status=_status(axis, total),
        summary=_summary(axis, total, sum(scanned.values())),
        evidence=_evidence(per_source, scanned) if total else (),
        severity=_SEVERITY_FOR_ACTION[axis.action],
        remedy=_REMEDY_FOR_ACTION[axis.action],
    )


def _status(axis: Axis, total: int) -> Status:
    """PRESERVE axes never fail: they report a measurement, not a defect."""
    if axis.action is Action.PRESERVE or total == 0:
        return Status.PASSED
    return Status.FAILED


def _summary(axis: Axis, total: int, scanned: int) -> str:
    """The axis name is already in the check name — do not repeat it here."""
    if total == 0:
        return f"no variation across {scanned:,} lines"
    note = " (expected — preserve, do not fold)" if axis.action is Action.PRESERVE else ""
    return f"{total:,} of {scanned:,} lines — {axis.rationale}{note}"


def _evidence(per_source: dict[str, int], scanned: dict[str, int]) -> tuple[str, ...]:
    """Worst source first — variation is usually one broken extractor, not a diffuse issue.

    Ties break on source name. Counts arrive in worker-completion order, so ranking on count
    alone would let two equally-affected sources swap places between runs, and a report that
    changes with `--jobs` is a report you cannot diff.
    """
    ranked = sorted(per_source.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        f"{source}: {count:,} / {scanned[source]:,} lines"
        for source, count in ranked[:MAX_EVIDENCE_ITEMS]
        if count
    )


def _skipped(reason: str) -> CheckResult:
    return CheckResult(
        check="corpus_scan",
        status=Status.SKIPPED,
        summary=reason,
        severity=Severity.BLOCKER,
        remedy=Remedy.FIX_CORPUS,
    )
