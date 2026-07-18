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
"""

from collections.abc import Iterable, Mapping

from sentencepiece_ocr_guide.checks.result import (
    MAX_EVIDENCE_ITEMS,
    CheckResult,
    Remedy,
    Report,
    Severity,
    Status,
)
from sentencepiece_ocr_guide.corpus.axes import DEFAULT_AXES, Action, Axis

INVALID_UTF8 = "invalid_utf8"

# `errors="surrogateescape"` maps each undecodable byte to one of these, so their presence in a
# decoded line is exactly the evidence that the source was not valid UTF-8.
_SURROGATES = frozenset(map(chr, range(0xDC80, 0xDD00)))

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


class _Tally:
    """Mutable counters for one scan. Kept private; `scan_corpus` returns immutable results."""

    def __init__(self, axes: tuple[Axis, ...]) -> None:
        self.affected: dict[str, dict[str, int]] = {axis.name: {} for axis in axes}
        self.scanned: dict[str, int] = {}
        self.undecodable: dict[str, int] = {}

    def record(self, axis_name: str, source: str) -> None:
        counts = self.affected[axis_name]
        counts[source] = counts.get(source, 0) + 1


def scan_corpus(
    sources: Mapping[str, Iterable[str]],
    axes: tuple[Axis, ...] = DEFAULT_AXES,
) -> Report:
    """Count, per axis and per source, how many lines change under that axis."""
    if not sources:
        return Report(results=(_skipped("no sources supplied"),))

    tally = _count(sources, axes)
    results = (
        _invalid_utf8_result(tally),
        *(_result_for(axis, tally.affected[axis.name], tally.scanned) for axis in axes),
    )
    return Report(results=results)


def _count(sources: Mapping[str, Iterable[str]], axes: tuple[Axis, ...]) -> _Tally:
    """One pass per source, every axis counted together.

    The ASCII short-circuit is the difference between scanning a real corpus in seconds and in
    minutes: no axis can fire on a pure-ASCII line, so one C-level test replaces every transform
    for what is typically a large share of the corpus.
    """
    tally = _Tally(axes)

    for source, lines in sources.items():
        tally.scanned[source] = 0
        tally.undecodable[source] = 0

        for line in lines:
            tally.scanned[source] += 1
            if line.isascii():
                continue

            if not _SURROGATES.isdisjoint(line):
                tally.undecodable[source] += 1

            for axis in axes:
                if axis.affects(line):
                    tally.record(axis.name, source)

    return tally


def _invalid_utf8_result(tally: _Tally) -> CheckResult:
    """Undecodable bytes are corrupt data, not an encoding preference — canonicalizing cannot
    recover the intended text, so this reports rather than offering a transform."""
    total = sum(tally.undecodable.values())
    scanned = sum(tally.scanned.values())

    if total == 0:
        return CheckResult(
            check=INVALID_UTF8,
            status=Status.PASSED,
            summary=f"every one of {scanned:,} lines decoded as valid UTF-8",
            severity=Severity.BLOCKER,
            remedy=Remedy.FIX_CORPUS,
        )

    return CheckResult(
        check=INVALID_UTF8,
        status=Status.FAILED,
        summary=(
            f"{total:,} of {scanned:,} lines contain bytes that are not valid UTF-8 — "
            "fix the extractor; these bytes cannot be recovered by normalizing"
        ),
        evidence=_evidence(tally.undecodable, tally.scanned),
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
    """Worst source first — variation is usually one broken extractor, not a diffuse issue."""
    ranked = sorted(per_source.items(), key=lambda item: item[1], reverse=True)
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
