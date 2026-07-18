"""Measuring which axes actually vary, and in which source.

You cannot canonicalize what you have not characterised: the exception list beyond NFC is a
judgement about *your* extractors, and guessing at it is how a soft hyphen that was on the page
gets stripped. This produces the evidence that judgement needs.

Results reuse `CheckResult` so the corpus checklist and the model checklist render, rank and
exit identically. The mapping from `Action` to severity is the whole ranking:

- `COLLAPSE` variation is a **BLOCKER** — the label space is split before training starts.
- `DECIDE` variation is **HIGH** — it needs a call, and the counts are how you make it.
- `PRESERVE` variation is **INFO** — a non-zero count is expected and is not a defect.

Sources are supplied as iterables of lines, so the core needs no filesystem: each source is
consumed exactly once, with every axis counted in the same pass.
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


def scan_corpus(
    sources: Mapping[str, Iterable[str]],
    axes: tuple[Axis, ...] = DEFAULT_AXES,
) -> Report:
    """Count, per axis and per source, how many lines change under that axis."""
    if not sources:
        return Report(
            results=(
                CheckResult.skipped("corpus_scan", "no sources supplied").model_copy(
                    update={"severity": Severity.BLOCKER, "remedy": Remedy.FIX_CORPUS}
                ),
            )
        )

    affected, scanned = _tally(sources, axes)
    return Report(results=tuple(_result_for(axis, affected[axis.name], scanned) for axis in axes))


def _tally(
    sources: Mapping[str, Iterable[str]], axes: tuple[Axis, ...]
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """One pass per source, every axis counted together."""
    affected: dict[str, dict[str, int]] = {axis.name: {} for axis in axes}
    scanned: dict[str, int] = {}

    for source, lines in sources.items():
        scanned[source] = 0
        for line in lines:
            scanned[source] += 1
            for axis in axes:
                if axis.affects(line):
                    affected[axis.name][source] = affected[axis.name].get(source, 0) + 1

    return affected, scanned


def _result_for(axis: Axis, per_source: dict[str, int], scanned: dict[str, int]) -> CheckResult:
    total = sum(per_source.values())
    severity = _SEVERITY_FOR_ACTION[axis.action]
    remedy = _REMEDY_FOR_ACTION[axis.action]

    summary = _summary(axis, total, sum(scanned.values()))
    status = _status(axis, total)
    evidence = _evidence(per_source, scanned) if total else ()

    return CheckResult(
        check=f"axis[{axis.name}]",
        status=status,
        summary=summary,
        evidence=evidence,
        severity=severity,
        remedy=remedy,
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
    )
