"""Rendering a `Report`. The only module that decides how results look.

Shared by both checklists so the corpus scan and the model check rank, read and exit the same
way. Checks and scanners return data; nothing below this line runs any logic on a tokenizer.
"""

import json

from sentencepiece_ocr_guide.checks.result import (
    SEVERITY_RANK,
    CheckResult,
    Remedy,
    Report,
    Severity,
    Status,
)

_STATUS_MARKS = {Status.PASSED: "PASS", Status.FAILED: "FAIL", Status.SKIPPED: "SKIP"}

_NEXT_STEP = {
    Remedy.FIX_CORPUS: "canonicalize the corpus, then retrain",
    Remedy.RETRAIN_CONFIG: "change the trainer flags and retrain",
    Remedy.FIX_INTEGRATION: "fix the wiring between tokenizer and checkpoint",
    Remedy.NOT_APPLICABLE: "nothing — reported for visibility",
}


def as_json(report: Report) -> str:
    return json.dumps(
        {
            "ok": report.ok,
            "worst_severity": report.worst_severity().value,
            "remedies": [remedy.value for remedy in report.remedies()],
            "results": [result.model_dump(mode="json") for result in report.ranked()],
        },
        indent=2,
        ensure_ascii=False,
    )


def as_text(report: Report) -> str:
    lines = [_format_result(result) for result in report.ranked()]
    lines.extend(["", _format_totals(report)])
    lines.extend(_format_next_steps(report))
    return "\n".join(lines)


def _format_result(result: CheckResult) -> str:
    mark = _STATUS_MARKS[result.status]
    tag = "" if result.status is Status.PASSED else f" [{result.severity.value}]"
    head = f"{mark}{tag}  {result.check}: {result.summary}"
    evidence = "".join(f"\n        {item}" for item in result.evidence)
    return head + evidence


def _format_totals(report: Report) -> str:
    counts = {status: len(report.with_status(status)) for status in Status}
    return (
        f"{counts[Status.PASSED]} passed, "
        f"{counts[Status.FAILED]} failed, "
        f"{counts[Status.SKIPPED]} skipped"
    )


def _format_next_steps(report: Report) -> list[str]:
    remedies = [remedy for remedy in report.remedies() if remedy is not Remedy.NOT_APPLICABLE]
    if not remedies:
        return []

    lines = ["", "Next:"]
    lines.extend(f"  {index}. {_NEXT_STEP[remedy]}" for index, remedy in enumerate(remedies, 1))
    if Remedy.FIX_CORPUS in remedies and Remedy.RETRAIN_CONFIG in remedies:
        lines.append("  (in that order — a corpus defect survives any number of retrains)")
    return lines


def exit_code(report: Report, fail_on: Severity) -> int:
    """Non-zero when any failure is at or above `fail_on`."""
    if report.ok:
        return 0
    return 1 if SEVERITY_RANK[report.worst_severity()] >= SEVERITY_RANK[fail_on] else 0
