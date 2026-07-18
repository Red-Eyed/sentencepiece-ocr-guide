"""Check outcomes, and how to act on them.

A check has three outcomes, not two. `SKIPPED` exists because a check that could not run must
never report success — silently passing an un-run check is the same class of silent failure
this whole package exists to catch. `SKIPPED` therefore always carries the reason it was
skipped, in the same field a failure would carry its evidence.

`Status` says whether a check failed. `Severity` says whether you should care, and `Remedy` says
what to do next — the two questions a bare pass/fail list leaves unanswered.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from sentencepiece_ocr_guide.checks.protocols import Tokenizer

MAX_EVIDENCE_ITEMS = 5


class Status(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Severity(StrEnum):
    """How much a failure matters. Ordered worst-first by `SEVERITY_RANK`."""

    BLOCKER = "blocker"
    """The label space is permanently broken. Do not spend training compute."""

    HIGH = "high"
    """Measurable accuracy loss, but the system will function."""

    MEDIUM = "medium"
    """Efficiency or vocabulary geometry — worth fixing, not worth blocking on."""

    INFO = "info"
    """Reported for visibility. Never a defect: a non-zero count here is expected."""


SEVERITY_RANK: dict[Severity, int] = {
    Severity.BLOCKER: 3,
    Severity.HIGH: 2,
    Severity.MEDIUM: 1,
    Severity.INFO: 0,
}


class Remedy(StrEnum):
    """What fixes this. The distinction that a per-artifact checklist cannot express.

    A model check can carry `FIX_CORPUS`: `nfc_vocabulary` fails on the *artifact* but retraining
    alone will never fix it, because the defect is in the data that produced it.
    """

    RETRAIN_CONFIG = "retrain_config"
    """Change a trainer flag and retrain. Cheap."""

    FIX_CORPUS = "fix_corpus"
    """Fix the data, *then* retrain. Retraining alone reproduces the failure."""

    FIX_INTEGRATION = "fix_integration"
    """Neither the tokenizer nor the corpus — the wiring around them."""

    NOT_APPLICABLE = "not_applicable"
    """Carried by `INFO` results, which report a measurement rather than a defect."""


class CheckResult(BaseModel, frozen=True):
    """The outcome of one check, serializable straight to JSON."""

    check: str
    status: Status
    summary: str
    evidence: tuple[str, ...] = Field(default=())
    severity: Severity = Severity.HIGH
    remedy: Remedy = Remedy.RETRAIN_CONFIG

    @classmethod
    def passed(cls, check: str, summary: str) -> "CheckResult":
        return cls(check=check, status=Status.PASSED, summary=summary)

    @classmethod
    def failed(cls, check: str, summary: str, evidence: Sequence[str] = ()) -> "CheckResult":
        return cls(
            check=check,
            status=Status.FAILED,
            summary=summary,
            evidence=tuple(evidence[:MAX_EVIDENCE_ITEMS]),
        )

    @classmethod
    def skipped(cls, check: str, reason: str) -> "CheckResult":
        """A check that could not run. `reason` explains why, and travels with the result."""
        return cls(check=check, status=Status.SKIPPED, summary=reason)

    @property
    def blocks(self) -> bool:
        return self.status is Status.FAILED and self.severity is Severity.BLOCKER


@dataclass(frozen=True)
class Check:
    """A named, fully-configured check.

    Builders in the sibling modules bind their parameters into `run` at construction, so the
    runner needs to know nothing about what any individual check requires. `severity` and
    `remedy` describe what it *means* when this check fails, and are stamped onto its result.
    """

    name: str
    run: Callable[[Tokenizer], CheckResult]
    severity: Severity
    remedy: Remedy


class Report(BaseModel, frozen=True):
    """The result of a full run."""

    results: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        """True when nothing failed. Skipped checks do not fail a run, but they are visible."""
        return not self.failures()

    def failures(self) -> tuple[CheckResult, ...]:
        return self.with_status(Status.FAILED)

    def with_status(self, status: Status) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status is status)

    def ranked(self) -> tuple[CheckResult, ...]:
        """Failures worst-first, then skips, then passes — the order to read them in."""
        return tuple(sorted(self.results, key=_reading_order))

    def worst_severity(self) -> Severity:
        """The severity of the most serious failure, or `INFO` when nothing failed."""
        failures = self.failures()
        if not failures:
            return Severity.INFO
        return max((result.severity for result in failures), key=lambda s: SEVERITY_RANK[s])

    def remedies(self) -> tuple[Remedy, ...]:
        """Distinct remedies across failures, in the order they must be applied."""
        needed = {result.remedy for result in self.failures()}
        return tuple(remedy for remedy in _REMEDY_ORDER if remedy in needed)


# Corpus fixes come first: a FIX_CORPUS defect survives any number of retrains.
_REMEDY_ORDER: tuple[Remedy, ...] = (
    Remedy.FIX_CORPUS,
    Remedy.RETRAIN_CONFIG,
    Remedy.FIX_INTEGRATION,
    Remedy.NOT_APPLICABLE,
)

_STATUS_ORDER: dict[Status, int] = {Status.FAILED: 0, Status.SKIPPED: 1, Status.PASSED: 2}


def _reading_order(result: CheckResult) -> tuple[int, int, str]:
    return (
        _STATUS_ORDER[result.status],
        -SEVERITY_RANK[result.severity],
        result.check,
    )
