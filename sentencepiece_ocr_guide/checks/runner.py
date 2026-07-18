"""Running a suite of checks."""

from collections.abc import Sequence

from sentencepiece_ocr_guide.checks.protocols import Tokenizer
from sentencepiece_ocr_guide.checks.result import Check, CheckResult, Report


def run_checks(checks: Sequence[Check], tokenizer: Tokenizer) -> Report:
    """Run every check against `tokenizer` and collect the results.

    Exceptions are not caught. A check signals a failed *tokenizer* by returning a failed
    `CheckResult`; an exception therefore means the check itself is broken, which is a bug to
    surface rather than a result to report.
    """
    return Report(results=tuple(_run_one(check, tokenizer) for check in checks))


def _run_one(check: Check, tokenizer: Tokenizer) -> CheckResult:
    """Stamp the check's severity and remedy onto its result.

    Individual checks do not repeat this metadata: what a failure *means* is a property of the
    check, not of one run of it. Skipped results keep the declared severity too — a skipped
    BLOCKER is precisely what a reader must not mistake for a clean result.
    """
    result = check.run(tokenizer)
    return result.model_copy(update={"severity": check.severity, "remedy": check.remedy})
