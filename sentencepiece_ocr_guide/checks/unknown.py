"""No `<unk>` anywhere in the encoding path.

Failure mode #1 in docs/09-failure-modes.md: for OCR an `<unk>` is a *label*, so the character
becomes permanently unreadable rather than merely misread. This is the one check with no
acceptable failure threshold.
"""

from collections.abc import Sequence

from sentencepiece_ocr_guide.checks.protocols import Tokenizer
from sentencepiece_ocr_guide.checks.result import Check, CheckResult, Remedy, Severity

NAME = "no_unknown"


def no_unknown(samples: Sequence[str]) -> Check:
    """Assert no sample encodes to a sequence containing the unknown token."""

    def run(tokenizer: Tokenizer) -> CheckResult:
        if not samples:
            return CheckResult.skipped(NAME, "no samples supplied")

        failures = _samples_with_unknown(tokenizer, samples)
        if failures:
            return CheckResult.failed(
                NAME,
                f"{len(failures)} samples produced <unk> — those characters are unreadable",
                failures,
            )
        return CheckResult.passed(NAME, f"no <unk> across {len(samples)} samples")

    return Check(
        name=NAME,
        run=run,
        severity=Severity.BLOCKER,
        remedy=Remedy.RETRAIN_CONFIG,
    )


def _samples_with_unknown(tokenizer: Tokenizer, samples: Sequence[str]) -> list[str]:
    failures = []
    for sample in samples:
        unknown_positions = [
            index
            for index, token_id in enumerate(tokenizer.encode(sample))
            if tokenizer.is_unknown(token_id)
        ]
        if unknown_positions:
            failures.append(f"{sample!r} has <unk> at token index {unknown_positions}")
    return failures
