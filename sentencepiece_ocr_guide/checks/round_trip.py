"""`decode(encode(x)) == x` — the highest-value check in the guide.

One assertion catches normalization mismatches, dropped characters and byte-fallback failures
together. See docs/08-validation.md and failure modes #2, #3, #4 in docs/09-failure-modes.md.
"""

from collections.abc import Sequence

from sentencepiece_ocr_guide.checks.protocols import Encoder
from sentencepiece_ocr_guide.checks.result import Check, CheckResult, Remedy, Severity

NAME = "round_trip"


def round_trip(samples: Sequence[str]) -> Check:
    """Assert every sample survives an encode/decode cycle byte-for-byte."""

    def run(tokenizer: Encoder) -> CheckResult:
        if not samples:
            return CheckResult.skipped(NAME, "no samples supplied")

        failures = _mismatches(tokenizer, samples)
        if failures:
            return CheckResult.failed(
                NAME,
                f"{len(failures)} of {len(samples)} samples did not survive encode/decode",
                failures,
            )
        return CheckResult.passed(NAME, f"all {len(samples)} samples round-tripped exactly")

    return Check(
        name=NAME,
        run=run,
        severity=Severity.BLOCKER,
        remedy=Remedy.RETRAIN_CONFIG,
    )


def _mismatches(tokenizer: Encoder, samples: Sequence[str]) -> list[str]:
    failures = []
    for sample in samples:
        decoded = tokenizer.decode(tokenizer.encode(sample))
        if decoded != sample:
            failures.append(f"{sample!r} -> {decoded!r}")
    return failures
