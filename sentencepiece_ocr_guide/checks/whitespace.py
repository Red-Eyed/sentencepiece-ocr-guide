"""No phantom leading space on text that does not begin with one.

`add_dummy_prefix=True` (the default) prepends a space marker to every line. Round-trip still
passes — decoding strips it again — so this corruption is invisible to the highest-value check
in the guide, and needs its own. For CJK and Thai, where lines carry no leading whitespace, it
puts a spurious token at the start of every single label. See docs/04-scripts.md.
"""

from collections.abc import Sequence

from sentencepiece_ocr_guide.checks.piece_text import has_space_marker
from sentencepiece_ocr_guide.checks.protocols import Tokenizer
from sentencepiece_ocr_guide.checks.result import Check, CheckResult, Remedy, Severity

NAME = "no_phantom_prefix"


def no_phantom_prefix(samples: Sequence[str]) -> Check:
    """Assert no sample gains a leading space it did not have."""

    def run(tokenizer: Tokenizer) -> CheckResult:
        candidates = [sample for sample in samples if sample and not sample[0].isspace()]
        if not candidates:
            return CheckResult.skipped(
                NAME, "no samples that begin with a non-whitespace character"
            )

        failures = _samples_with_phantom_prefix(tokenizer, candidates)
        if failures:
            return CheckResult.failed(
                NAME,
                f"{len(failures)} samples gained a leading space — add_dummy_prefix is on",
                failures,
            )
        return CheckResult.passed(NAME, f"no phantom prefix across {len(candidates)} samples")

    return Check(
        name=NAME,
        run=run,
        severity=Severity.HIGH,
        remedy=Remedy.RETRAIN_CONFIG,
    )


def _samples_with_phantom_prefix(tokenizer: Tokenizer, samples: Sequence[str]) -> list[str]:
    failures = []
    for sample in samples:
        token_ids = tokenizer.encode(sample)
        if not token_ids:
            continue
        first_piece = tokenizer.piece(token_ids[0])
        if has_space_marker(first_piece):
            failures.append(f"{sample!r} starts with piece {first_piece!r}")
    return failures
