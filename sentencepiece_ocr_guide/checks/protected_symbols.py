"""Every `user_defined_symbols` entry must encode as exactly one token.

Failure mode #12's neighbour in docs/06-math-latex.md: a LaTeX command that fragments inflates
sequence length and multiplies the ways a decoder can emit an invalid command. Declaring a
symbol is not the same as it working — this check verifies the declaration took effect.
"""

from collections.abc import Sequence

from sentencepiece_ocr_guide.checks.protocols import Encoder
from sentencepiece_ocr_guide.checks.result import Check, CheckResult, Remedy, Severity

NAME = "protected_symbols"


def protected_symbols(symbols: Sequence[str]) -> Check:
    """Assert each declared symbol survives encoding as a single unsplit token."""

    def run(tokenizer: Encoder) -> CheckResult:
        if not symbols:
            return CheckResult.skipped(
                NAME, "no protected symbols supplied — pass the trainer's user_defined_symbols"
            )

        failures = _fragmented(tokenizer, symbols)
        if failures:
            return CheckResult.failed(
                NAME, f"{len(failures)} of {len(symbols)} protected symbols fragment", failures
            )
        return CheckResult.passed(NAME, f"all {len(symbols)} protected symbols are atomic")

    return Check(
        name=NAME,
        run=run,
        severity=Severity.HIGH,
        remedy=Remedy.RETRAIN_CONFIG,
    )


def _fragmented(tokenizer: Encoder, symbols: Sequence[str]) -> list[str]:
    failures = []
    for symbol in symbols:
        token_count = len(tokenizer.encode(symbol))
        if token_count != 1:
            failures.append(f"{symbol!r} splits into {token_count} tokens")
    return failures
