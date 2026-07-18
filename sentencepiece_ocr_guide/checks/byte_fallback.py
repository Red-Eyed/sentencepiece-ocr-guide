"""Byte-fallback firing rate on representative text.

Byte fallback is the safety net that prevents `<unk>` — but a character that falls through it
costs 2–4 tokens the decoder must emit in the correct order, instead of one. A high rate on
ordinary text means `character_coverage` is too low for that script (failure mode #17 in
docs/09-failure-modes.md), which is a systematic accuracy cliff on exactly the rare characters
that matter for names and classical text.

Near zero is the target; the threshold is a tunable, not a law.
"""

from collections.abc import Sequence

from sentencepiece_ocr_guide.checks.protocols import Tokenizer
from sentencepiece_ocr_guide.checks.result import Check, CheckResult, Remedy, Severity

NAME = "byte_fallback_rate"
DEFAULT_MAX_RATE = 0.01


def byte_fallback_rate(samples: Sequence[str], max_rate: float = DEFAULT_MAX_RATE) -> Check:
    """Assert the share of byte-fallback tokens stays at or below `max_rate`."""

    def run(tokenizer: Tokenizer) -> CheckResult:
        if not samples:
            return CheckResult.skipped(NAME, "no samples supplied")

        byte_tokens, total_tokens = _count_byte_tokens(tokenizer, samples)
        if total_tokens == 0:
            return CheckResult.skipped(NAME, "samples produced no tokens")

        rate = byte_tokens / total_tokens
        summary = f"byte fallback fired on {rate:.2%} of tokens ({byte_tokens}/{total_tokens})"
        if rate > max_rate:
            return CheckResult.failed(NAME, summary, _worst_samples(tokenizer, samples))
        return CheckResult.passed(NAME, summary)

    return Check(
        name=NAME,
        run=run,
        severity=Severity.HIGH,
        remedy=Remedy.FIX_CORPUS,
    )


def _count_byte_tokens(tokenizer: Tokenizer, samples: Sequence[str]) -> tuple[int, int]:
    byte_tokens = 0
    total_tokens = 0
    for sample in samples:
        token_ids = tokenizer.encode(sample)
        total_tokens += len(token_ids)
        byte_tokens += sum(1 for token_id in token_ids if tokenizer.is_byte(token_id))
    return byte_tokens, total_tokens


def _worst_samples(tokenizer: Tokenizer, samples: Sequence[str]) -> list[str]:
    """The samples with the highest byte-fallback share, worst first — where to look."""
    rated = []
    for sample in samples:
        token_ids = tokenizer.encode(sample)
        if not token_ids:
            continue
        share = sum(1 for token_id in token_ids if tokenizer.is_byte(token_id)) / len(token_ids)
        if share > 0:
            rated.append((share, sample))

    rated.sort(reverse=True)
    return [f"{share:.0%} bytes: {sample!r}" for share, sample in rated]
