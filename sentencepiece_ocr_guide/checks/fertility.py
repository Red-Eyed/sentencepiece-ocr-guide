"""Tokens per character for one labelled group of text.

Failure mode #13 in docs/09-failure-modes.md: the vocab budget follows frequency, so an
under-represented script gets few merges and its common sequences fragment into many short
pieces. Fertility is how that becomes visible *before* training — a Devanagari group sitting at
1.0 tokens/char has learned no conjuncts at all, whatever the corpus proportions claimed.

Run one instance per script so the number means something; a single global average hides
exactly the imbalance you are looking for.
"""

from collections.abc import Sequence

from sentencepiece_ocr_guide.checks.protocols import Encoder
from sentencepiece_ocr_guide.checks.result import Check, CheckResult, Remedy, Severity


def fertility(label: str, samples: Sequence[str], max_tokens_per_char: float) -> Check:
    """Assert `label`'s text encodes at or below `max_tokens_per_char` tokens per character."""
    name = f"fertility[{label}]"

    def run(tokenizer: Encoder) -> CheckResult:
        if not samples:
            return CheckResult.skipped(name, f"no samples supplied for {label}")

        tokens, characters = _measure(tokenizer, samples)
        if characters == 0:
            return CheckResult.skipped(name, f"samples for {label} contain no characters")

        rate = tokens / characters
        summary = f"{label}: {rate:.2f} tokens/char ({tokens} tokens, {characters} chars)"
        if rate > max_tokens_per_char:
            return CheckResult.failed(
                name,
                f"{summary} — exceeds {max_tokens_per_char:.2f}, script is fragmenting",
            )
        return CheckResult.passed(name, summary)

    return Check(
        name=name,
        run=run,
        severity=Severity.MEDIUM,
        remedy=Remedy.FIX_CORPUS,
    )


def _measure(tokenizer: Encoder, samples: Sequence[str]) -> tuple[int, int]:
    tokens = sum(len(tokenizer.encode(sample)) for sample in samples)
    characters = sum(len(sample) for sample in samples)
    return tokens, characters
