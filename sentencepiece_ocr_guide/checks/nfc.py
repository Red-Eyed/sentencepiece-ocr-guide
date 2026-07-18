"""Every vocabulary piece must already be in NFC.

Failure mode #4 in docs/09-failure-modes.md. A corpus mixing NFC and NFD trains a tokenizer
where `café` (U+00E9) and `café` (U+0065 U+0301) are unrelated token sequences for text that
renders identically — every affected grapheme trains at a fraction of its true frequency.

This check exists because round-trip cannot see the defect. With `identity` normalization both
forms round-trip perfectly; the tokenizer is faithfully reproducing an inconsistency it was
handed. The evidence is in the vocabulary instead: decomposed sequences in the corpus become
decomposed pieces.

The signal is one-directional. Non-NFC pieces prove the corpus contained decomposed text; their
absence is strong but not conclusive evidence that it did not.
"""

import unicodedata

from sentencepiece_ocr_guide.checks.protocols import Vocabulary
from sentencepiece_ocr_guide.checks.result import Check, CheckResult, Remedy, Severity

NAME = "nfc_vocabulary"


def nfc_vocabulary() -> Check:
    """Assert no vocabulary piece is in a decomposed form."""

    def run(tokenizer: Vocabulary) -> CheckResult:
        offenders = _decomposed_pieces(tokenizer)
        if offenders:
            return CheckResult.failed(
                NAME,
                f"{len(offenders)} vocabulary pieces are not NFC — the corpus mixed NFC and NFD",
                offenders,
            )
        return CheckResult.passed(NAME, "every vocabulary piece is in NFC")

    return Check(
        name=NAME,
        run=run,
        severity=Severity.BLOCKER,
        remedy=Remedy.FIX_CORPUS,
    )


def _decomposed_pieces(tokenizer: Vocabulary) -> list[str]:
    offenders = []
    for token_id in range(len(tokenizer)):
        if _is_special(tokenizer, token_id):
            continue
        piece = tokenizer.piece(token_id)
        composed = unicodedata.normalize("NFC", piece)
        if composed != piece:
            offenders.append(_describe(piece, composed))
    return offenders


def _describe(piece: str, composed: str) -> str:
    """Escaped, because the two forms are visually identical — that is the whole problem."""
    return f"{ascii(piece)} should be {ascii(composed)}"


def _is_special(tokenizer: Vocabulary, token_id: int) -> bool:
    return (
        tokenizer.is_byte(token_id)
        or tokenizer.is_control(token_id)
        or tokenizer.is_unknown(token_id)
    )
