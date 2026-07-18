"""No digit-only vocabulary piece longer than the allowed length.

Failure mode #12 in docs/09-failure-modes.md: merged multi-digit tokens teach the model to
reproduce *frequent* numbers rather than read the digits on the page. On invoices, tables and
math — where the numbers are the entire point and are never the ones in the training corpus —
this is a direct accuracy loss on the highest-value content.
"""

from sentencepiece_ocr_guide.checks.piece_text import surface
from sentencepiece_ocr_guide.checks.protocols import Vocabulary
from sentencepiece_ocr_guide.checks.result import Check, CheckResult, Remedy, Severity

NAME = "digit_pieces"
DEFAULT_MAX_DIGIT_PIECE_LENGTH = 1


def digit_pieces(max_length: int = DEFAULT_MAX_DIGIT_PIECE_LENGTH) -> Check:
    """Assert no purely numeric piece exceeds `max_length` characters."""

    def run(tokenizer: Vocabulary) -> CheckResult:
        offenders = _long_digit_pieces(tokenizer, max_length)
        if offenders:
            return CheckResult.failed(
                NAME,
                f"{len(offenders)} digit-only pieces exceed {max_length} characters",
                offenders,
            )
        return CheckResult.passed(NAME, f"no digit-only piece longer than {max_length}")

    return Check(
        name=NAME,
        run=run,
        severity=Severity.HIGH,
        remedy=Remedy.RETRAIN_CONFIG,
    )


def _long_digit_pieces(tokenizer: Vocabulary, max_length: int) -> list[str]:
    offenders = []
    for token_id in range(len(tokenizer)):
        if _is_special(tokenizer, token_id):
            continue
        piece = tokenizer.piece(token_id)
        text = surface(piece).strip()
        if text.isdigit() and len(text) > max_length:
            # Report the raw piece: '250' and '▁250' are distinct entries in the vocabulary.
            offenders.append(repr(piece))
    return offenders


def _is_special(tokenizer: Vocabulary, token_id: int) -> bool:
    return (
        tokenizer.is_byte(token_id)
        or tokenizer.is_control(token_id)
        or tokenizer.is_unknown(token_id)
    )
