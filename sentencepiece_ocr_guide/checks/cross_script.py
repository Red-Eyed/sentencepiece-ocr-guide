"""No vocabulary piece may straddle two writing systems.

Failure mode #11 in docs/09-failure-modes.md: a piece spanning Latin and Han spends a vocab slot
on a sequence that only occurs at incidental boundaries, and adds a confusable class. This is
what `split_by_unicode_script=True` is meant to prevent — this check confirms it did.
"""

from sentencepiece_ocr_guide.checks.piece_text import surface
from sentencepiece_ocr_guide.checks.protocols import Vocabulary
from sentencepiece_ocr_guide.checks.result import Check, CheckResult, Remedy, Severity
from sentencepiece_ocr_guide.checks.scripts import Script, scripts_in

NAME = "cross_script_pieces"


def cross_script_pieces(digits_are_a_script: bool = True) -> Check:
    """Assert no piece mixes scripts.

    `digits_are_a_script` treats a digit fusing with a letter as a cross-script merge — the
    case docs/03-configuration.md names explicitly. Set it False to allow digit-letter pieces
    and flag only merges between two writing systems.
    """

    def run(tokenizer: Vocabulary) -> CheckResult:
        offenders = _mixed_script_pieces(tokenizer, digits_are_a_script)
        if offenders:
            return CheckResult.failed(
                NAME, f"{len(offenders)} vocabulary pieces span more than one script", offenders
            )
        return CheckResult.passed(NAME, "no cross-script pieces in the vocabulary")

    return Check(
        name=NAME,
        run=run,
        severity=Severity.HIGH,
        remedy=Remedy.RETRAIN_CONFIG,
    )


def _mixed_script_pieces(tokenizer: Vocabulary, digits_are_a_script: bool) -> list[str]:
    offenders = []
    for token_id in range(len(tokenizer)):
        if _is_special(tokenizer, token_id):
            continue

        text = surface(tokenizer.piece(token_id))
        present = scripts_in(text)
        if not digits_are_a_script:
            present -= {Script.DIGIT}

        if len(present) > 1:
            names = "+".join(sorted(present))
            offenders.append(f"{text!r} spans {names}")
    return offenders


def _is_special(tokenizer: Vocabulary, token_id: int) -> bool:
    return (
        tokenizer.is_byte(token_id)
        or tokenizer.is_control(token_id)
        or tokenizer.is_unknown(token_id)
    )
