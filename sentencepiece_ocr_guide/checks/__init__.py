"""Tokenizer validation checks.

Each check corresponds to a failure mode in docs/09-failure-modes.md that is detectable from
the tokenizer artifact alone — no training compute, no corpus, no model.
"""

from sentencepiece_ocr_guide.checks.byte_fallback import byte_fallback_rate
from sentencepiece_ocr_guide.checks.cross_script import cross_script_pieces
from sentencepiece_ocr_guide.checks.digit_pieces import digit_pieces
from sentencepiece_ocr_guide.checks.fertility import fertility
from sentencepiece_ocr_guide.checks.nfc import nfc_vocabulary
from sentencepiece_ocr_guide.checks.protected_symbols import protected_symbols
from sentencepiece_ocr_guide.checks.protocols import Encoder, Tokenizer, Vocabulary
from sentencepiece_ocr_guide.checks.result import Check, CheckResult, Report, Status
from sentencepiece_ocr_guide.checks.round_trip import round_trip
from sentencepiece_ocr_guide.checks.runner import run_checks
from sentencepiece_ocr_guide.checks.suite import standard_suite
from sentencepiece_ocr_guide.checks.unknown import no_unknown
from sentencepiece_ocr_guide.checks.whitespace import no_phantom_prefix

__all__ = [
    "Check",
    "CheckResult",
    "Encoder",
    "Report",
    "Status",
    "Tokenizer",
    "Vocabulary",
    "byte_fallback_rate",
    "cross_script_pieces",
    "digit_pieces",
    "fertility",
    "nfc_vocabulary",
    "no_phantom_prefix",
    "no_unknown",
    "protected_symbols",
    "round_trip",
    "run_checks",
    "standard_suite",
]
