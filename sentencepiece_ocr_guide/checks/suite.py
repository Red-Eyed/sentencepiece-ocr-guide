"""The recommended suite — every artifact-inspectable failure mode from the guide.

Assembling the suite is deliberately separate from running it. Callers who want a different set
build their own tuple of `Check`s; adding a check to the guide means adding a module and one
line here, never editing an existing check.
"""

from collections.abc import Mapping, Sequence

from sentencepiece_ocr_guide.checks.byte_fallback import DEFAULT_MAX_RATE, byte_fallback_rate
from sentencepiece_ocr_guide.checks.cross_script import cross_script_pieces
from sentencepiece_ocr_guide.checks.digit_pieces import (
    DEFAULT_MAX_DIGIT_PIECE_LENGTH,
    digit_pieces,
)
from sentencepiece_ocr_guide.checks.fertility import fertility
from sentencepiece_ocr_guide.checks.nfc import nfc_vocabulary
from sentencepiece_ocr_guide.checks.protected_symbols import protected_symbols
from sentencepiece_ocr_guide.checks.result import Check
from sentencepiece_ocr_guide.checks.round_trip import round_trip
from sentencepiece_ocr_guide.checks.unknown import no_unknown
from sentencepiece_ocr_guide.checks.whitespace import no_phantom_prefix


def standard_suite(
    samples: Mapping[str, Sequence[str]],
    protected: Sequence[str] = (),
    max_digit_piece_length: int = DEFAULT_MAX_DIGIT_PIECE_LENGTH,
    max_byte_fallback_rate: float = DEFAULT_MAX_RATE,
    digits_are_a_script: bool = True,
    fertility_ceilings: Mapping[str, float] | None = None,
) -> tuple[Check, ...]:
    """Build the standard suite from grouped samples.

    `fertility_ceilings` is opt-in per group and has no defaults on purpose: what counts as
    acceptable fragmentation depends on your vocab size and script mix, and a threshold nobody
    chose is a threshold nobody should trust. Groups without a ceiling get no fertility check.
    """
    flat = _flatten(samples)

    return (
        round_trip(flat),
        no_unknown(flat),
        no_phantom_prefix(flat),
        nfc_vocabulary(),
        byte_fallback_rate(flat, max_byte_fallback_rate),
        protected_symbols(protected),
        digit_pieces(max_digit_piece_length),
        cross_script_pieces(digits_are_a_script),
        *_fertility_checks(samples, fertility_ceilings or {}),
    )


def _flatten(samples: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    return tuple(sample for group in samples.values() for sample in group)


def _fertility_checks(
    samples: Mapping[str, Sequence[str]], ceilings: Mapping[str, float]
) -> tuple[Check, ...]:
    return tuple(
        fertility(label, samples.get(label, ()), ceiling) for label, ceiling in ceilings.items()
    )
