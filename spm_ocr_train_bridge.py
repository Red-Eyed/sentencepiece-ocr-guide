"""Replay OCR trainer requests with explicit, editable SentencePiece defaults."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import sentencepiece as spm


def main() -> int:
    """Replay one request and record success or failure beside its artifacts."""
    if len(sys.argv) != 2:
        print(
            "usage: python -m spm_ocr_train_bridge <trainer_request.json>",
            file=sys.stderr,
        )
        return 2

    request_path = Path(sys.argv[1])
    request = _read_request(request_path)
    sentencepiece_args = _sentencepiece_args(request)
    output_path = Path(request["output"]["trainer_output"])
    started = time.monotonic()

    try:
        _train(sentencepiece_args)
    except Exception as error:
        _write_output(
            output_path,
            {
                "status": "failed",
                "model": request["output"]["model"],
                "vocab": request["output"]["vocab"],
                "elapsed_ms": _elapsed_ms(started),
                "error": str(error),
            },
        )
        raise

    _write_output(
        output_path,
        {
            "status": "succeeded",
            "model": request["output"]["model"],
            "vocab": request["output"]["vocab"],
            "elapsed_ms": _elapsed_ms(started),
        },
    )
    return 0


def _train(args: dict[str, Any]) -> None:
    """Train using request overrides and visible defaults for SentencePiece 0.2.2.

    Request values take precedence. Remaining options pass through to SentencePiece,
    which rejects unsupported names. Training writes model and vocabulary files.
    """
    args = dict(args)
    spm.SentencePieceTrainer.train(
        input=args.pop("input"),
        model_prefix=args.pop("model_prefix"),
        input_format=args.pop("input_format", ""),
        model_type=args.pop("model_type", "bpe"),
        vocab_size=args.pop("vocab_size", 8192),
        accept_language=args.pop("accept_language", []),
        self_test_sample_size=args.pop("self_test_sample_size", 0),
        enable_differential_privacy=args.pop("enable_differential_privacy", False),
        differential_privacy_noise_level=args.pop(
            "differential_privacy_noise_level", 0.0
        ),
        differential_privacy_clipping_threshold=args.pop(
            "differential_privacy_clipping_threshold", 0
        ),
        character_coverage=args.pop("character_coverage", 0.9998),
        input_sentence_size=args.pop("input_sentence_size", 20000000),
        shuffle_input_sentence=args.pop("shuffle_input_sentence", True),
        seed_sentencepiece_size=args.pop("seed_sentencepiece_size", 1000000),
        shrinking_factor=args.pop("shrinking_factor", 0.75),
        max_sentence_length=args.pop("max_sentence_length", 16384),
        num_threads=args.pop("num_threads", 16),
        num_sub_iterations=args.pop("num_sub_iterations", 2),
        max_sentencepiece_length=args.pop("max_sentencepiece_length", 8),
        split_by_unicode_script=args.pop("split_by_unicode_script", True),
        split_by_number=args.pop("split_by_number", True),
        split_by_whitespace=args.pop("split_by_whitespace", True),
        treat_whitespace_as_suffix=args.pop("treat_whitespace_as_suffix", False),
        allow_whitespace_only_pieces=args.pop("allow_whitespace_only_pieces", False),
        split_digits=args.pop("split_digits", True),
        pretokenization_delimiter=args.pop("pretokenization_delimiter", ""),
        control_symbols=args.pop("control_symbols", []),
        user_defined_symbols=args.pop("user_defined_symbols", []),
        required_chars=args.pop("required_chars", ""),
        byte_fallback=args.pop("byte_fallback", True),
        vocabulary_output_piece_score=args.pop("vocabulary_output_piece_score", True),
        hard_vocab_limit=args.pop("hard_vocab_limit", True),
        use_all_vocab=args.pop("use_all_vocab", False),
        unk_id=args.pop("unk_id", 0),
        bos_id=args.pop("bos_id", 1),
        eos_id=args.pop("eos_id", 2),
        pad_id=args.pop("pad_id", -1),
        unk_piece=args.pop("unk_piece", "<unk>"),
        bos_piece=args.pop("bos_piece", "<s>"),
        eos_piece=args.pop("eos_piece", "</s>"),
        pad_piece=args.pop("pad_piece", "<pad>"),
        unk_surface=args.pop("unk_surface", " ⁇ "),
        train_extremely_large_corpus=args.pop("train_extremely_large_corpus", True),
        normalization_rule_name=args.pop("normalization_rule_name", "identity"),
        normalization_rule_tsv=args.pop("normalization_rule_tsv", ""),
        denormalization_rule_tsv=args.pop("denormalization_rule_tsv", ""),
        add_dummy_prefix=args.pop("add_dummy_prefix", False),
        remove_extra_whitespaces=args.pop("remove_extra_whitespaces", False),
        escape_whitespaces=args.pop("escape_whitespaces", True),
        **args,
    )


def _read_request(path: Path) -> dict[str, Any]:
    """Load a serialized trainer request."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _sentencepiece_args(request: dict[str, Any]) -> dict[str, Any]:
    """Convert corpus part paths to SentencePiece's comma-separated input format."""
    args = dict(request["sentencepiece"])
    inputs = args.get("input")
    if isinstance(inputs, list):
        args["input"] = ",".join(str(path) for path in inputs)
    return args


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    """Write the trainer outcome, creating its parent directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _elapsed_ms(started: float) -> int:
    """Measure elapsed training time using the monotonic clock."""
    return round((time.monotonic() - started) * 1000)


if __name__ == "__main__":
    raise SystemExit(main())
