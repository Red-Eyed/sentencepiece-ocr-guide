"""Validate and replay OCR trainer requests without rebuilding the Rust executable.

Construct ``SentencePieceOptions`` for Python-only training. Its attribute docstrings
are included in ``model_json_schema()`` alongside defaults and allowed values.
The schema targets file-based training with the SentencePiece 0.2.2 wheel; corpus
feasibility and file access remain SentencePiece's responsibility.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Annotated, Literal, NotRequired, Self, TypedDict

import sentencepiece as spm
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _validate_path(value: object) -> object:
    """Reject empty path strings before pathlib converts them to the current directory."""
    match value:
        case "":
            raise ValueError("file paths must not be empty")
        case _:
            return value


NonemptyString = Annotated[str, Field(min_length=1)]
FilePath = Annotated[Path, Field(strict=False), BeforeValidator(_validate_path)]
_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    strict=True,
    frozen=True,
    revalidate_instances="always",
    validate_default=True,
    use_attribute_docstrings=True,
)


class SentencePieceOptions(BaseModel):
    """Validate file-based trainer options before invoking the native library.

    Defaults retain OCR normalization and an 8192-token BPE budget. Fields without
    an OCR-specific choice use upstream defaults. Unknown options and invalid
    combinations raise Pydantic validation errors before training starts.
    """

    model_config = _MODEL_CONFIG

    input: NonemptyString | Annotated[list[NonemptyString], Field(min_length=1)]
    """Corpus filename, comma-separated filenames, or a list of corpus filenames."""
    model_prefix: FilePath
    """Output path prefix for the generated .model and .vocab files."""
    input_format: Literal["", "text", "tsv"] = ""
    """Empty or text reads one sentence per line; tsv reads sentence-tab-frequency."""
    model_type: Literal["unigram", "bpe", "word", "char"] = "bpe"
    """Segmentation algorithm: Unigram, byte pair encoding, words, or characters."""
    vocab_size: int = Field(default=8192, ge=1, le=2147483647)
    """Total vocabulary budget, including special symbols and byte fallback tokens."""
    accept_language: list[NonemptyString] = Field(default_factory=list)
    """Language labels stored as metadata; they do not restrict training scripts."""
    self_test_sample_size: int = Field(default=0, ge=0, le=1000)
    """Number of corpus examples embedded for tokenizer self-testing; zero disables it."""
    enable_differential_privacy: bool = False
    """Apply the legacy differential privacy procedure to precomputed TSV counts."""
    differential_privacy_noise_level: float = Field(
        default=0.0, ge=0, allow_inf_nan=False
    )
    """Standard deviation of noise applied to counts when differential privacy is enabled."""
    differential_privacy_clipping_threshold: int = Field(
        default=0, ge=0, le=18446744073709551615
    )
    """Drop noisy counts below this threshold; zero disables count clipping."""
    character_coverage: float = Field(
        default=0.9998, ge=0.98, le=1, allow_inf_nan=False
    )
    """Fraction of corpus character occurrences covered by retained alphabet characters."""
    input_sentence_size: int = Field(default=20000000, ge=0, le=18446744073709551615)
    """Maximum sampled input lines; zero loads all lines, otherwise the value must exceed 100."""
    shuffle_input_sentence: bool = True
    """Use reservoir sampling for a positive input_sentence_size; false takes the first lines."""
    seed_sentencepiece_size: int = Field(default=1000000, ge=1, le=2147483647)
    """Initial Unigram candidate budget before pruning; unused by other model types."""
    seed_sentencepieces_file: str = ""
    """Optional seed-piece TSV filename, supported only for Unigram training."""
    shrinking_factor: float = Field(default=0.75, ge=0.5, le=0.95, allow_inf_nan=False)
    """Fraction of Unigram candidates retained at each pruning step."""
    max_sentence_length: int = Field(default=16384, ge=10, le=1073741824)
    """Maximum input line length in UTF-8 bytes; longer lines are skipped by SentencePiece."""
    num_threads: int = Field(default=16, ge=1, le=1024)
    """Number of native training worker threads."""
    num_sub_iterations: int = Field(default=2, ge=1, le=10)
    """Number of Unigram expectation-maximization iterations per pruning step."""
    max_sentencepiece_length: int = Field(default=8, ge=1, le=512)
    """Maximum learned piece length in Unicode characters; explicit symbols are exempt."""
    split_by_unicode_script: bool = True
    """Prevent merges across script boundaries; Han, hiragana, and katakana share a group."""
    split_by_number: bool = True
    """Prevent merges across transitions between digits and non-digits."""
    split_by_whitespace: bool = True
    """Prevent learned pieces from spanning whitespace-separated words."""
    treat_whitespace_as_suffix: bool = False
    """Attach whitespace markers to the end of pieces instead of the beginning."""
    allow_whitespace_only_pieces: bool = False
    """Allow learned pieces consisting entirely of whitespace markers."""
    split_digits: bool = True
    """Keep individual ASCII and fullwidth digits separate from learned multi-character pieces."""
    pretokenization_delimiter: str = ""
    """Optional training delimiter that prevents crossing merges; requires BPE or Unigram."""
    control_symbols: list[NonemptyString] = Field(default_factory=list)
    """Additional control tokens inserted by callers and omitted from decoded text."""
    user_defined_symbols: list[NonemptyString] = Field(default_factory=list)
    """Text strings recognized as indivisible pieces during encoding."""
    required_chars: str = ""
    """Characters forced into the alphabet regardless of the character coverage threshold."""
    byte_fallback: bool = True
    """Reserve 256 byte tokens to represent characters outside the retained alphabet."""
    vocabulary_output_piece_score: bool = True
    """Include each piece's score in the generated vocabulary file."""
    hard_vocab_limit: bool = True
    """Require the requested vocabulary size; character models always use a soft limit."""
    use_all_vocab: bool = False
    """Retain every observed word or character; valid only for word and char models."""
    unk_id: int = Field(default=0, ge=0, le=2147483647)
    """Required unknown-token ID; it cannot be disabled."""
    bos_id: int = Field(default=1, ge=-1, le=2147483647)
    """Beginning-of-sequence ID; -1 disables this token."""
    eos_id: int = Field(default=2, ge=-1, le=2147483647)
    """End-of-sequence ID; -1 disables this token."""
    pad_id: int = Field(default=-1, ge=-1, le=2147483647)
    """Padding-token ID; -1 disables this token."""
    unk_piece: NonemptyString = "<unk>"
    """Vocabulary spelling of the required unknown token."""
    bos_piece: NonemptyString = "<s>"
    """Vocabulary spelling of the beginning-of-sequence token."""
    eos_piece: NonemptyString = "</s>"
    """Vocabulary spelling of the end-of-sequence token."""
    pad_piece: NonemptyString = "<pad>"
    """Vocabulary spelling of the padding token."""
    unk_surface: str = " ⁇ "
    """Text emitted when decoding an unknown token."""
    train_extremely_large_corpus: bool = True
    """Use the native large-corpus Unigram mode; this does not bound corpus memory."""
    normalization_rule_name: Literal[
        "identity", "nmt_nfkc", "nmt_nfkc_cf", "nfkc", "nfkc_cf"
    ] = "identity"
    """Built-in normalization rule; cf variants also case-fold, while identity preserves characters."""
    normalization_rule_tsv: str = ""
    """Optional custom normalization mapping file, which takes precedence over the built-in rule."""
    denormalization_rule_tsv: str = ""
    """Optional mapping file applied to decoded text to reverse custom normalization."""
    add_dummy_prefix: bool = False
    """Insert a synthetic whitespace marker at the start, or end in suffix mode."""
    remove_extra_whitespaces: bool = False
    """Collapse repeated whitespace and trim boundary whitespace during normalization."""
    escape_whitespaces: bool = True
    """Represent normalized spaces using the SentencePiece whitespace marker; training requires true."""
    minloglevel: Literal[0, 1, 2, 3] = 0
    """Native logging threshold: info, warning, error, or fatal; applied process-wide."""

    @field_validator("input_sentence_size")
    @classmethod
    def validate_sample_size(cls, value: int) -> int:
        """Reject nonzero sample limits that the native trainer cannot accept."""
        if 0 < value <= 100:
            raise ValueError("input_sentence_size must be zero or greater than 100")
        return value

    @field_validator("minloglevel", mode="before")
    @classmethod
    def validate_log_level(cls, value: object) -> object:
        """Require integer logging levels without boolean or float coercion."""
        match value:
            case bool():
                raise ValueError("minloglevel must be an integer logging level")
            case int():
                return value
            case _:
                raise ValueError("minloglevel must be an integer logging level")

    @model_validator(mode="after")
    def validate_model_options(self) -> Self:
        """Reject incompatible model-specific settings before native training."""
        if self.use_all_vocab and self.model_type not in ("word", "char"):
            raise ValueError("use_all_vocab requires a word or char model")
        if self.seed_sentencepieces_file and self.model_type != "unigram":
            raise ValueError("seed_sentencepieces_file requires a unigram model")
        if self.pretokenization_delimiter and self.model_type not in ("unigram", "bpe"):
            raise ValueError(
                "pretokenization_delimiter requires a unigram or bpe model"
            )
        if not self.escape_whitespaces:
            raise ValueError("SentencePiece training requires escape_whitespaces=True")
        return self

    @model_validator(mode="after")
    def validate_special_tokens(self) -> Self:
        """Check enabled special token IDs and spellings."""
        enabled = self._enabled_special_tokens()
        ids = [index for index, _ in enabled]
        pieces = [piece for _, piece in enabled]
        if len(ids) != len(set(ids)):
            raise ValueError("enabled special-token IDs must be distinct")
        if any(index >= self.vocab_size for index in ids):
            raise ValueError(
                "enabled special-token IDs must be smaller than vocab_size"
            )
        if len(pieces) != len(set(pieces)):
            raise ValueError("enabled special-token spellings must be distinct")
        return self

    def _enabled_special_tokens(self) -> list[tuple[int, str]]:
        """Pair each enabled built-in token's ID with its vocabulary spelling."""
        tokens = [
            (self.unk_id, self.unk_piece),
            (self.bos_id, self.bos_piece),
            (self.eos_id, self.eos_piece),
            (self.pad_id, self.pad_piece),
        ]
        return [(index, piece) for index, piece in tokens if index >= 0]

    @model_validator(mode="after")
    def validate_explicit_symbols(self) -> Self:
        """Check explicit symbol collisions and the minimum reserved vocabulary budget."""
        pieces = [piece for _, piece in self._enabled_special_tokens()]
        symbols = self.control_symbols + self.user_defined_symbols
        if len(symbols) != len(set(symbols)):
            raise ValueError("control and user-defined symbols must be distinct")
        if self.unk_piece in symbols:
            raise ValueError("unk_piece cannot be a control or user-defined symbol")
        if self.byte_fallback:
            byte_pieces = {f"<0x{byte:02X}>" for byte in range(256)}
            if byte_pieces.intersection(pieces + symbols):
                raise ValueError(
                    "special and explicit symbols cannot duplicate byte fallback tokens"
                )
        reserved = len(set(pieces + symbols)) + (256 if self.byte_fallback else 0)
        if reserved > self.vocab_size:
            raise ValueError(f"vocab_size must fit at least {reserved} reserved tokens")
        return self


class TrainerOutputPaths(BaseModel):
    """Artifact paths in the existing Rust-to-Python request format."""

    model_config = _MODEL_CONFIG

    model: FilePath
    """Expected trained model file path."""
    vocab: FilePath
    """Expected trained vocabulary file path."""
    trainer_output: FilePath
    """Destination JSON file for the trainer status and elapsed time."""


class TrainerRequest(BaseModel):
    """Validate the existing trainer request envelope and its nested options."""

    model_config = _MODEL_CONFIG

    sentencepiece: SentencePieceOptions
    """Explicit trainer settings; omitted settings use the documented Python defaults."""
    output: TrainerOutputPaths
    """Locations of the trained artifacts and status report."""


class TrainerOutcome(TypedDict):
    """JSON status record consumed by the existing Rust caller."""

    status: Literal["succeeded", "failed"]
    model: str
    vocab: str
    elapsed_ms: int
    error: NotRequired[str]


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
    output_path = request.output.trainer_output
    started = time.monotonic()

    try:
        _train(request.sentencepiece)
    except Exception as error:
        _write_output(
            output_path,
            {
                "status": "failed",
                "model": str(request.output.model),
                "vocab": str(request.output.vocab),
                "elapsed_ms": _elapsed_ms(started),
                "error": str(error),
            },
        )
        raise

    _write_output(
        output_path,
        {
            "status": "succeeded",
            "model": str(request.output.model),
            "vocab": str(request.output.vocab),
            "elapsed_ms": _elapsed_ms(started),
        },
    )
    return 0


def _train(options: SentencePieceOptions | Mapping[str, object]) -> None:
    """Write model and vocabulary files using validated, explicit trainer options.

    Native training errors propagate to the caller. No corpus data is loaded by
    this adapter; SentencePiece owns corpus loading and sampling.
    """
    options = SentencePieceOptions.model_validate(options)
    match options.input:
        case str(input_files):
            inputs = input_files
        case list(input_files):
            inputs = ",".join(input_files)
    spm.SentencePieceTrainer.train(
        input=inputs,
        model_prefix=str(options.model_prefix),
        input_format=options.input_format,
        model_type=options.model_type,
        vocab_size=options.vocab_size,
        accept_language=options.accept_language,
        self_test_sample_size=options.self_test_sample_size,
        enable_differential_privacy=options.enable_differential_privacy,
        differential_privacy_noise_level=options.differential_privacy_noise_level,
        differential_privacy_clipping_threshold=options.differential_privacy_clipping_threshold,
        character_coverage=options.character_coverage,
        input_sentence_size=options.input_sentence_size,
        shuffle_input_sentence=options.shuffle_input_sentence,
        seed_sentencepiece_size=options.seed_sentencepiece_size,
        shrinking_factor=options.shrinking_factor,
        max_sentence_length=options.max_sentence_length,
        num_threads=options.num_threads,
        num_sub_iterations=options.num_sub_iterations,
        max_sentencepiece_length=options.max_sentencepiece_length,
        split_by_unicode_script=options.split_by_unicode_script,
        split_by_number=options.split_by_number,
        split_by_whitespace=options.split_by_whitespace,
        treat_whitespace_as_suffix=options.treat_whitespace_as_suffix,
        allow_whitespace_only_pieces=options.allow_whitespace_only_pieces,
        split_digits=options.split_digits,
        pretokenization_delimiter=options.pretokenization_delimiter,
        control_symbols=options.control_symbols,
        user_defined_symbols=options.user_defined_symbols,
        required_chars=options.required_chars,
        byte_fallback=options.byte_fallback,
        vocabulary_output_piece_score=options.vocabulary_output_piece_score,
        hard_vocab_limit=options.hard_vocab_limit,
        use_all_vocab=options.use_all_vocab,
        unk_id=options.unk_id,
        bos_id=options.bos_id,
        eos_id=options.eos_id,
        pad_id=options.pad_id,
        unk_piece=options.unk_piece,
        bos_piece=options.bos_piece,
        eos_piece=options.eos_piece,
        pad_piece=options.pad_piece,
        unk_surface=options.unk_surface,
        train_extremely_large_corpus=options.train_extremely_large_corpus,
        normalization_rule_name=options.normalization_rule_name,
        normalization_rule_tsv=options.normalization_rule_tsv,
        denormalization_rule_tsv=options.denormalization_rule_tsv,
        add_dummy_prefix=options.add_dummy_prefix,
        remove_extra_whitespaces=options.remove_extra_whitespaces,
        escape_whitespaces=options.escape_whitespaces,
        seed_sentencepieces_file=options.seed_sentencepieces_file,
        minloglevel=options.minloglevel,
    )


def _read_request(path: Path) -> TrainerRequest:
    """Parse and validate JSON once before any native training starts."""
    return TrainerRequest.model_validate_json(path.read_bytes())


def _write_output(path: Path, payload: TrainerOutcome) -> None:
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
