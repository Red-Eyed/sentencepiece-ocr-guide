"""`spm-ocr` — the two checklists.

    spm-ocr corpus <files>                 before training: which axes vary, in which source
    spm-ocr model  <model>                 after training: the artifact checks
    spm-ocr all    <model> --corpus <files>  both, corpus findings first

The order is not cosmetic. Several model checks report defects that originate in the corpus —
`nfc_vocabulary` above all — so `all` runs the corpus scan first and says so when a model
failure needs a data fix rather than a retrain.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, CliPositionalArg, CliSubCommand

from sentencepiece_ocr_guide.adapters.spm import SentencePieceTokenizer
from sentencepiece_ocr_guide.checks.result import Report, Severity
from sentencepiece_ocr_guide.checks.runner import run_checks
from sentencepiece_ocr_guide.checks.suite import standard_suite
from sentencepiece_ocr_guide.corpus.scan import scan_corpus
from sentencepiece_ocr_guide.report import as_json, as_text, exit_code
from sentencepiece_ocr_guide.samples import DEFAULT_SAMPLES


class _Output(BaseModel):
    """Options shared by every subcommand."""

    json_output: CliImplicitFlag[bool] = Field(
        default=False, alias="json", description="Emit the report as JSON instead of text."
    )
    fail_on: Severity = Field(
        default=Severity.HIGH,
        description="Exit non-zero when a failure reaches this severity.",
    )

    def emit(self, report: Report) -> None:
        print(as_json(report) if self.json_output else as_text(report))


class CorpusChecklist(_Output):
    """Scan corpus files for encoding axes that vary between sources."""

    files: CliPositionalArg[list[Path]] = Field(description="Corpus text files, one line each")

    def cli_cmd(self) -> None:
        report = _scan(self.files)
        self.emit(report)
        raise SystemExit(exit_code(report, self.fail_on))


class ModelChecklist(_Output):
    """Run the artifact checks against a trained SentencePiece model."""

    model: CliPositionalArg[Path] = Field(description="Path to the trained .model file")
    samples: Path | None = Field(
        default=None,
        description="Text file, one sample per line. Defaults to the built-in stratified set.",
    )
    symbols: Path | None = Field(
        default=None,
        description="Text file of user_defined_symbols, one per line, to verify stay atomic.",
    )
    max_digit_piece_length: int = Field(
        default=1, description="Longest permitted digit-only vocabulary piece."
    )
    max_byte_fallback_rate: float = Field(
        default=0.01, description="Maximum share of tokens allowed to be byte-fallback pieces."
    )
    allow_digit_letter_pieces: CliImplicitFlag[bool] = Field(
        default=False,
        description="Do not treat a digit fused with a letter as a cross-script merge.",
    )

    def cli_cmd(self) -> None:
        report = self.run_checklist()
        self.emit(report)
        raise SystemExit(exit_code(report, self.fail_on))

    def run_checklist(self) -> Report:
        return run_checks(
            standard_suite(
                samples=self._sample_groups(),
                protected=_read_lines(self.symbols),
                max_digit_piece_length=self.max_digit_piece_length,
                max_byte_fallback_rate=self.max_byte_fallback_rate,
                digits_are_a_script=not self.allow_digit_letter_pieces,
            ),
            SentencePieceTokenizer.from_file(self.model),
        )

    def _sample_groups(self) -> dict[str, tuple[str, ...]]:
        if self.samples is None:
            return dict(DEFAULT_SAMPLES)
        return {"supplied": _read_lines(self.samples)}


class BothChecklists(ModelChecklist):
    """Run the corpus checklist, then the model checklist."""

    corpus: list[Path] = Field(default_factory=list, description="Corpus files to scan first")

    def cli_cmd(self) -> None:
        corpus_report = _scan(self.corpus)
        model_report = self.run_checklist()
        combined = Report(results=corpus_report.results + model_report.results)

        self.emit(combined)
        raise SystemExit(exit_code(combined, self.fail_on))


class SpmOcr(BaseSettings, cli_prog_name="spm-ocr", cli_kebab_case=True, populate_by_name=True):
    """Validate an OCR SentencePiece tokenizer and the corpus behind it."""

    corpus: CliSubCommand[CorpusChecklist]
    model: CliSubCommand[ModelChecklist]
    all: CliSubCommand[BothChecklists]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def _scan(files: list[Path]) -> Report:
    return scan_corpus({path.name: _stream_lines(path) for path in files})


def _stream_lines(path: Path) -> Iterator[str]:
    """Yield a corpus file line by line, never holding more than one line in memory.

    Decoding uses `surrogateescape` rather than the default strict mode. A corpus assembled from
    legacy extractors frequently contains bytes that are not valid UTF-8, and a tool whose
    purpose is finding encoding defects must report that rather than dying on it. The escaped
    bytes survive as lone surrogates, which is exactly what the scanner detects.
    """
    with path.open("r", encoding="utf-8", errors="surrogateescape") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if stripped.strip():
                yield stripped


def _read_lines(path: Path | None) -> tuple[str, ...]:
    """Read a small file whole — sample and symbol lists, which are iterated more than once."""
    if path is None:
        return ()
    with path.open("r", encoding="utf-8", errors="surrogateescape") as handle:
        return tuple(line.rstrip("\n") for line in handle if line.strip())


def main() -> None:
    CliApp.run(SpmOcr)


if __name__ == "__main__":
    sys.exit(main())
