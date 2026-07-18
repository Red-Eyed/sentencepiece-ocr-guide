"""`spm-ocr` — the two checklists.

    spm-ocr corpus <files>                 before training: which axes vary, in which source
    spm-ocr model  <model>                 after training: the artifact checks
    spm-ocr all    <model> --corpus <files>  both, corpus findings first

The order is not cosmetic. Several model checks report defects that originate in the corpus —
`nfc_vocabulary` above all — so `all` runs the corpus scan first and says so when a model
failure needs a data fix rather than a retrain.
"""

import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, CliPositionalArg, CliSubCommand

from sentencepiece_ocr_guide.adapters.spm import SentencePieceTokenizer
from sentencepiece_ocr_guide.checks.result import Report, Severity
from sentencepiece_ocr_guide.checks.runner import run_checks
from sentencepiece_ocr_guide.checks.suite import standard_suite
from sentencepiece_ocr_guide.concurrency import default_workers
from sentencepiece_ocr_guide.corpus.canonicalize import canonicalizer
from sentencepiece_ocr_guide.corpus.discover import (
    Discovery,
    TextFile,
    discover_text_files,
    summarize,
)
from sentencepiece_ocr_guide.corpus.rewrite import RewriteRun, UndecodableLineError, rewrite_lines
from sentencepiece_ocr_guide.corpus.scan import scan_corpus
from sentencepiece_ocr_guide.progress import byte_progress
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
    jobs: int = Field(
        default_factory=default_workers,
        description="Worker threads. Defaults to one fewer than the machine's cores.",
    )

    def emit(self, report: Report) -> None:
        print(as_json(report) if self.json_output else as_text(report))


class CorpusChecklist(_Output):
    """Scan corpus files for encoding axes that vary between sources."""

    files: CliPositionalArg[list[Path]] = Field(
        description="Corpus files or directories (directories walked recursively for text files)"
    )

    def cli_cmd(self) -> None:
        report = _scan(self.files, self.jobs)
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
        corpus_report = _scan(self.corpus, self.jobs)
        model_report = self.run_checklist()
        combined = Report(results=corpus_report.results + model_report.results)

        self.emit(combined)
        raise SystemExit(exit_code(combined, self.fail_on))


class CanonicalizeCorpus(_Output):
    """Rewrite corpus files into canonical form, then verify the result.

    Never writes over the input unless asked: corpora are expensive to reassemble, and a
    canonicalizer configured with the wrong `--decide` axes is not obviously wrong afterwards.
    """

    files: CliPositionalArg[list[Path]] = Field(
        description="Corpus files or directories to canonicalize (directories walked recursively)"
    )
    out: Path | None = Field(
        default=None, description="Directory to write canonicalized copies into"
    )
    in_place: CliImplicitFlag[bool] = Field(
        default=False, description="Overwrite the input files instead of writing to --out."
    )
    decide: list[str] = Field(
        default_factory=list,
        description="DECIDE axes to apply, e.g. soft_hyphen_line_final. Measure first.",
    )
    drop_invalid: CliImplicitFlag[bool] = Field(
        default=False,
        description="Skip lines containing invalid UTF-8 instead of refusing. Loses data.",
    )

    def cli_cmd(self) -> None:
        destination = self._destination()
        canonicalize = canonicalizer(decide=tuple(self.decide))
        run = RewriteRun()

        discovery = _discover(self.files, self.jobs)
        written: list[Path] = []

        with byte_progress(_total_bytes(discovery.files), "canonicalizing") as track:
            for found in discovery.files:
                lines = track(_stream_lines(found.path))
                target = destination(found)
                written.append(
                    _canonicalize_file(found, lines, target, canonicalize, run, self.drop_invalid)
                )

        for source, tally in run.per_source.items():
            print(f"{source}: {tally.summary()}")

        # Re-scan the output: the invariant is only established if it is observed.
        report = _scan(written, self.jobs)
        print()
        self.emit(report)
        raise SystemExit(exit_code(report, self.fail_on))

    def _destination(self) -> Callable[[TextFile], Path]:
        if self.in_place == (self.out is not None):
            raise SystemExit("error: pass exactly one of --out DIR or --in-place")
        if self.in_place:
            return lambda found: found.path

        out = self.out
        assert out is not None  # guarded above

        def target(found: TextFile) -> Path:
            """Mirror the input tree, so a recursive run does not flatten shards together."""
            path = out / found.relative
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        return target


class SpmOcr(BaseSettings, cli_prog_name="spm-ocr", cli_kebab_case=True, populate_by_name=True):
    """Validate an OCR SentencePiece tokenizer and the corpus behind it."""

    corpus: CliSubCommand[CorpusChecklist]
    model: CliSubCommand[ModelChecklist]
    all: CliSubCommand[BothChecklists]
    canonicalize: CliSubCommand[CanonicalizeCorpus]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def _canonicalize_file(
    source: TextFile,
    lines: Iterator[str],
    target: Path,
    canonicalize: Callable[[str], str],
    run: RewriteRun,
    drop_invalid: bool,
) -> Path:
    """Write through a temporary file so a rejected line leaves no partial output behind.

    The input arrives as a stream rather than being opened here, so the caller decides what it
    is reading from — a plain file, or the same file behind a progress bar.
    """
    tally = run.tally_for(source.label)
    temporary = target.with_name(target.name + ".canonicalizing")

    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for line in rewrite_lines(lines, canonicalize, tally, source.label, drop_invalid):
                handle.write(line + "\n")
    except (UndecodableLineError, OSError) as error:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"error: {error}") from error

    temporary.replace(target)
    return target


def _discover(paths: list[Path], jobs: int) -> Discovery:
    """Expand paths to text files, reporting what was passed over."""
    discovery = discover_text_files(paths, workers=jobs)
    note = summarize(discovery.skipped)
    if note:
        print(note)
    return discovery


def _scan(paths: list[Path], jobs: int) -> Report:
    discovery = _discover(paths, jobs)
    with byte_progress(_total_bytes(discovery.files), "scanning") as track:
        return scan_corpus(
            {found.label: track(_stream_lines(found.path)) for found in discovery.files},
            workers=jobs,
        )


def _total_bytes(files: Iterable[TextFile]) -> int:
    """The size of everything about to be read, for a bar that can show a percentage.

    Above one worker this runs slightly ahead of what has been *counted* — the reader stays a
    bounded number of chunks in front of the pool — so the bar fills just before the report does.
    """
    return sum(found.path.stat().st_size for found in files)


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
