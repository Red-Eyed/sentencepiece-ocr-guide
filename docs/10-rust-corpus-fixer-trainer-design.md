# Rust corpus fixer and SentencePiece trainer

## Metadata

| Field | Value |
|---|---|
| Author | Vadym Stupakov <vadim.stupakov@gmail.com> |
| Date created | 2026-07-30 |
| Status | Draft |
| Authoritative URL | `docs/10-rust-corpus-fixer-trainer-design.md` |

## Objective

Build a Rust command-line program that reads `cfg.json`, canonicalizes and balances a text-file
OCR corpus according to the failure-mode guidance, trains SentencePiece on the resulting corpus,
and emits enough reports to catch silent tokenizer defects before OCR model training.

## Background

The existing guide treats the tokenizer as part of the OCR label space, not as a generic language
model preprocessing tool. That changes the priorities:

- Corpus text must be normalized once at ingestion, then SentencePiece must use
  `normalization_rule_name=identity`.
- Mixed corpus encodings are the most probable defect. Round-trip validation does not catch
  this because both spellings decode faithfully; vocabulary inspection is needed after training.
- Compatibility folding such as blanket NFKC is too lossy for faithful OCR ground truth.
- Balancing happens while assembling the training corpus. SentencePiece's own sampling controls
  are RAM controls, not fairness controls.
- SentencePiece training is still the reference implementation. Current Rust crates expose model
  loading, encoding, decoding, and normalization APIs, but not an equivalent trainer surface.
  The Python `sentencepiece` package does expose `SentencePieceTrainer.train`, and this
  repository already declares that package dependency.

The program should therefore make corpus canonicalization and reporting the hard center of the
tool, and keep SentencePiece training behind a replaceable adapter.

The corpus should be treated as an IRL delivery, not a clean benchmark folder. The tool must
expect unreliable filenames, missing source metadata, duplicate vendor drops, archives inside
archives, compressed text files, binary sidecars, mixed Unicode encodings, extractor artifacts,
and badly imbalanced scripts. Directory structure can be kept as provenance, but correctness
must come from content sniffing and typed repair policy rather than user pre-sorting.

## Goals

- Accept a small JSON config that defines corpus location, output location, and preset choice.
  The heavier canonicalization, balancing, validation, and SentencePiece trainer defaults come
  from `cfg.json.ocr`.
- Stream input text files, canonicalize each line, write a fixed training corpus, and produce a
  pre/post axis report for the transformations in `09-failure-modes.md`.
- Keep visible feedback on screen for every long stage: progress bars when totals are known,
  spinners with counters when totals are unknown, and status updates while the Python trainer is
  running.
- Train SentencePiece using conservative OCR defaults: BPE, byte fallback, identity
  normalization, no dummy prefix, preserved whitespace, script splitting, digit splitting, short
  learned pieces, and a raised max sentence length.
- Refuse dangerous configs by default, especially `byte_fallback=false`,
  `normalization_rule_name != "identity"`, `split_digits=false`, or
  `remove_extra_whitespaces=true`.
- Emit machine-readable reports and a manifest that bind together config hash,
  canonicalizer version, fixed corpus hash, Python trainer request, model hash, and validation
  findings.

## Non-goals

- Do not implement SentencePiece training from scratch in Rust.
- Do not make a TUI the primary workflow. A TUI can inspect reports later; batch training should
  stay scriptable.
- Do not infer visual truth from glyph appearance. Homoglyph folding is intentionally out of
  scope.
- Do not solve OCR image preprocessing, model training, or CER evaluation.
- Do not silently rewrite the input corpus in place.

## Decision

Use a CLI-first Rust program named `spm-ocr` with one public command:

```text
spm-ocr train --config cfg.json
```

`train` runs the complete pipeline: scan raw corpus, fix corpus, balance and assemble the
training file, train SentencePiece, and validate the resulting model. Corpus defects are fixed
when the policy is lossless, skipped when a line cannot be safely repaired, and logged to a
JSONL issue file. Validation findings are reported; they do not turn a completed run into a
failed command.

The MVP calls the Python `sentencepiece` package through a `Trainer` trait. Rust writes a typed
`trainer_request.json`, invokes a tiny Python bridge, and reads a typed `trainer_output.json`.
This keeps the training boundary stable, avoids a large C++ FFI surface in v1, and still lets a
future backend switch to `spm_train`, C++ FFI, or a native Rust trainer without changing corpus
logic.

## Config Format

The program accepts JSON because the requested input is `cfg.json`. Unknown fields are errors so
typos cannot silently change a training run.

The normal user config should be small:

```json
{
  "preset": "ocr_multilingual",
  "corpus": {
    "path": "data/raw-corpus"
  },
  "output": {
    "work_dir": "runs/ocr-spm-v1",
    "model_prefix": "ocr_tokenizer"
  }
}
```

The bundled `cfg.json.ocr` file defines named presets:

```json
{
  "version": 1,
  "default": "ocr_multilingual",
  "presets": {
    "ocr_multilingual": {
      "description": "General multilingual OCR tokenizer preset.",
      "canonicalization": {},
      "balancing": {},
      "sentencepiece": {},
      "validation": {}
    },
    "ocr_cjk_heavy": {
      "extends": "ocr_multilingual",
      "sentencepiece": {
        "vocab_size": 48000,
        "character_coverage": 0.99995,
        "max_sentencepiece_length": 4
      }
    },
    "ocr_math_heavy": {
      "extends": "ocr_multilingual",
      "sentencepiece": {
        "vocab_size": 48000,
        "max_sentence_length": 32768
      }
    }
  }
}
```

After preset expansion, the effective config contains the full policy:

```json
{
  "canonicalization": {
    "unicode_form": "nfc",
    "strip": ["bom", "zero_width_space"],
    "map_nbsp_to_space": true,
    "fold_arabic_presentation_forms": true,
    "soft_hyphen": "line_final_to_hyphen_midline_strip",
    "preserve_zwj_zwnj": true,
    "preserve_compatibility_chars": true
  },
  "balancing": {
    "enabled": true,
    "mode": "conservative",
    "total_lines": 20000000,
    "alpha": 0.7,
    "hierarchy": ["domain", "script", "language_hint", "source_group", "length_bin"],
    "min_keep_fraction": 0.5,
    "max_downsample_ratio": 4.0,
    "collapse_buckets_below_lines": 1000,
    "max_part_lines": 1000000,
    "shuffle_seed": 1337
  },
  "sentencepiece": {
    "trainer": "python_sentencepiece",
    "python": {
      "runner": "uv",
      "args": ["run", "python"],
      "module": "spm_ocr_train_bridge"
    },
    "model_type": "bpe",
    "vocab_size": 40000,
    "character_coverage": 0.9998,
    "byte_fallback": true,
    "normalization_rule_name": "identity",
    "add_dummy_prefix": false,
    "remove_extra_whitespaces": false,
    "split_by_unicode_script": true,
    "split_by_whitespace": true,
    "split_digits": true,
    "max_sentencepiece_length": 8,
    "max_sentence_length": 16384,
    "input_sentence_size": 20000000,
    "shuffle_input_sentence": true,
    "train_extremely_large_corpus": true,
    "user_defined_symbols": ["\\frac", "\\sqrt", "\\sum", "\\int", "^", "_", "{", "}"],
    "num_threads": 16
  },
  "validation": {
    "mode": "report",
    "line_policy": "fix_or_skip",
    "issue_log": "reports/corpus_issues.jsonl",
    "include_line_text_in_log": false,
    "round_trip_sample_per_bucket": 1000
  }
}
```

## Pipeline

### 1. Load and validate config

`RawConfig` is deserialized with strict Rust types, merged with the selected preset from
`cfg.json.ocr`, and converted into `EffectiveConfig`. Paths become `PathBuf`, trainer settings
become typed enums or bounded numeric wrappers, and date/time-like values are not accepted as
strings unless they are parsed at the boundary.

Preset merge rules are simple:

- `preset` defaults to `cfg.json.ocr.default`.
- A preset can `extend` one parent preset.
- User config wins over preset fields.
- List-valued fields replace the parent list rather than appending to it, so
  `user_defined_symbols` stays predictable.
- The expanded `EffectiveConfig` is written to `work_dir/effective_config.json`.

Dangerous SentencePiece settings are repaired to the selected preset's safe value and logged in
`reports/corpus_issues.jsonl`. The command fails only when the config cannot be parsed, a path
cannot be accessed, an output cannot be written, the Python trainer fails, or an internal error
occurs.

| Setting | Required default |
|---|---|
| `model_type` | `bpe` |
| `byte_fallback` | `true` |
| `normalization_rule_name` | `identity` |
| `add_dummy_prefix` | `false` |
| `remove_extra_whitespaces` | `false` |
| `split_by_unicode_script` | `true` |
| `split_digits` | `true` |
| `max_sentencepiece_length` | preset value |
| `max_sentence_length` | preset value; over-limit lines are skipped and logged |

### 2. Scan raw corpus

The scanner accepts `corpus.path` as either one file or a directory. Discovery is based on
content, not filename: it sniffs magic bytes for archives and compressed streams, recursively
unpacks nested containers into `work_dir/unpacked/`, and treats UTF-8 text-like payloads as
corpus sources regardless of extension. Unsupported files and broken archives are skipped and
logged. Text sources are streamed, line endings are normalized to LF in the fixed corpus, and
the scanner records:

- line count, byte count, char count, and max line length;
- source attribution from file path;
- script/domain bucket estimates;
- line counts affected by each canonicalization axis;
- soft-hyphen counts split by line-final and mid-line position;
- preserved compatibility rows such as fullwidth forms, ligatures, dashes, quotes, ZWJ, and ZWNJ.

This produces `scan.raw.json` and `scan.raw.md`. The report intentionally includes rows that
must remain non-zero, because preserving a distinction is also a decision.

### 3. Canonicalize

Every line passes through one chokepoint. Repairable issues are fixed in place for the fixed
corpus; unrepairable issues are skipped and logged with file path, line number, issue id, and
reason:

```text
strip BOM and ZWSP
map NBSP to U+0020
fold Arabic presentation forms only
handle soft hyphen according to configured positional policy
normalize to NFC
assert canonicalize(line) == line after transformation
```

The canonicalizer preserves fullwidth forms, ligatures, Arabic-Indic digits, quotes, dashes,
U+3000, ZWJ, ZWNJ, and cross-script homoglyphs.

Examples of skipped lines:

- invalid UTF-8 byte sequence;
- line still not idempotent after canonicalization;
- line exceeds the configured `max_sentence_length` and cannot be chunked safely;
- empty or whitespace-only line, if the preset disables whitespace-only examples.

The repaired audit stream is written to `work_dir/fixed_corpus.txt`; training inputs are
written as named corpus parts under `work_dir/train_corpus/`. The raw corpus is never
modified.

### 4. Balance and assemble

If balancing is enabled, the program classifies canonicalized lines by multiple weak signals:
domain, dominant Unicode script, ISO-like language hints in paths such as `en`, `eng`, `es`,
`spa`, `ko`, or `kor`, source group, and line-length bin. Paths are treated as provenance, not
truth: the script is discovered from text content, and path language hints only become one
bucket feature among several. It computes target counts using exponential smoothing:

```text
P'(bucket) proportional to P(bucket)^alpha
```

Small buckets are fully retained. Large buckets are downsampled with a seeded streaming
reservoir, bounded by `min_keep_fraction` and `max_downsample_ratio` so balancing remains
conservative. The assembled training corpus is split into named part files such as
`train-text-script_latin-lang_es-source_books-len_normal-part_0001-a1b2c3d4.txt`, so operators
can inspect what each file means without decoding numeric ids.

The output report includes actual versus target counts. If the final distribution misses
targets beyond configured tolerance, training stops.

### 5. Train SentencePiece

The trainer adapter receives a fully materialized `TrainerRequest`:

```rust
trait Trainer {
    fn train(&self, request: &TrainerRequest) -> Result<TrainerOutput, TrainError>;
}
```

The initial implementation, `PythonSentencePieceTrainer`, writes this request as JSON and spawns
the configured Python runner:

```text
uv run python -m spm_ocr_train_bridge runs/ocr-spm-v1/trainer_request.json
```

The Python bridge is deliberately thin:

```python
import json
import sentencepiece as spm

request = json.load(open(request_path, encoding="utf-8"))
spm.SentencePieceTrainer.train(**request["sentencepiece"])
```

The real bridge should also write `trainer_output.json` with artifact paths, package version,
stdout/stderr summaries, and elapsed time. It should not inspect corpus text and should not
decide tokenizer policy. Rust remains responsible for validation and for generating a safe
trainer request.

The program writes `trainer_request.json` before execution so the exact Python call can be
reviewed and replayed.

### 6. Validate model and corpus together

After training, the validation stage inspects `.model`, `.vocab`, and the fixed corpus:

- no `<unk>` on sampled corpus encoding path;
- model trainer settings match the config;
- model normalizer settings match identity/no dummy prefix/preserve whitespace;
- no non-NFC vocabulary pieces;
- no Arabic presentation forms in corpus or vocab;
- no cross-script pieces;
- no digit-only pieces longer than one digit by default;
- no phantom leading prefix on non-whitespace-leading text;
- round-trip exactness on stratified samples;
- byte-fallback rate by script/domain;
- long-line drop estimate using configured `max_sentence_length`;
- user-defined symbols encode atomically inside real examples.

Findings are emitted as structured records:

```json
{
  "id": "non_nfc_vocabulary",
  "severity": "warning",
  "action": "reported",
  "remedy": "retrain_from_training_corpus",
  "message": "3 vocabulary pieces are not NFC",
  "examples": ["café"]
}
```

Corpus repair findings use the same schema with `action` set to `fixed` or `skipped` and are
also appended to `reports/corpus_issues.jsonl` as the run proceeds.

## Progress Feedback

The command must never leave the terminal in an undefined waiting state.

- File discovery uses a spinner with discovered-file and elapsed-time counters.
- Raw scanning uses a progress bar when file sizes are known, otherwise a spinner with lines,
  bytes, and files processed.
- Canonicalization uses a progress bar and live counters for fixed and skipped lines.
- Balancing reports bucket counts and sampling progress.
- Python SentencePiece training uses a spinner that shows elapsed time and the latest captured
  trainer status line.
- Validation uses a progress bar over checks and sampled lines.
- Every progress UI is backed by `indicatif`; terminal styling uses `console`; structured logs
  use `tracing`.
- `--json` disables decorative progress on stdout and sends progress/status messages to stderr
  while keeping stdout machine-readable.

## Architecture

```text
cfg.json
  |
  v
ConfigLoader -> PresetResolver -> PolicyValidator -> EffectiveConfig
  |
  v
CorpusWalker -> AxisScanner -> RawScanReport
  |
  v
Canonicalizer -> BucketClassifier -> Balancer -> train_corpus/*.txt
  |
  v
Trainer adapter -> Python bridge -> ocr_tokenizer.model + ocr_tokenizer.vocab
  |
  v
ModelInspector + RuntimeTokenizer -> ValidationReport + manifest.json
```

Suggested crate layout:

```text
src/
  main.rs
  cli.rs
  config.rs
  corpus/
    mod.rs
    walk.rs
    line.rs
    bucket.rs
  normalize/
    mod.rs
    canonicalize.rs
    axis.rs
  balance/
    mod.rs
    smoothing.rs
    reservoir.rs
  spm/
    mod.rs
    trainer.rs
    inspect.rs
    args.rs
  validate/
    mod.rs
    findings.rs
    corpus_checks.rs
    model_checks.rs
    roundtrip.rs
  report/
    mod.rs
    json.rs
    markdown.rs
```

## SOLID Checklist

- S: Each module owns one reason to change: config, scanning, canonicalization, balancing,
  training, validation, or reporting.
- O: Trainer backends are added by implementing `Trainer`; validation checks are added as new
  check units returning `Finding` values rather than threading observer callbacks through core
  functions.
- L: Every `Trainer` implementation must return structured failures instead of panicking or
  writing directly to stdout.
- I: Core traits stay narrow: `Trainer`, `TokenizerRuntime`, and `ReportSink` are separate.
- D: High-level orchestration depends on traits for process execution, tokenization runtime, and
  report writing; concrete filesystem/process dependencies live at the edges.

## Error Model

The CLI exits with distinct operational error classes. Corpus and model findings are written to
reports and do not create a failure exit code by themselves:

| Exit | Meaning |
|---|---|
| `0` | Completed; findings may have been fixed, skipped, or reported |
| `2` | Invalid config |
| `3` | Corpus read/write failure |
| `4` | Trainer failed |
| `5` | Internal error |

Findings are not bare strings. They carry `id`, `severity`, `action`, `remedy`, `location`,
`examples`, and `accepted_by_config` so automated pipelines can distinguish a fixed line, a
skipped line, and a reported model warning.

## Dependencies

Recommended Rust crates:

| Concern | Crate |
|---|---|
| CLI | `clap` |
| JSON config/reporting | `serde`, `serde_json` |
| Precise config errors | `serde_path_to_error` |
| Errors | `thiserror`, `anyhow` only in CLI boundary |
| Unicode normalization | `unicode-normalization` |
| Unicode script classification | `unicode-script` or generated Unicode tables |
| File walking | `ignore` |
| Random sampling/shuffling | `rand`, `rand_chacha` |
| Hashing manifest inputs | `blake3` |
| Progress bars/spinners | `indicatif` |
| Terminal styling | `console` |
| Structured logs | `tracing`, `tracing-subscriber` |
| SentencePiece runtime checks | `sentencepiece` crate initially, or a subprocess verifier |

Use established crates for terminal UI, config parsing, file walking, Unicode operations,
sampling, hashing, logging, and errors. Do not hand-roll these unless a crate is missing a
specific OCR policy primitive; in that case keep the custom code inside the relevant small
domain module.

The model trainer dependency should be the Python `sentencepiece` package in v1. Current Rust
SentencePiece crates are suitable for runtime inspection/tokenization, but do not provide the
trainer API needed to replace the Python package. The Python bridge should be small enough that
it can be tested with a golden `trainer_request.json` and changed independently of the Rust
corpus pipeline.

## Interfaces

### CLI

```text
spm-ocr train --config cfg.json
```

`--json` prints only a top-level JSON summary to stdout. Human-readable Markdown reports are
always written under `work_dir/reports/`.

### Outputs

```text
runs/ocr-spm-v1/
  fixed_corpus.txt
  train_corpus/
    train-text-script_latin-lang_es-source_books-len_normal-part_0001-a1b2c3d4.txt
    train-text-script_hangul-lang_ko-source_scans-len_short-part_0001-b2c3d4e5.txt
  effective_config.json
  ocr_tokenizer.model
  ocr_tokenizer.vocab
  manifest.json
  trainer_request.json
  trainer_output.json
  reports/
    corpus_issues.jsonl
    scan.raw.json
    scan.raw.md
    scan.fixed.json
    scan.fixed.md
    balance.json
    validation.json
    validation.md
```

## Open Issues

- Should soft hyphen policy be global, per source, or both? The guide implies source behavior
  matters, so the config may need per-source overrides before production use.
- Should the Python bridge be distributed as a checked-in helper module, generated into
  `work_dir`, or embedded into the Rust binary and written to a temporary file at runtime?
- Should the first implementation parse `.model` protobuf directly, rely on the Rust
  `sentencepiece` crate for runtime checks, or ask the Python bridge to perform runtime
  encode/decode checks?
- Should `train_extremely_large_corpus=true` be kept in the recommended BPE config even though
  upstream documents it as Unigram-specific in some versions? The program can omit it for BPE if
  the Python package rejects it or reports it as unsupported.
- How should script/domain classification be customized for corpora where filenames do not
  encode source or domain?

## Milestones

1. Config schema and policy validator.
2. Corpus scanner with axis report from `09-failure-modes.md`.
3. Canonicalizer with idempotence assertion and fixed-corpus writer.
4. Balancer with hierarchical smoothing and deterministic reservoir sampling.
5. Python `sentencepiece` adapter and reproducible trainer request emission.
6. Model/corpus validation checks with nonblocking findings and issue logs.
7. Optional TUI/report browser once the batch CLI is stable.

## References

- `docs/03-configuration.md`
- `docs/05-corpus-engineering.md`
- `docs/07-normalization.md`
- `docs/08-validation.md`
- `docs/09-failure-modes.md`
- SentencePiece upstream training options: https://github.com/google/sentencepiece/blob/master/doc/options.md
- Rust `sentencepiece` runtime crate docs: https://docs.rs/sentencepiece/latest/sentencepiece/
