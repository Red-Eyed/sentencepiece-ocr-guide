# SentencePiece for multilingual OCR

Notes on designing and training a SentencePiece tokenizer for an OCR system covering 100+
languages — including CJK, Arabic and Brahmic scripts — plus mathematical notation transcribed
as LaTeX.

Most published guidance on multilingual SentencePiece targets **language models**. An OCR
tokenizer defines the *label space of a decoder* rather than the input to an encoder, and that
single difference flips several standard recommendations, including the choice of algorithm.
This guide works through where and why.

Nearly every failure mode described here is **silent**: training completes, no error is raised,
and the damage surfaces later as unexplained per-script accuracy gaps. That's what the
[validation checklist](docs/08-validation.md) is for.

## Contents

| | |
|---|---|
| [1. Prior art](docs/01-prior-art.md) | Repos, papers, and what Glot500 actually does |
| [2. BPE vs Unigram](docs/02-bpe-vs-unigram.md) | Why the standard LM answer flips for OCR |
| [3. Configuration](docs/03-configuration.md) | The full config block, per-parameter reasoning |
| [4. Script considerations](docs/04-scripts.md) | CJK, Arabic, Brahmic, Latin — traps per script |
| [5. Corpus engineering](docs/05-corpus-engineering.md) | α-smoothing for balance; RAM-bounded training |
| [6. Math and LaTeX](docs/06-math-latex.md) | Command atomicity, digit splitting |
| [7. Normalization](docs/07-normalization.md) | identity vs. NFKC vs. NFC — and the common mistake |
| [8. Validation](docs/08-validation.md) | Pre-flight checks, before spending training compute |
| [9. Failure modes](docs/09-failure-modes.md) | Inverted: everything that can go wrong, what it costs, how to mitigate |

## The short version

If you read nothing else:

- **Use BPE, not Unigram.** Unigram's subword regularization is inert when the decoder trains
  against one canonical ground-truth segmentation, and it costs you determinism.
- **`normalization_rule_name='identity'`** — but only if your ground truth is un-normalized. The
  real rule is that tokenizer normalization must *match* ground-truth normalization.
- **`byte_fallback=True`, always.** An `<unk>` in OCR is a permanent, unrecoverable error for
  that character.
- **Balance the corpus at assembly time.** `input_sentence_size` samples without any category
  awareness — subsampling a 98%-English file gives you a smaller 98%-English file.
- **`split_digits=True`.** Off by default, and ordinary corpus frequency is enough to merge
  `100` into one token — after which the model reproduces familiar numbers instead of reading
  the ones on the page.
- **You don't need your full corpus.** NLLB trained a 200+ language tokenizer on 100M sentences.

## Quick reference

```bash
spm-ocr train --config cfg.json
# or:
just train cfg.json
```

Start from [`cfg.json.example`](cfg.json.example). It points at
[`latex-symbols.txt.example`](latex-symbols.txt.example) for common math symbols. Config mode is
strict: every key in the example must be present, unknown keys fail, and command-line training
flags conflict with `--config` so a misspelled or misplaced parameter cannot be silently skipped.

The command scans the corpus first, memory-maps source files, applies the same canonicalization
rules the guide validates, balances writing systems with α-smoothing, writes a bounded prepared
training sample, then invokes the official SentencePiece trainer. Math-like lines are detected
and reported, but they are not given a separate balancing bucket by default; broad OCR corpora
usually should not let a small amount of LaTeX distort the text tokenizer. Set
`"balance_math": true` with `"math_max_share"` for math-heavy deployments.

The default trainer backend is `uv run python` with the project `sentencepiece` dependency; an
external binary is available by setting `"trainer_backend": "spm-train"` and `"spm_train"` in
the config.

The prepared training sample is a real temporary file, not a FIFO. SentencePiece reads its input
more than once, so a named pipe can hang. Set `"keep_training_file": true` to retain the exact
file used for training; otherwise it is deleted after the trainer exits. Put it on a disk with
enough space via `"training_temp_dir"`.

The underlying SentencePiece parameters are the guide's OCR-safe defaults: BPE,
`byte_fallback=true`, `normalization_rule_name=identity`, no dummy prefix, whitespace
preserved, Unicode-script splitting, digit splitting, `max_sentencepiece_length=8`,
`max_sentence_length=16384`, `train_extremely_large_corpus=true`. Every parameter is justified
in [configuration](docs/03-configuration.md). Don't copy this without reading that page —
several defaults are wrong for OCR in ways that produce no error.

## Why not just run SentencePiece directly?

The model trainer is still official SentencePiece. The difference is everything upstream and
around it — the places where the silent failures in [failure modes](docs/09-failure-modes.md)
enter:

- **Corpus mmap and bounded preparation.** Source files are mapped and streamed, so a 50GB
  corpus does not become a 50GB heap allocation.
- **Measured canonicalization.** NFC and the guide's safe collapse axes are applied before
  training, while OCR-visible compatibility distinctions are preserved.
- **α-smoothed corpus assembly.** SentencePiece's own `input_sentence_size` is unaware of
  language, script or domain; this tool balances the prepared sample before the trainer sees it.
- **Math visibility without accidental math dominance.** Math-like lines get a capped balancing
  bucket by default, so formulas are present without taking over the tokenizer.
- **Preflight and post-train report.** The command refuses to spend trainer compute on
  unresolved corpus defects, then checks the trained `.model` for the settings and vocabulary
  failure modes the guide names.
- **Reproducible training input.** `"keep_training_file": true` preserves the exact prepared text used
  to train the tokenizer.

Running raw SentencePiece with the same flags only covers the trainer settings. It does not fix
mixed encodings, does not balance the input, does not tell you what it skipped, and does not
validate the artifact afterwards.

## Training

There is one path:

```bash
cp cfg.json.example cfg.json
just train cfg.json
```

The example config is meant to cover most multilingual on-device OCR runs. It samples 20M lines,
uses a 40K BPE vocabulary, preserves byte fallback, splits digits and Unicode scripts, protects
common LaTeX/math symbols from [`latex-symbols.txt.example`](latex-symbols.txt.example), and
caps math-like lines at 5% so formulas are represented without taking over the tokenizer.

Edit `"paths"` and `"model_prefix"` first. The other fields are explicit so typos and accidental
omissions fail before training starts. Unknown JSON keys fail too.

The train command does the full pipeline:

- discovers corpus files and reports skipped inputs
- scans for corpus defects before spending trainer compute
- canonicalizes safe normalization axes in-stream
- computes alpha-smoothed script/math quotas
- writes a bounded shuffled training file under `"training_temp_dir"`
- runs the official SentencePiece trainer
- reads the trained `.model` and reports artifact checks

The Rust side is memory-bounded: source files are memory-mapped, accepted lines flow through a
bounded shuffle buffer, and the selected sample is written to scratch storage for SentencePiece
to read. SentencePiece is still a batch trainer, so `"lines"` and scratch disk capacity matter.

Findings are ranked worst-first and carry both a severity and a remedy. A failed preflight
finding blocks training unless the config explicitly opts into the lossy handling for that case,
such as `"drop_invalid": true` or `"drop_long_lines": true`.

## Repo layout

A Rust project (`cargo` + `just`) plus a minimal `uv` Python project for the default
SentencePiece trainer backend.

```
just          # list recipes, grouped into [workflow] and [dev]
just check    # fmt-check + lint + test
just build    # the release binary
just runtime  # toolchain, and the parallelism the scan defaults to
just hooks    # install the pre-commit hook
```

Training with the default backend also needs `uv sync` so `sentencepiece` is available to
`uv run python`; alternatively install `spm_train` yourself and set `"trainer_backend":
"spm-train"` in `cfg.json`.

The checks live alongside the guide they encode: [`src/corpus/`](src/corpus/) scans and
canonicalizes, [`src/model/`](src/model/) reads the artifact, and both produce the same
[`Finding`](src/report.rs) vocabulary so a corpus result and a model result rank, read and exit
identically.

Two trained tokenizers are checked in at [`tests/fixtures/`](tests/fixtures/) — one configured
the way this guide argues for, one with SentencePiece's stock defaults. The integration tests run
the full checklist against both, so the claim that the defaults are wrong for OCR is executable
rather than asserted.

---

*By [Vadym Stupakov](https://github.com/Red-Eyed) · MIT licensed · corrections and additions welcome*
