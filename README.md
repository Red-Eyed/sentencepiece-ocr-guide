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
spm-ocr train corpus/ \
  --model-prefix ocr_tokenizer \
  --lines 20000000 \
  --alpha 0.3 \
  --memory-budget-gb 32 \
  --training-temp-dir /path/to/scratch \
  --keep-training-file \
  --user-defined-symbols-file latex-symbols.txt
```

The command scans the corpus first, memory-maps source files, applies the same canonicalization
rules the guide validates, balances writing systems with α-smoothing, writes a bounded prepared
training sample, then invokes the official SentencePiece trainer. Math-like lines are detected
and reported, but they are not given a separate balancing bucket by default; broad OCR corpora
usually should not let a small amount of LaTeX distort the text tokenizer. Use `--balance-math`
with `--math-max-share` for math-heavy deployments.

The default trainer backend is `uv run python` with the project `sentencepiece` dependency; an
external binary is available with `--trainer-backend spm-train --spm-train /path/to/spm_train`.

The prepared training sample is a real temporary file, not a FIFO. SentencePiece reads its input
more than once, so a named pipe can hang. Pass `--keep-training-file` to retain the exact file
used for training; otherwise it is deleted after the trainer exits. Put it on a disk with enough
space via `--training-temp-dir`.

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
- **Math visibility without accidental math dominance.** Math-like lines are counted and
  reported by default, with explicit capped balancing when the deployment really is math-heavy.
- **Preflight and post-train report.** The command refuses to spend trainer compute on
  unresolved corpus defects, then checks the trained `.model` for the settings and vocabulary
  failure modes the guide names.
- **Reproducible training input.** `--keep-training-file` preserves the exact prepared text used
  to train the tokenizer.

Running raw SentencePiece with the same flags only covers the trainer settings. It does not fix
mixed encodings, does not balance the input, does not tell you what it skipped, and does not
validate the artifact afterwards.

## Examples

### Pilot: check the shape before spending compute

```bash
spm-ocr train corpus/ \
  --model-prefix pilots/ocr_pilot \
  --lines 2000000 \
  --alpha 0.5 \
  --training-temp-dir scratch/ \
  --keep-training-file
```

This is the run to do first. It scans the corpus, reports raw defects, prepares a 2M-line
training sample, keeps that exact sample on disk, trains a tokenizer, and then checks the
resulting `.model`. `--alpha 0.5` is moderate script balancing: enough to lift tails, not as
aggressive as the guide's strong `0.3` default from multilingual LM work.

Read the `training_buckets` finding before trusting the tokenizer. It tells you how many lines
were eligible in each script bucket and how many were selected.

### Broad OCR tokenizer: script-balanced text

```bash
spm-ocr train corpus/ \
  --model-prefix ocr_tokenizer \
  --vocab-size 40000 \
  --lines 20000000 \
  --alpha 0.5 \
  --training-temp-dir scratch/ \
  --keep-training-file \
  --spm-threads 16 \
  --jobs 16
```

This is the default production-shaped command for mostly-text OCR. Buckets are dominant writing
systems such as Latin, Cyrillic, Arabic, Han, Devanagari and Thai. Math-like lines are counted
and reported, but they do not get their own quota bucket, so a small amount of LaTeX cannot
accidentally dominate the tokenizer.

The Rust side stays memory-bounded: source files are memory-mapped, accepted lines flow through
a bounded shuffle buffer, and the selected sample is written to `scratch/` for SentencePiece to
read. SentencePiece is still a batch trainer, so `--lines` and scratch disk capacity matter.

### Stronger tail-script boost

```bash
spm-ocr train corpus/ \
  --model-prefix ocr_tokenizer_tail \
  --lines 20000000 \
  --alpha 0.3 \
  --training-temp-dir scratch/ \
  --keep-training-file
```

Lower alpha means stronger balancing. Use this if pilot reports show low-resource scripts are
still underrepresented. It will make rare scripts much more visible to BPE merge learning, but
it can also move the tokenizer further from your deployment distribution, so inspect the kept
training file and post-train `script_coverage`.

### Math-heavy deployment

```bash
spm-ocr train corpus/ \
  --model-prefix ocr_tokenizer_math \
  --lines 20000000 \
  --alpha 0.5 \
  --balance-math \
  --math-max-share 0.10 \
  --user-defined-symbols-file latex-symbols.txt \
  --training-temp-dir scratch/ \
  --keep-training-file
```

This gives math-like lines their own balancing bucket, capped at 10% of the selected sample.
Use it for exams, papers, formula-heavy pages or math OCR. Keep the cap unless the deployment is
truly math-dominant; otherwise text scripts lose too much budget.

`latex-symbols.txt` should be curated from your corpus. Each line is one protected symbol or
command, for example:

```text
\frac
\sqrt
\sum
\operatorname
^
_
{
}
```

### External SentencePiece binary

```bash
spm-ocr train corpus/ \
  --model-prefix ocr_tokenizer \
  --trainer-backend spm-train \
  --spm-train /opt/homebrew/bin/spm_train \
  --training-temp-dir scratch/
```

The default backend uses `uv run python` and the project `sentencepiece` dependency. This
variant uses an installed `spm_train` executable instead. The corpus preparation and validation
are identical; only the final trainer invocation changes.

### Lossy data triage

```bash
spm-ocr train corpus/ \
  --model-prefix ocr_tokenizer_triage \
  --drop-invalid \
  --drop-long-lines \
  --training-temp-dir scratch/ \
  --keep-training-file
```

This is for triage, not a clean production tokenizer. Invalid UTF-8 and over-length lines are
data loss. The defaults refuse them because bad bytes and very long lines usually identify an
extractor problem or exactly the hard examples you wanted to train on. Use these flags only when
you have decided losing those lines is better than blocking the run.

## The two checklists

The [validation checklist](docs/08-validation.md) is implemented as `spm-ocr`. There are two,
and the order matters — most model defects originate in the corpus, so scanning first saves you
a training run:

```
just workflow                              # the whole flow, with runnable examples
just scan       corpus/                    # 1. measure
just canon      corpus/ canonical/         # 2. fix, then verify
spm-ocr train   canonical/ --model-prefix ocr_tokenizer \
                 --training-temp-dir scratch/ --keep-training-file
                                           # 3. train with mmap prep + balanced sample
just check-model ocr_tokenizer.model       # 4. after training
just check-all   ocr_tokenizer.model canonical/
                                           # both checklists, corpus findings first
just options                               # every flag for every subcommand
```

Recipes forward extra flags (`just scan corpus/ --jobs 2 --json`) and print the underlying
`spm-ocr` command, so nothing is hidden. `just` on its own lists everything, split into
`[workflow]` and `[dev]`.

Directories are walked recursively. Binary files are skipped by content, not extension — the
trained `.model` sitting beside your corpus is the usual reason that matters — and what was
skipped is always reported.

The corpus scan and train preparation are **parallel by default**, across sources and within
each one: a file above a
megabyte is split into line-aligned chunks, so a single large file parallelizes as well as many
shards. Files are memory-mapped rather than read, which is what lets a corpus larger than RAM be
scanned at all. `--jobs` overrides the default, and the report is byte-identical at any setting.

`train` is bounded on the Rust side: it does not keep the corpus or the selected sample in RAM.
It uses a first pass for counts, computes α-smoothed bucket quotas, then streams accepted
canonicalized lines through a bounded shuffle buffer into the prepared training file.
SentencePiece itself is still a batch trainer, so the training sample size and temp-disk
location matter.

Findings are ranked worst-first and carry both a severity and a remedy:

```
FAIL [blocker]  axis[nfc_composition]: 412,883 of 2,104,556 lines — canonically equivalent
        vendor_b.txt: 402,110 / 900,000 lines
FAIL [high]  digit_pieces: 9 digit-only pieces exceed 1 characters
        '100'
        '▁250'
SKIP [high]  protected_symbols: no protected symbols supplied
PASS  axis[fullwidth_forms]: 118,655 of 2,104,556 lines (expected — preserve, do not fold)

Next:
  1. canonicalize the corpus, then retrain
  2. change the trainer flags and retrain
  (in that order — a corpus defect survives any number of retrains)
```

Three things that matter about that output:

- **Remedy is per finding, not per checklist.** `nfc_vocabulary` fails on the *model* but is a
  corpus defect — retraining alone reproduces it exactly. The report says so.
- **`PASS` with a non-zero count is not a defect.** Fullwidth forms showing up means you have
  CJK data. A report that cannot tell "found variation" from "found a problem" trains you to
  ignore it.
- **`SKIP` is never `PASS`,** and keeps its severity. A skipped blocker is exactly what must not
  read as clean.

Exit is non-zero at `--fail-on` severity (default `high`).

## What the model checks read

The model checklist reads the `.model` protobuf and nothing else — its piece inventory and the
trainer and normalizer settings recorded inside it. No samples, no tokenizer runtime, no corpus.

That is a narrower input than a checklist like this usually assumes, and for most of it a
stronger one. Encoding samples to see whether `<unk>` appears tells you it did not appear *in
those samples*; reading `byte_fallback` together with a complete set of 256 byte pieces tells you
it cannot appear **for any input**. For the one check the guide calls out as having no acceptable
failure threshold, that is the difference between evidence and proof. The same applies to
`add_dummy_prefix`, which the sampling version infers by experiment and the artifact simply
states.

It also reaches settings a tokenizer runtime does not expose at all, so the checklist can ask
whether the model was *actually* trained as BPE, with `split_digits`, at the recommended piece
length — the contents of [configuration](docs/03-configuration.md), verified against the artifact.

Two checks need what this cannot give them. `fertility` and the exact byte-fallback rate measure
the tokenizer against real text, so both require encoding it. They report as `SKIP` with the
reason attached and keep their severity.

Where a recommendation depends on something the tool cannot see, the setting is reported rather
than graded. `normalization_rule` is the case that matters: `identity` is correct only if your
ground truth is un-normalized, so the report states what the model does and leaves the verdict
to you.

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

Validation-only commands use the Rust binary. Training with the default backend also needs
`uv sync` so `sentencepiece` is available to `uv run python`; alternatively install
`spm_train` yourself and pass `--trainer-backend spm-train`.

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
