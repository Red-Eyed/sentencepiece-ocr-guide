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

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='balanced_corpus.txt',
    model_prefix='ocr_tokenizer',
    model_type='bpe',
    vocab_size=40000,
    character_coverage=0.9998,
    byte_fallback=True,
    normalization_rule_name='identity',
    add_dummy_prefix=False,
    remove_extra_whitespaces=False,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_digits=True,
    max_sentencepiece_length=8,
    max_sentence_length=16384,
    input_sentence_size=20000000,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=True,
    user_defined_symbols=[...],  # curate from real LaTeX command frequencies
    num_threads=16,
)
```

Every parameter is justified in [configuration](docs/03-configuration.md). Don't copy this
without reading that page — several defaults are wrong for OCR in ways that produce no error.

## The two checklists

The [validation checklist](docs/08-validation.md) is implemented as `spm-ocr`. There are two,
and the order matters — most model defects originate in the corpus, so scanning first saves you
a training run:

```
just workflow                              # the whole flow, with runnable examples
just scan       corpus/                    # 1. measure
just canon      corpus/ canonical/         # 2. fix, then verify
                                           # 3. train your tokenizer on canonical/
just check-model ocr.model                 # 4. after training
just check-all   ocr.model canonical/      # both checklists, corpus findings first
just options                               # every flag for every subcommand
```

Recipes forward extra flags (`just scan corpus/ --jobs 2 --json`) and print the underlying
`spm-ocr` command, so nothing is hidden. `just` on its own lists everything, split into
`[workflow]` and `[dev]`.

Directories are walked recursively. Binary files are skipped by content, not extension — the
trained `.model` sitting beside your corpus is the usual reason that matters — and what was
skipped is always reported.

The corpus scan is **parallel by default**, across sources and within each one: a file above a
megabyte is split into line-aligned chunks, so a single large file parallelizes as well as many
shards. Files are memory-mapped rather than read, which is what lets a corpus larger than RAM be
scanned at all. `--jobs` overrides the default, and the report is byte-identical at any setting.

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

A Rust project (`cargo` + `just`), with no runtime dependencies beyond the binary itself.

```
just          # list recipes, grouped into [workflow] and [dev]
just check    # fmt-check + lint + test
just build    # the release binary
just runtime  # toolchain, and the parallelism the scan defaults to
just hooks    # install the pre-commit hook
```

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
