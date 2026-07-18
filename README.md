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
uv run spm-ocr corpus       corpus/                    # 1. measure
uv run spm-ocr canonicalize corpus/ --out canonical/   # 2. fix, then verify
uv run spm-ocr model        ocr_tokenizer.model        # 3. after training
uv run spm-ocr all          ocr_tokenizer.model --corpus canonical/
uv run spm-ocr model        ocr_tokenizer.model --json # for CI
```

Directories are walked recursively. Binary files are skipped by content, not extension — the
trained `.model` sitting beside your corpus is the usual reason that matters — and what was
skipped is always reported.

The corpus scan is **threaded by default**, dispatching chunks of lines so a single large file
parallelizes as well as many shards. The project targets free-threaded CPython (`.python-version`
pins `3.14t`), where threads run Python genuinely in parallel; on a GIL build everything still
works and simply stops being faster. `--jobs` overrides the default, and the report is identical
at any setting.

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

Exit is non-zero at `--fail-on` severity (default `high`). Defaults use a small built-in
stratified sample set; pass `--samples` from your own ground truth for a real verdict, and
`--symbols` with your `user_defined_symbols`.

Architecture: [`checks/`](sentencepiece_ocr_guide/checks/README.md) (the artifact suite) and
[`corpus/`](sentencepiece_ocr_guide/corpus/README.md) (axis scanning and canonicalization).

## Repo layout

The repo is a Python project (`uv` + `just`); the checks live alongside the guide they encode.

```
just          # list recipes
just check    # fmt-check + lint + types + test
just hooks    # install prek git hooks
```

Every claim in the guide that can be demonstrated has a test that induces the failure and proves
the corresponding check catches it — see [`tests/pitfalls/`](tests/pitfalls/).

---

*By [Vadym Stupakov](https://github.com/Red-Eyed) · MIT licensed · corrections and additions welcome*
