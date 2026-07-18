# Recommended configuration

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
    user_defined_symbols=[
        '\\frac', '\\sqrt', '\\sum', '\\int', '\\prod', '\\lim',
        '\\alpha', '\\beta', '\\gamma', '\\theta', '\\pi', '\\sigma',
        '\\times', '\\cdot', '\\leq', '\\geq', '\\neq', '\\approx',
        '\\left', '\\right', '\\begin', '\\end', '\\mathbf', '\\mathrm',
        '^', '_', '{', '}',
    ],
    num_threads=16,
)
```

## Per-parameter reasoning

**`model_type='bpe'`** — deterministic segmentation, nothing to disable. See
[BPE vs Unigram](02-bpe-vs-unigram.md).

**`normalization_rule_name='identity'`** — the single most consequential setting. The default
`nmt_nfkc` silently folds fullwidth↔halfwidth, ligatures and compatibility forms, changing what
"correct" means at eval time with no error and no warning. There is an important exception when
your ground truth is *already* normalized — see [normalization](07-normalization.md) before
committing to this.

**`character_coverage=0.9998` + `byte_fallback=True`** — used as a pair. High coverage keeps
nearly every character directly in the vocab; byte fallback ensures whatever's left decomposes
to UTF-8 bytes rather than an unrecoverable `<unk>`. In OCR an `<unk>` is a permanent error for
that character — the model has no path to ever emit it.

**`add_dummy_prefix=False`, `remove_extra_whitespaces=False`** — prevents phantom leading spaces
in non-whitespace scripts (CJK, Thai) and preserves whitespace that carries meaning (tables,
indentation, code blocks).

**`split_by_unicode_script=True`** — blocks cross-script merges, e.g. a Latin digit fusing with
a CJK character. Those merges waste vocab budget and increase confusability.

**`split_digits=True`** (default `False`) — forces every digit to its own piece. Without it BPE
merges frequent numbers into single tokens and the model learns to reproduce *those* numbers
rather than read the digits on the page. Frequency alone is enough to trigger it: a corpus
where `100` recurs produces a `100` piece. See [math and LaTeX](06-math-latex.md).

**`max_sentencepiece_length=8`** (default 16) — caps piece length, keeping CJK near
character-level. Improves error locality: one wrong token is one wrong character, not several
lost at once. Note the interaction with `user_defined_symbols`: this cap applies to *learned*
merges, so any LaTeX command longer than 8 characters (`\operatorname`, `\displaystyle`) can
never become atomic through frequency — it must be declared explicitly or it will fragment
regardless of how common it is.

**`vocab_size=40000`** — much smaller than LM vocabs (250K+). More examples per class means
sharper softmax separation and better per-class accuracy. The cost is longer sequences, which
is a latency cost, not an accuracy cost.

**`max_sentence_length=16384`** (default 4192 bytes) — SentencePiece **silently drops** lines
longer than this. Raise it for long LaTeX (matrices, multi-line derivations) and verify against
your actual line-length distribution rather than assuming.

**`user_defined_symbols`** — forces frequent LaTeX commands to survive as single unsplittable
tokens. Curate the list from real command-frequency counts on your corpus, not from the example
above. See [math and LaTeX](06-math-latex.md).

**`input_sentence_size` + `shuffle_input_sentence=True` + `train_extremely_large_corpus=True`** —
RAM-bounded training controls. See [corpus engineering](05-corpus-engineering.md); note in
particular that these do **not** solve corpus imbalance.
