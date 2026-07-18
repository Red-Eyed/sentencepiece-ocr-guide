# Corpus engineering

Two practical problems dominate everything else: the corpus is unbalanced, and it doesn't fit in
RAM. They're independent, and the tempting fix for the second one does nothing for the first.

## Problem 1 — unbalanced corpus

BPE and Unigram have no concept of language, script or domain. They allocate vocab budget in
proportion to raw frequency. A corpus that is 70% English and 0.5% Devanagari produces a vocab
that serves English well and Devanagari poorly.

The dangerous part is that this is **invisible without measuring per-script coverage**. Training
completes without error. Overall loss looks fine, because it's dominated by the majority
category. Nothing tells you the tokenizer is bad at the thing you care about.

### Fix: exponential smoothing at assembly time

    P'(L) ∝ P(L)^α,  α ≈ 0.3

Apply it **hierarchically** when there are two levels of category — domain first (text vs.
math), then script within text.

**This must happen when assembling the corpus, not via SentencePiece's own subsampling.**
`input_sentence_size` and `shuffle_input_sentence` do unweighted random sampling with no
category awareness. Subsampling a 98%-English file just gives you a smaller 98%-English file.

### Assembly procedure

1. Split the corpus into per-category shards; measure sizes.
2. Compute the target line count per category from the smoothing formula.
3. Categories **smaller** than target → take the whole shard. The over-representation is
   intentional.
4. Categories **larger** than target → subsample down.
5. Concatenate and shuffle so categories interleave.

## Problem 2 — corpus larger than RAM

Both algorithms need the corpus (or a representative sample) in memory; Unigram additionally
needs memory for the seed vocab and segmentation lattices.

**You do not need to train on the full corpus.** This is standard practice, not a hardware
workaround:

- NLLB trained its 200+ language tokenizer on **100M sampled sentences**.
- XLM-V-related experiments trained on **1B lines** sampled from multi-TB CC-100.

Tokenizer training is cheap relative to model pretraining, and it only needs a *representative*
sample — one where rare-but-important patterns appear often enough to be learned.

### Memory-ceiling controls

| Flag | Effect |
|---|---|
| `input_sentence_size` | Hard cap on lines loaded, regardless of file size on disk. |
| `shuffle_input_sentence=True` | Makes the truncation a random sample rather than the first N lines — prevents residual file ordering from reintroducing bias. |
| `train_extremely_large_corpus=True` | More memory-efficient internal statistics construction. Enable as a safety margin. |

### Weighted reservoir sampling

For shards too large to load at all: stream line by line, keeping a fixed-size reservoir of size
`k`. For each new line at position `i` beyond the initial fill, replace a random slot with
probability `k/i`. Single-pass, unbiased, and needs neither the shard size in advance nor the
shard in memory.

Target count per category = P'(L | D) × total training-file size.

### `max_sentence_length` vs. memory

Raising `max_sentence_length` for long LaTeX increases worst-case per-line memory. If memory is
tight, prefer chunking long math examples (large matrices, long derivations) during corpus prep
rather than raising the ceiling. This has a second benefit: it stops outlier-long lines from
dominating merge statistics.
