# BPE vs Unigram — the answer flips for OCR

For multilingual **language modelling**, Unigram is the standard choice. XLM-R, Glot500, mBART,
NLLB and mT5 all use it, for two reasons:

1. **Subword regularization.** Unigram can sample multiple valid segmentations of the same
   string, acting as data augmentation for a representation-learning encoder. BPE has no native
   equivalent — one canonical segmentation per string.
2. **More even cross-lingual vocabulary allocation.** Unigram's EM-based global optimization is
   less prone than BPE's greedy merges to letting whichever language dominates early training
   entrench its patterns.

For **OCR, BPE is the better choice.** The reason isn't that those two properties stop being
true — it's that the tokenizer is doing a different job. It defines the *label space of a
decoder* rather than the input to an encoder, and that reframes every priority:

| Concern | LM tokenizer | OCR tokenizer |
|---|---|---|
| Segmentation | Probabilistic, regularized (Unigram) | Deterministic, reproducible (BPE) |
| Unicode normalization | Aggressive (NFKC) is fine | Must be `identity` — visual fidelity is the label |
| OOV handling | Acceptable to lose rare chars | Never lose a target char — always byte-fallback |
| Vocab size | Large (250K+) for shared semantics | Smaller, biased toward per-class accuracy |
| Piece length | Long merges fine | Short merges preferred (localizes errors, esp. CJK) |

## Why BPE fits

**Determinism matters more than regularization.** The decoder is trained against one canonical
ground-truth segmentation. Unigram's stochastic segmentation is inert here — there's no
mechanism to reward alternative segmentations — while it removes a guarantee (determinism) that
the loss actually depends on. You pay a cost for a benefit you can't collect.

**There's no cross-lingual representation-sharing objective.** No MLM loss is pulling the
vocabulary in competing directions, so Unigram's more even allocation is solving a problem you
don't have. Balance still matters, but it's achieved at corpus-assembly time — see
[corpus engineering](05-corpus-engineering.md) — not by the choice of algorithm.

**Decode efficiency and error locality.** Autoregressive decode cost scales with sequence
length, and short deterministic merges keep the error radius small: one wrong token corrupts
one or two characters rather than a whole word.

**Compatibility with pretrained decoders.** Staying in BPE-space matches TrOCR-style
architectures that reuse an existing byte-level BPE decoder vocabulary.
