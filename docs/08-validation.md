# Validation checklist

Run all of this **before spending any model-training compute**. None of it requires a trained
model — these are static properties of the tokenizer artifact and the corpus that produced it.

The reason this checklist exists: nearly every failure mode in this guide is silent. Training
completes, no error is raised, and the damage only surfaces as unexplained per-script accuracy
gaps after you've paid for a training run. See [failure modes](09-failure-modes.md) for the full
inverted list and what each one costs the finished system.

Most of this is automated as two checklists — run them rather than working through this by hand:

```
spm-ocr corpus my_corpus/            # before training
spm-ocr model  ocr_tokenizer.model   # after
```

The model checks read the `.model` file itself — its pieces, and the trainer and normalizer
settings recorded inside it — so they need neither samples nor a tokenizer runtime. Two items
below do: fertility and the exact byte-fallback rate measure the tokenizer against real text,
and are reported as `SKIP` with the reason rather than silently omitted.

Run the corpus one **first**. Several vocabulary checks below only ever fail because of a corpus
defect, and discovering that after training costs you the training run. Findings are ranked by
severity and each carries its remedy — `fix_corpus` findings must be acted on before any
retrain, because retraining alone reproduces them.

Two items under **Corpus** remain manual — per-category share and lines dropped by
`max_sentence_length`. Note that neither is fixed by canonicalizing, which is why the remedy is
tracked per finding rather than per checklist.

## Corpus

- [ ] Per-category share of the assembled training file matches intended targets — verify the
      balancing actually produced what you computed.
- [ ] No Arabic presentation-form codepoints (U+FB50–FDFF, U+FE70–FEFF) anywhere in the corpus.
- [ ] Single consistent Unicode normalization form throughout (NFC).
- [ ] No training lines silently dropped by `max_sentence_length` — check the logs against your
      line-length distribution, especially for math.

## Vocabulary

- [ ] Byte-fallback firing rate near zero for the top-N frequent characters per script, and for
      common LaTeX commands not in `user_defined_symbols`.
- [ ] Every `user_defined_symbols` entry encodes as a single unsplit token in real examples.
- [ ] No digit-only vocab pieces longer than 1–2 characters, unless deliberately accepted. This
      requires `split_digits=True` at training time — see [configuration](03-configuration.md).
- [ ] No vocabulary piece spans two writing systems.
- [ ] No phantom leading space on text that does not start with one. Round-trip does **not**
      catch this — decoding strips the prefix again — so it needs its own check.
- [ ] Average piece length per category is as expected: ~1 char/token for CJK, short protected
      tokens for LaTeX, and *not* suspiciously long for low-resource scripts — that last one is
      the tell-tale sign of under-representation surviving your balancing.

## Round-trip

- [ ] `decode(encode(x)) == x` on a stratified sample from every script and from math, including
      long lines near the `max_sentence_length` ceiling.

This is the single highest-value check. It catches tokenizer-vs-ground-truth normalization
mismatches, dropped characters and byte-fallback failures in one test, and it's cheap.

**What it does not catch:** inconsistency *within* the corpus. A tokenizer trained on mixed
NFC/NFD text round-trips both forms perfectly — it is faithfully reproducing an inconsistency it
was handed — while the two forms encode to different token sequences. Round-trip stays green
throughout. That failure is visible in the vocabulary instead, which is what the NFC check
below is for. See [normalization](07-normalization.md).

- [ ] No vocabulary piece is in a decomposed (non-NFC) form.
