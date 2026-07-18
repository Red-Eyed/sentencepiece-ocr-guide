# Script-specific considerations

## CJK (Chinese, Japanese, Korean)

- **No whitespace boundaries.** The `add_dummy_prefix=False` /
  `remove_extra_whitespaces=False` flags are essential here, not optional polish — a phantom
  leading space on every line is a systematic corruption of the label space.
- **Fullwidth forms.** The fullwidth block (U+FF00–FFEF) duplicates ASCII at a different visual
  width. With `identity` normalization these stay distinct, which is correct for OCR — but it
  means the corpus needs real examples of *both* forms, or the model learns only one.
- **Mixed scripts per line.** Japanese mixes kanji, hiragana, katakana and Latin freely.
  `split_by_unicode_script` protects the boundaries, but spot-check that common kanji-kana
  compounds aren't being over-fragmented.
- Push character coverage close to 1.0 for CJK if you can segment coverage by script.

## Arabic, Persian, Urdu

- **Contextual shaping is not your problem.** Arabic letters have up to four positional shapes,
  but that's a *rendering-time* property — Unicode text uses one logical codepoint per letter,
  so SentencePiece never sees shape variants. This is simpler than it first appears.
- **The presentation-forms trap.** Legacy Unicode blocks Arabic Presentation Forms-A
  (U+FB50–FDFF) and Forms-B (U+FE70–FEFF) encode pre-shaped glyphs directly. If ground-truth
  extraction — PDF text layers, legacy document formats — emits these instead of logical
  codepoints, you get two redundant token sets for the same letters and halve your effective
  training signal. Catch this **upstream in the data pipeline**; the tokenizer can't fix it and
  won't warn you. These forms are one row in the canonicalization table in
  [failure modes](09-failure-modes.md#mitigation-canonicalize-what-renders-identically-preserve-what-does-not) —
  fold them to logical codepoints, and note that blanket NFKC would do it at the cost of
  distinctions you need.
- **Diacritics (tashkeel)** are sparse in general text but dense and meaning-bearing in
  religious texts, poetry and pedagogical material. If those are in the target domain, they need
  real representation in the corpus.

## Devanagari and other Brahmic scripts (Hindi, Bengali, Tamil, …)

- **Abugidas span multiple codepoints per grapheme.** A single rendered akshara can be 2–4+
  codepoints (consonant + virama + consonant + vowel sign). BPE learns conjuncts from frequency,
  so merge quality depends heavily on how well the script is represented in the corpus —
  under-representation shows up as absurdly fragmented common conjuncts.
- **NFC/NFD consistency.** Because `identity` disables the tokenizer's own normalization, mixed
  NFC/NFD input trains a tokenizer that treats identical text as two different sequences.
  Standardize the corpus to **NFC** before training — it's lossless for these scripts. Note this
  is NFC, not NFKC; see [normalization](07-normalization.md) for why that distinction matters.

## Latin, Cyrillic, Greek

Low direct risk, but usually the most abundant scripts in any corpus you assemble. Without
balancing they consume a disproportionate share of the vocab budget at the expense of scripts
with higher information density per character. This is fixed at corpus-assembly time — see
[corpus engineering](05-corpus-engineering.md).
