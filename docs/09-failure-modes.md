# Failure modes — inverted

> "All I want to know is where I'm going to die, so I'll never go there." — Charlie Munger

The rest of this guide argues forward: here is the config, here is why. This page inverts it.
Instead of asking *how do I build a good tokenizer*, it asks **what could make the finished OCR
system worse, and how would the tokenizer be to blame?**

Inversion is worth the separate page because the forward argument has a blind spot: it only
covers decisions you knew you were making. Most tokenizer damage comes from decisions you
didn't know were decisions — a default you never read, a corpus property you never measured.

Two things make these failures expensive:

- **They are silent.** Training completes, no error, no warning. Every failure below produces a
  working tokenizer and a working training run.
- **They are upstream of everything.** The tokenizer defines the decoder's label space. A defect
  here is not a bug you fix later — it is baked into every checkpoint trained against it, and
  the only remedy is retraining from scratch.

## Where you die

Numbers are used throughout this guide and by `spm-ocr`, which cites them in its findings. They
match the detailed entries below one-for-one.

| # | Failure | Effect on the OCR product | Silent? | Checked by |
|---|---|---|---|---|
| 1 | `<unk>` in the vocab path | Character is **permanently unreadable** — no path to ever emit it | Yes | `no_unknown` |
| 2 | Tokenizer normalizes more than GT | Model asked to emit what its vocab can't represent | Yes | `normalization_rule`, `whitespace_preserved` |
| 3 | Normalization strips ZWJ/ZWNJ | Wrong words in Persian, Hindi, Bengali | Yes | `axis[zero_width_joiners]` |
| 4 | **Mixed corpus encodings** — *the one to expect* | Identical text trains as two sequences — signal divided | Yes — *round-trip misses it* | `axis[*]`, `nfc_vocabulary` |
| 5 | Arabic presentation forms | Two redundant token sets for the same letters | Yes | `axis[arabic_presentation_forms]` |
| 6 | Subword regularization as a decoder target | Same image maps to different label sequences across epochs | Yes | `algorithm` |
| 7 | Invalid byte sequences at inference | Decoder emits bytes that aren't valid UTF-8 → mojibake or crash | **No** — but only in production | `invalid_utf8` |
| 8 | Vocab too large | Rare classes undertrained; per-class accuracy drops | Yes | `trainer_settings` |
| 9 | Vocab too small | Longer sequences, more steps to derail, slower | Yes | `trainer_settings` |
| 10 | Pieces that are too long | Poor error locality — one wrong token, several wrong characters | Yes | `trainer_settings` |
| 11 | Cross-script merges | Wasted budget, higher confusability | Yes | `cross_script_pieces`, `script_splitting` |
| 12 | Multi-digit pieces | Memorizes frequent numbers, misreads the ones on the page | Yes | `digit_pieces`, `split_digits` |
| 13 | Script imbalance | Low-resource scripts fragment → measurably worse accuracy | Yes | `script_coverage`, `script_balance` |
| 14 | Text/math imbalance | Either notation starves the other | Yes | — |
| 15 | Lines dropped by `max_sentence_length` | Loses exactly the hard examples (matrices, tables) | Yes | `long_lines` |
| 16 | Corpus/deployment domain mismatch | Sequences run long on the documents you deployed for | Yes | — |
| 17 | `character_coverage` too low for CJK | Rare-but-real characters fall to byte fallback — 3 tokens each | Yes | `byte_fallback_rate` |
| 18 | Tokenizer swapped post-training | Total label-space shift — garbage output | Yes | — |
| 19 | Special-token id collision | Off-by-one label shift across the whole vocab | Yes | — |
| 20 | Vocab size vs. output layer mismatch | A vocab just above the byte floor is nearly character-level | Yes | `trainer_settings` |
| 21 | Whitespace policy mismatch | Eval measures a difference introduced after the model was right | Yes | `no_phantom_prefix` |

## If you fix one thing, fix #4

Ranking by *consequence*, #1 is the worst. Ranking by **probability that it is in your corpus
right now**, #4 wins, and it is not close.

Every other failure on this list is a decision you make once. You read this page, you set
`byte_fallback=True` and `split_digits=True`, and those failures are gone permanently. A
configuration flag stays fixed.

Mixed encodings are not a decision. They are a *property of data you did not create*, and they
re-enter the moment anyone adds a source — a new language, a new scanner, a synthetic generator,
a vendor delivery. You cannot fix it once. You can only build a chokepoint that keeps fixing it,
which is what the mitigation below is.

It is also the only failure here that is invisible to the guide's highest-value check. Round-trip
passes. The vocabulary is the only place it shows.

---

## Class A — The label is unrepresentable

The model *cannot* be right. No amount of training data or compute fixes these, because the
target itself is outside what the tokenizer can express.

**1. `<unk>` anywhere in the encoding path.** With `byte_fallback=False`, any character outside
the vocab encodes to `<unk>`. For a language model that's a minor information loss on the input
side. For OCR it is fatal in a specific way: `<unk>` is a *label*, so the model is trained to
emit `<unk>` for that character, and at inference there is no path from `<unk>` back to the
original character. The character is permanently unreadable by that system — not misread,
**unreadable**. This is why `byte_fallback=True` is the one setting in this guide with no
exception attached.

**2. The tokenizer normalizes more than ground truth does.** The default
`normalization_rule_name='nmt_nfkc'` folds fullwidth→halfwidth, ligatures ﬁ→fi, superscripts
²→2. If your GT preserves those distinctions, the model is being trained to emit a token
sequence that decodes to *different text than the label*. Round-trip exactness is broken before
training starts. See [normalization](07-normalization.md) — the invariant is that tokenizer
normalization must **equal** GT normalization, in both directions.

**3. Normalization strips zero-width joiners.** A specific case of #2 that deserves its own
entry because it is so easy to miss. `nmt_nfkc` removes control characters, including ZWNJ
(U+200C) and ZWJ (U+200D). These are **meaning-bearing**, not formatting: in Persian they
separate morphemes (می‌رود vs میرود), in Devanagari and Bengali they control conjunct formation.
Stripping them produces text that is *wrong*, not merely differently-normalized — and it will
never show up in an English-language test set.

## Class B — The target is ambiguous

The label is representable, but the same rendered text maps to more than one token sequence.
The model is being taught two answers to one question, and the training signal for each is
diluted.

**4. Mixed encodings in the corpus.** The textbook case is NFC vs NFD — `é` as one codepoint and
`é` as `e` + U+0301 become unrelated token sequences for identical-looking text, so every
affected grapheme trains at a fraction of its true frequency. But normalization form is only one
of the axes on which two sources can disagree, and in practice they disagree on several at once.

**Why it is structural, not bad luck.** You never have "a corpus". You have a union of
extraction pipelines: PDF text layers, HTML scrapes, human transcription, vendor deliveries,
legacy OCR output, and — for OCR specifically — synthetic renderers, which are usually the
largest single source. Each was written by different people against different libraries, and
each has its own opinion about how text should come out. Heterogeneity is the default state of
an assembled corpus. The useful question is never *is my corpus mixed* but *on how many axes*.

The damage compounds rather than adding: each independent axis of variation multiplies the
number of distinct encodings for the same rendered string, and your effective examples per
grapheme divide by that number.

**Why round-trip does not catch it.** Both forms decode back to exactly what went in — the
tokenizer is faithfully reproducing an inconsistency it was handed. The highest-value check in
the guide stays green while the label space is silently split. The evidence lives in the
*vocabulary*: decomposed corpus text yields decomposed pieces.

### Mitigation: collapse encoding differences, preserve character differences

It is tempting to derive the rule from appearance — *if two encodings render the same, they are
the same label*. *That rule is wrong*, and it is worth seeing why before stating the right one.

**OCR does not map appearance to Unicode.** The mapping is not a function: Latin `A` (U+0041),
Cyrillic `А` (U+0410) and Greek `Α` (U+0391) are the same glyph in most fonts, and neither NFC
nor NFKC collapses them — correctly, because they are three different letters. Appearance alone
underdetermines the codepoint. What resolves it is **context**: a Cyrillic `А` sits inside
Cyrillic words. So OCR maps *appearance plus context* to Unicode, and the model is expected to
learn the context part. Homoglyphs must stay distinct in the label space, or you have destroyed
the distinction the model was supposed to make. (`split_by_unicode_script=True` is on your side
here — it stops the vocabulary blurring script boundaries.)

The rule that does work is about the text, not the pixels:

> Two encodings are the same label when they are the **same sequence of abstract characters**.
> Unicode calls this *canonical equivalence*, and **NFC is exactly the operation that collapses
> it**. Anything that is a genuinely different character survives — even when the glyph is
> identical.

That is the whole principle, and it explains the guide's advice better than the visual rule did.
NFC collapses differences that are *not differences in the text at all* — pure encoding artifacts.
NFKC goes further and collapses *compatibility* equivalents, which are different characters that
merely resemble each other, and that is precisely why it is wrong for OCR:

| | Canonically equal | Compatibility equal | Verdict |
|---|---|---|---|
| `café` NFD vs NFC | **Yes** | Yes | Collapse — same text |
| Fullwidth `Ａ` vs `A` | No | Yes | Preserve — different characters |
| Ligature `ﬁ` vs `fi` | No | Yes | Preserve |
| NBSP vs space | No | Yes | *Exception — collapse* |
| Presentation `ﻛ` vs `ك` | No | Yes | *Exception — collapse* |
| Latin `A` vs Cyrillic `А` | No | No | Preserve — context decides |

So the pipeline is **NFC, plus a short list of deliberate exceptions** — each one a legacy
encoding artifact rather than a real distinction, and each one justified individually:

| Variation | Action | Why |
|---|---|---|
| NFC vs NFD, combining-mark order | **Compose** (NFC) | Canonically equivalent — not a difference in the text |
| Arabic presentation forms (U+FB50–FDFF, U+FE70–FEFF) | **Map** to logical codepoints | Legacy pre-shaped glyphs; shaping is derivable from context |
| NBSP vs space | **Map** to U+0020 | Differs only in line-breaking, which a line image cannot show |
| BOM, ZWSP | **Strip** | Not page content at all |
| Soft hyphen U+00AD | **Decide by position** | See below |
| Fullwidth, ligatures, Arabic-Indic digits, quotes, dashes, U+3000 | **Preserve** | Different characters, visibly different glyphs |
| ZWJ / ZWNJ | **Preserve** | Different characters that change neighbours' shapes |
| Homoglyphs across scripts | **Preserve** | Different letters; context disambiguates, not pixels |

The row people most often get backwards is ZWNJ — invisible *as a glyph*, but it reshapes its
neighbours, so it is visible on the page (failure mode #3). Arabic presentation forms are the
runner-up: they render correctly, which is exactly why they survive undetected.

**Soft hyphen is the row that proves the rule is contextual.** U+00AD is a *conditional* hyphen:
invisible mid-line, rendered as a real hyphen glyph where the line breaks. OCR ground truth is
usually segmented per line — which is precisely the position where it is visible. PDF extractors
emit it for discretionary hyphens that the page really did draw. Strip it blindly and you train
the model to omit a mark that is on the page (failure mode #2); keep it blindly and you carry an
invisible codepoint through the middle of words. The correct action depends on where it sits and
on how your extractor behaves, so **decide it deliberately after looking at your data** — a
line-final U+00AD usually wants mapping to U+002D, a mid-line one usually wants stripping.

This is the general lesson: the exception list is not universal. Canonical equivalence is
objective and you can apply NFC without thinking; every exception beyond it is a judgement about
*your* sources and *your* rendering context. Which is why you measure before you canonicalize.

### Measure first — you cannot canonicalize what you have not characterised

"Is my corpus mixed?" is not answerable by inspection, and guessing at the exception list is how
you end up stripping a soft hyphen that was on the page. Both problems have the same answer:
before writing the canonicalizer, scan the corpus and count, per axis, how many lines change
under each candidate transformation.

```
axis                          lines affected   sources
NFC composition                    412,883      wiki_dump, vendor_b
Arabic presentation forms           38,204      pdf_extract_2019
soft hyphen (line-final)             9,117      pdf_extract_2019
soft hyphen (mid-line)                 226      pdf_extract_2019
NBSP                                 2,940      html_scrape
fullwidth forms                    118,655      cjk_corpus          <- preserve, do not fold
```

That report does three things a canonicalizer alone cannot. It tells you **which axes are
actually present**, so you write exceptions for real problems instead of imagined ones. It
attributes them **per source**, which usually identifies one broken extractor rather than a
diffuse issue. And it quantifies the blast radius, so the soft-hyphen question stops being
theoretical — 9,117 line-final versus 226 mid-line is a decision you can now make.

Re-run it after canonicalizing: every collapsible axis should read zero, and the preserve rows
should be unchanged. That difference is the proof the stage did what you intended.

In code this is one function, and the presentation-form row is why it cannot just be
`unicodedata.normalize`:

```python
import unicodedata

_STRIP = {0xFEFF, 0x200B}   # BOM, zero-width space — never rendered anywhere
_REPLACE = {0x00A0: " "}    # NBSP renders as an ordinary space
# Deliberately absent: U+200C ZWNJ and U+200D ZWJ (they alter shaping), and U+00AD soft
# hyphen (visible at a line break — handle it by position, see above).

def _fold_presentation_forms(char: str) -> str:
    """Compatibility folding, applied only where it is lossless: pre-shaped Arabic glyphs."""
    if 0xFB50 <= ord(char) <= 0xFDFF or 0xFE70 <= ord(char) <= 0xFEFF:
        return unicodedata.normalize("NFKC", char)
    return char

def canonicalize(text: str) -> str:
    text = text.translate({cp: None for cp in _STRIP} | _REPLACE)
    text = "".join(_fold_presentation_forms(char) for char in text)
    return unicodedata.normalize("NFC", text)
```

(Strip the BOM before the presentation-form pass — U+FEFF sits inside the Forms-B range.)

Four things make it hold:

1. **One chokepoint.** Every source passes through `canonicalize` on the way in, and it is the
   only way to add data. A per-source normalization step that each pipeline is *supposed* to
   call is the same failure with extra steps.
2. **Assert, don't hope.** `canonicalize` is idempotent, so `line == canonicalize(line)` is a
   valid assertion at corpus-write time. It costs microseconds per line and converts a silent
   failure into a loud one.
3. **The same function on evaluation ground truth.** Otherwise CER counts your own
   normalization difference as model error, and you will spend a week debugging the model.
4. **Version it with the tokenizer.** Changing `canonicalize` changes the label space as surely
   as retraining the tokenizer does — see #18. Hash both into the checkpoint.

After training, `nfc_vocabulary` in the check suite confirms the composition half actually took:
non-NFC pieces in the vocabulary prove decomposed text reached the trainer. See
[normalization](07-normalization.md) for why NFKC is the wrong shortcut.

**5. Arabic presentation forms.** Covered in [scripts](04-scripts.md); listed here because the
consequence is the same as #4 — two redundant token sets, halved signal per set — and because
the tokenizer cannot detect or fix it. It must be caught in the data pipeline.

**6. Subword regularization.** Unigram's sampled segmentation is the intended behavior for LM
*input*. As a decoder *target* it means the same image maps to different label sequences across
epochs. See [BPE vs Unigram](02-bpe-vs-unigram.md).

## Class C — The failure that only appears in production

**7. The decoder emits an invalid UTF-8 byte sequence.** This one is different from every other
entry: it is not silent, and no pre-flight check on the tokenizer will find it, because the
tokenizer is not defective. Byte fallback means 256 byte pieces are valid outputs. An
autoregressive decoder can emit *any* sequence of them — including sequences that are not valid
UTF-8 (a continuation byte with no lead byte, a truncated multi-byte sequence at a length
limit).

`decode()` on that produces replacement characters, mojibake, or an exception depending on your
decoding path. In a batch inference job an uncaught exception here takes down the whole batch
over one bad page.

This is the cost of `byte_fallback=True`, and it is still worth paying — #1 is worse. But it
means **your inference path needs an explicit decode-failure policy**, decided deliberately
rather than discovered at 3am. Test it directly: hand your decode path a deliberately invalid
byte-piece sequence and confirm it does what you chose.

## Class D — The label space has bad geometry

Everything is representable and unambiguous. Accuracy is still worse than it should be, because
of how the vocab budget was spent.

**8. Vocab too large.** Fewer examples per class, softmax probability mass spread thinner, rare
pieces undertrained. LM-scale vocabs (250K+) are actively wrong here — see
[configuration](03-configuration.md).

**9. Vocab too small.** Sequence length grows, which for an autoregressive decoder means more
steps per page, more opportunities to derail, and higher latency. Note this trades against #8;
there is no free direction, which is why 40K is a *balance* and not a maximum.

**10. Pieces that are too long.** Error locality: with `max_sentencepiece_length=16`, a single
wrong token can be 16 wrong characters. Capping at 8 means a mistake costs less CER. This
matters more for OCR than for an LM, because OCR errors are judged per character.

**11. Cross-script merges.** A piece spanning Latin and CJK wastes a vocab slot on a sequence
that occurs only at incidental boundaries, and adds a confusable class.

**12. Multi-digit pieces.** The model learns frequent numbers rather than reading digits. On
invoices, tables and math — where the numbers are the *entire point* and are by definition not
the ones in your training corpus — this is a direct accuracy loss on the highest-value content.
See [math and LaTeX](06-math-latex.md).

## Class E — The corpus, not the config

**13. Script imbalance.** The vocab budget follows frequency. An under-represented script gets
few merges, so its common sequences fragment into many short pieces — longer sequences and worse
accuracy for exactly the languages that were already weakest. The tell-tale sign is an average
piece length near 1.0 for a script that should have learned conjuncts. Fixed at assembly time
with α-smoothing, not by `input_sentence_size` — see [corpus engineering](05-corpus-engineering.md).

**14. Text/math imbalance**, in both directions — see [math and LaTeX](06-math-latex.md).

**15. Lines silently dropped by `max_sentence_length`.** The default is 4192 *bytes*, and
SentencePiece drops longer lines without an error. The dropped lines are not a random sample:
they are your longest, most complex examples — multi-line derivations, wide tables, dense
pages. You lose precisely the hard cases, and the loss is invisible unless you compare the
trainer's line count against your own.

**16. Corpus/deployment domain mismatch.** A tokenizer trained on clean digital text and
deployed on receipts, forms and handwriting has learned merges for the wrong distribution. Not
fatal — byte fallback covers it — but sequences run long on exactly the documents you deployed
for.

**17. `character_coverage` set too low for CJK.** At 0.9995 with a large Han inventory, the
long tail of rare-but-real characters falls out of the vocab and into byte fallback. Each such
character becomes 3 byte tokens the model must emit in the correct order — much harder than one
token, and a systematic accuracy cliff on exactly the rare characters that matter for names and
classical text.

## Class F — Integration

These have nothing to do with tokenizer quality. They are how a *correct* tokenizer still ruins
the product.

**18. The tokenizer changes after the model is trained.** Retraining the tokenizer on a slightly
different corpus produces different ids for the same pieces. The checkpoint's output layer is
now indexed against a vocab that no longer exists — output is garbage, and the cause looks like
a model bug. **Hash the `.model` file and store it in the checkpoint**; assert on load. This is
the cheapest insurance in this entire guide.

**19. Special-token id collisions.** SentencePiece reserves ids for `<unk>`/`<s>`/`</s>` and
places `user_defined_symbols` immediately after. If your training framework independently
defines pad/bos/eos ids, the two schemes can overlap or offset each other — an off-by-one across
the whole vocab, producing plausible-looking but systematically wrong text. Read ids from the
tokenizer, never hardcode them.

**20. Vocab size vs. output layer mismatch.** With `byte_fallback=True` you spend 256 slots on
byte pieces plus 3 on specials before a single learned merge. A `vocab_size` set near that floor
is mostly bytes. SentencePiece will refuse a `vocab_size` below the floor — but it will happily
accept one just above it, and that tokenizer is nearly character-level with none of the benefit.

**21. Whitespace policy mismatch between training and inference.** If GT preserves leading
whitespace (`remove_extra_whitespaces=False`) but your inference postprocessing strips it, your
eval metric measures a difference you introduced after the model was already right.

---

## How to use this page

Read it once forward to know the terrain. Then use it as the source for
[validation](08-validation.md): every failure above that can be detected from the tokenizer
artifact has a check, and every check has a test that deliberately induces the failure and
proves the check fires.

The ones that **cannot** be checked from the artifact — #5 (corpus), #7 (inference path),
#16 (domain), #18–21 (integration) — are the ones to handle with process instead: a data
pipeline assertion, an explicit decode policy, a hash in the checkpoint.
