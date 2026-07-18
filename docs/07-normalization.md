# Normalization — identity, NFKC, and the one that's usually right

The `normalization_rule_name='identity'` advice in [configuration](03-configuration.md) assumes
your ground truth *preserves* the distinctions NFKC would collapse. That assumption is worth
making explicit, because the real invariant is not "always use identity". It's:

> **The tokenizer's normalization must equal the normalization applied to ground truth.**
> A mismatch in either direction breaks round-trip exactness.

## What the forms are — two knobs, not four choices

Everything below depends on knowing what the four forms actually do, and the names are opaque
enough to hide it. They are built from four letters, all of which mean something:

```
N F K C
│ │ │ └── Composition — or D, Decomposition
│ │ └──── compatibility
│ └────── Form
└──────── Normalization
```

The letter that misleads is `C`. It is **not "canonical"** — it is *Composition*, and its
opposite is `D`. Canonical is what you get when the `K` is *absent*: it is marked by nothing at
all. [UAX #15](https://www.unicode.org/reports/tr15/) spells out why compatibility got an
unrelated letter — "a `K` is used to stand for *compatibility* to avoid confusion with the `C`
standing for *composition*". So the official expansions are:

| | expansion in UAX #15 |
|---|---|
| NFD | Canonical Decomposition |
| NFC | Canonical Decomposition, *followed by* Canonical Composition |
| NFKD | Compatibility Decomposition |
| NFKC | Compatibility Decomposition, *followed by* Canonical Composition |

Two things fall out of that table. The composition step is always *canonical* — there is no such
thing as compatibility composition, so the `K` changes only how text is pulled apart, never how
it is put back together. And every form decomposes first, including the two whose names end in
`C`.

The problem all four exist to solve is that Unicode allows one rendered string to be stored as
more than one sequence of code points:

```
'café' == 'café'   ->   False        <- identical on screen, different in memory
 ^NFC      ^NFD

U+0063 U+0061 U+0066 U+00E9          NFC: 4 code points, "e with acute"
U+0063 U+0061 U+0066 U+0065 U+0301   NFD: 5 code points, "e" + "combining acute"
```

Every equality test, dictionary lookup, dedup pass and tokenizer sees two different strings.
Normalization is picking one spelling and forcing everything into it.

**Knob one: composed or decomposed.** NFD splits accented characters into a base letter plus
separate combining marks; NFC squashes them back wherever a single code point exists. The same
information either way — you can convert back and forth forever.

The decompose-first step in NFC's expansion is not an implementation detail, and it is what makes
NFC the right tool for the scripts in [script considerations](04-scripts.md). Give a base two
combining marks and they can be typed in either order — `q` with a dot above and a dot below is
`U+0071 U+0307 U+0323` or `U+0071 U+0323 U+0307`, one glyph, two encodings, and composing fixes
nothing because there is no precomposed code point to compose *to*. Pulling the marks apart is
what lets them be sorted into canonical order; both spellings then land on `U+0071 U+0323 U+0307`.
That sorting step is why NFC is worth anything for Arabic and Devanagari, where stacked marks are
the norm.

**Knob two: the `K`, for compatibility.** A different kind of operation. Without it you are
saying *these are the same character, encoded differently*. With it you are saying *these are
different characters that mean roughly the same thing — collapse them anyway*:

| | NFC | NFKC | |
|---|---|---|---|
| superscript | `x²` | `x2` | exponent, gone |
| ligature `U+FB01` | `ﬁle` | `file` | split into two letters |
| fullwidth | `Ａ` | `A` | folded to ASCII |
| `U+338F` | `㎏` | `kg` | one code point becomes two |

**And `K` is a one-way door.** Canonical composition has an inverse — NFD undoes NFC exactly.
Compatibility folding has none. Nothing recovers the superscript from `x2`, because what
distinguished it is no longer in the string. "Lossy" undersells it: this is deletion, not
degradation.

Two knobs, so four forms:

| | no `K` — lossless | `K` — lossy |
|---|---|---|
| **composed** | NFC | NFKC |
| **decomposed** | NFD | NFKD |

One last property, and it is the one the next section leans on: all four are **idempotent**.
`NFC(NFC(x)) == NFC(x)`. Normalizing already-normalized text is a no-op, which is why "normalize
once at the boundary" can be enforced unconditionally rather than tracked as a thing that has or
has not happened yet to a given string.

## `identity` does not mean "un-normalized"

This is the most common misreading of the recommendation. `identity` means the *tokenizer* does
no normalization — because the normalization already happened, once, at data ingestion. It is
not a stance against normalizing. It is what you set **because** you normalized.

The rule to hold onto: **normalize once, as early as possible, and make everything downstream a
no-op.** Corpus, training labels and eval ground truth all pass through the same NFC step at the
boundary; the tokenizer then leaves canonical text alone. Normalization applied in two places is
two things that can drift apart between now and your next retraining.

Note that SentencePiece cannot do this for you even if you wanted it to. The built-in charsmaps
are `identity`, `nfkc`, `nmt_nfkc` and the two case-folding variants — **there is no NFC rule**.
Asking for one is an error, not a fallback:

```
normalization_rule_name='nfc'
# NOT_FOUND: No precompiled charsmap is found: nfc
```

So "just make the tokenizer normalize everything" is not an option that preserves fidelity. The
only always-on setting available is NFKC, and the next section is why that is the wrong trade
for OCR.

## The two failure directions

**Tokenizer normalizes more than GT.** The model cannot reproduce distinctions that are present
in the labels — it is being asked to emit something its vocabulary cannot represent. This is the
failure the `identity` recommendation exists to prevent.

**Tokenizer normalizes less than GT** (identity tokenizer, NFKC labels). Harmless for
exactness — the distinctions are already gone from the labels — but raw text entering at
inference time is unnormalized and will tokenize inconsistently with training.

**If ground truth is already NFKC-normalized:** set the tokenizer to `nmt_nfkc`/`nfkc` to match.
No fidelity is lost at the tokenizer level, because those distinctions are already absent from
the labels. Matching is what matters.

## The third case: a mixed corpus

The two directions above assume the corpus is *consistently* normalized one way or the other.
The common real-world case is neither — text assembled from several sources, some NFC, some NFD.
Here `identity` does not cause the problem, but it faithfully propagates it, and the result is
the worst failure in this page because nothing anywhere reports it.

Trained on a corpus containing both forms of `café`:

```
identity      encode(NFC) == encode(NFD)?  False     <- two labels for one rendered string
              round-trips both forms?      True      <- and the check stays green
              non-NFC pieces in vocab:     8         ('caf' + U+0301, 'ré', 'aïve', ...)

nmt_nfkc      encode(NFC) == encode(NFD)?  True      <- unified
              round-trips both forms?      False     <- NFD labels no longer survive
              non-NFC pieces in vocab:     0
```

Two things to take from that table.

**Round-trip cannot see this.** Both forms decode back to exactly what went in, because the
tokenizer is faithfully reproducing an inconsistency it was handed. Every assertion passes while
each affected grapheme trains at a fraction of its true frequency. The detectable evidence is in
the *vocabulary* — decomposed text in the corpus produces decomposed pieces — which is what the
`nfc_vocabulary` check looks at.

**NFKC is the wrong fix.** It does unify the two forms, and it is tempting for exactly that
reason. But it buys consistency by folding fullwidth, ligatures and superscripts — visual
evidence that is present in the image and that a faithful transcription system is supposed to
reproduce. NFC at ingestion buys the same consistency and costs nothing, because NFC is
canonical: `NFC(x)` renders identically to `x` by definition. That single property is the whole
difference between the two, and it is why "always normalize" is right for NFC and wrong for
NFKC.

Fix the corpus, not the tokenizer. And expect to need to: a corpus assembled from several
sources is mixed by default, not by accident. [Failure modes](09-failure-modes.md) argues this
is the single most probable defect in the guide and gives the canonicalization rule and a
worked implementation.

## Should you NFKC your ground truth at all?

For OCR, usually **no**. The compatibility folds from the first section — fullwidth, ligatures,
superscripts — erase typography the model can actually see in the image. You are training it to
ignore evidence that is right there in the pixels.

A faithful-transcription system should keep GT free of *compatibility* folding and use
`identity` — which, per the section above, means NFC-normalized ground truth rather than
un-normalized ground truth. NFKC-normalized GT is only reasonable when downstream genuinely
doesn't care — feeding a search index, or an NLP pipeline that normalizes anyway.

## The common mistake: NFKC when you meant NFC

NFKC turns both knobs. People reach for it when they only need the first — canonical
composition — and take the compatibility folding as collateral damage.

If the actual goal is fixing inconsistent combining-mark composition — the Arabic and Devanagari
NFC/NFD problem from [script considerations](04-scripts.md) — use **NFC**. It's lossless for
those scripts and leaves fullwidth and ligature distinctions alone. Apply it in your data
pipeline, not in the tokenizer, which has no NFC setting to give you.

| Form | Composes combining marks | Folds compatibility chars | Right for OCR GT? |
|---|---|---|---|
| `identity` | No | No | Yes — as the *tokenizer* setting, over an NFC corpus |
| NFC | Yes | No | Yes — this is the one you usually want, applied at ingestion |
| NFKC | Yes | Yes (lossy) | Only if downstream truly doesn't care |
