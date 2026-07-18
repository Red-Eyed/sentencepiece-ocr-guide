# corpus

Pre-training tooling: measure how your sources disagree about encoding, then collapse the
disagreements that are not real. Runs *before* the checks in [`../checks/`](../checks/README.md)
— several of those only ever report defects that originate here.

## Why measuring comes first

The exception list beyond NFC is a judgement about *your* extractors, not a universal. Soft
hyphen is the proof: U+00AD is invisible mid-line but renders as a real hyphen where the line
breaks, and OCR ground truth is usually segmented per line. Strip it blindly and you train the
model to omit a mark that is on the page.

So [`scan.py`](scan.py) reports counts per axis *and per source*, which turns that from a
guess into a decision — 9,117 line-final versus 226 mid-line is answerable. It also usually
identifies one broken extractor rather than a diffuse problem.

## Axes

An [`Axis`](axes.py) is a name, a pure `str -> str` transform, and an `Action`:

| Action | Meaning | Severity when found |
|---|---|---|
| `COLLAPSE` | Not a difference in the text | `BLOCKER` |
| `DECIDE` | Depends on your sources — opt in explicitly | `HIGH` |
| `PRESERVE` | A real character difference | `INFO` — never a failure |

A line is affected when the transform changes it, so one definition serves both the scanner and
the canonicalizer. `PRESERVE` axes carry a transform purely for *detection*: `canonicalize`
filters on `Action`, so folding one by accident is structurally impossible rather than merely
discouraged.

`PRESERVE` counts being non-zero is the expected case — fullwidth forms showing up means you
have CJK data. A report that cannot distinguish "found variation" from "found a problem" trains
you to ignore it.

## Canonicalizing

```python
from sentencepiece_ocr_guide.corpus import canonicalizer, is_canonical

canonicalize = canonicalizer(decide=("soft_hyphen_line_final",))   # after measuring

for line in incoming:
    assert is_canonical(canonicalize(line), canonicalize)   # idempotent by construction
```

`canonicalize` applies `COLLAPSE` axes plus whichever `DECIDE` axes you name. Naming one that
does not exist raises rather than silently doing nothing — a typo there would quietly disable a
transform you believed was running.

It is idempotent, which is what makes `line == canonicalize(line)` a valid assertion at
corpus-write time. That assertion is the point: it moves the guarantee from a documented step
that every pipeline is *supposed* to call into an invariant that fails loudly.

## Verifying the stage worked

Re-scan after canonicalizing. Every `COLLAPSE` axis should read zero and every `PRESERVE` axis
should be unchanged; that difference is the proof. `tests/corpus/test_scan.py` asserts exactly
this round trip.

## What this does not cover

The corpus checklist in [docs/08-validation.md](../../docs/08-validation.md) is broader than
encoding: per-category share, and lines silently dropped by `max_sentence_length`. Those need
different inputs and are not implemented here — and note that neither is fixed by canonicalizing,
which is why remedy is tracked per finding rather than per checklist.
